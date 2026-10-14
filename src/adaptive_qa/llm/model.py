"""Provider-independent generation, measured usage, and durable response caching."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import re
import time
import urllib.request
from urllib.parse import urlparse
from adaptive_qa.io import digest, write_json
from adaptive_qa.llm.prompts import PROMPT_VERSION, build_prompt


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    latency_seconds: float = 0.0

    def __post_init__(self):
        for name in ("input_tokens", "output_tokens", "llm_calls", "retrieval_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative measured integer")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and nonnegative")

    def __add__(self, other):
        return Usage(**{k: v + getattr(other, k) for k, v in asdict(self).items()})


@dataclass(frozen=True)
class Generation:
    raw: str
    usage: Usage
    model: str
    timestamp: str
    cache_hit: bool = False
    usage_kind: str = "measured_model_tokens"


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt, temperature, seed, confidence=True):
        raise NotImplementedError

    def answer(self, question_text, temperature=0.0, seed=0):
        return self.generate(build_prompt(question_text, confidence=False), temperature, seed, False)

    def answer_with_confidence(self, question_text, temperature=0.0, seed=0):
        return self.generate(build_prompt(question_text), temperature, seed)

    def deliberate(self, question_text, previous_answer, temperature=0.0, seed=0):
        return self.generate(build_prompt(question_text, "DELIBERATE", previous_answer), temperature, seed)


class FixtureLLM(BaseLLM):
    """Scripted arithmetic behavior. Not an LLM and never used for benchmark claims.

    Reads only the prompt, not dataset labels. Deliberate mistakes exercise the
    pipeline. Counts are whitespace units, never actual language-model tokens.
    """
    def generate(self, prompt, temperature, seed, confidence=True):
        start = time.perf_counter()
        data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
        text = data["revealed_question"] + " " + " ".join(d["text"] for d in data.get("evidence", []))
        base, increment = re.search(r"base is (\d+)", text), re.search(r"increment is (\d+)", text)
        value = int(base[1]) + int(increment[1]) if base and increment else int(base[1]) if base else -999
        bucket = int(digest([prompt, seed])[:8], 16) % 10
        # Deterministic simulation variations for integration testing only.
        if base and increment and bucket < (1 if "Reconsider" in prompt else 3):
            value += 1
        conf = (0.65 + bucket * 0.03) if base and increment else 0.15 + bucket * 0.045
        obj = {"answer": str(value)}
        if confidence:
            obj["confidence"] = min(conf, 0.99)
        raw = json.dumps(obj)
        return Generation(raw, Usage(len(prompt.split()), len(raw.split()), 1, 0,
                                     time.perf_counter() - start), "scripted-fixture-v1",
                          datetime.now(timezone.utc).isoformat(), usage_kind="FIXTURE_WHITESPACE_UNITS")


class OllamaLLM(BaseLLM):
    """Native non-streaming Ollama API. The server/model must already be installed."""
    def __init__(self, config):
        self.config = config
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not self.host.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_HOST must include http:// or https://")
        url = urlparse(self.host)
        if url.username or url.password or url.query:
            raise ValueError("Keep credentials out of OLLAMA_HOST; use LLM_API_KEY")
        headers = {"Authorization": "Bearer " + os.environ["LLM_API_KEY"]} if os.getenv("LLM_API_KEY") else {}
        request = urllib.request.Request(self.host + "/api/tags", headers=headers)
        with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
            tags = json.load(response)["models"]
        matches = [m for m in tags if m.get("name") == config["name"] or m.get("model") == config["name"]]
        if len(matches) != 1 or not matches[0].get("digest"):
            raise ValueError("Model name must exactly match an installed Ollama tag with a digest")
        self.identity = {"model_digest": matches[0]["digest"], "details": matches[0].get("details", {})}
        if config["revision"] not in (None, "auto", self.identity["model_digest"]):
            raise ValueError("Ollama model digest differs from model.revision; pin the correct digest or use auto for a pilot")

    def generate(self, prompt, temperature, seed, confidence=True):
        payload = {"model": self.config["name"], "prompt": prompt, "stream": False, "format": "json",
                   "options": {"temperature": temperature, "seed": seed,
                               "num_predict": self.config["max_output_tokens"],
                               "num_ctx": self.config["max_input_tokens"] + self.config["max_output_tokens"]}}
        headers = {"Content-Type": "application/json"}
        if os.getenv("LLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
        request = urllib.request.Request(self.host + "/api/generate", json.dumps(payload).encode(), headers)
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
            obj = json.load(response)
        latency = time.perf_counter() - start
        if not obj.get("done") or "prompt_eval_count" not in obj or "eval_count" not in obj:
            raise RuntimeError("Incomplete response or missing usage; do not invent counts")
        if obj["prompt_eval_count"] >= self.config["max_input_tokens"]:
            raise RuntimeError("Input at/above context guard; increase context and inspect truncation before rerunning")
        return Generation(obj["response"], Usage(obj["prompt_eval_count"], obj["eval_count"], 1, 0, latency),
                          obj.get("model", self.config["name"]) + "@" + self.identity["model_digest"], datetime.now(timezone.utc).isoformat())


class HuggingFaceLLM(BaseLLM):
    """Optional local backend. GPU_PENDING until tested with an actual checkpoint."""
    def __init__(self, config):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Install optional dependencies: python -m pip install -e '.[gpu]'") from e
        if not config["revision"] or config["revision"] in {"main", "fixture-v1", "SET_COMMIT_SHA"}:
            raise ValueError("Pin model.revision to an actual Hugging Face commit SHA")
        self.torch, self.config = torch, config
        device = config["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU_PENDING: CUDA is not available; choose cpu for a sufficiently small model")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(config["name"], revision=config["revision"], trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(config["name"], revision=config["revision"],
                            trust_remote_code=False, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        self.model.to(device).eval()
        self.identity = {"revision":config["revision"], "device":device}

    def runtime_metrics(self):
        if self.config["device"] != "cuda":
            return {"device": self.config["device"]}
        return {"device": "cuda", "gpu_name": self.torch.cuda.get_device_name(0),
                "gpu_total_bytes": self.torch.cuda.get_device_properties(0).total_memory,
                "peak_torch_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": self.torch.cuda.max_memory_reserved(),
                "note": "PyTorch allocator peaks for this process, not total GPU process memory"}

    def generate(self, prompt, temperature, seed, confidence=True):
        from transformers import set_seed
        set_seed(seed)
        start = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.config["device"]) for k, v in inputs.items()}
        n_input = inputs["input_ids"].shape[-1]
        if n_input > self.config["max_input_tokens"]:
            raise ValueError("Prompt exceeds max_input_tokens; no silent truncation is allowed")
        args = {"max_new_tokens": self.config["max_output_tokens"], "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            args["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **args)[0, n_input:]
        if self.config["device"] == "cuda":
            self.torch.cuda.synchronize()
        raw = self.tokenizer.decode(output, skip_special_tokens=True)
        return Generation(raw, Usage(n_input, len(output), 1, 0, time.perf_counter() - start),
                          self.config["name"] + "@" + self.config["revision"], datetime.now(timezone.utc).isoformat())


class CachedLLM(BaseLLM):
    def __init__(self, backend, config, cache_dir, max_new_calls=5000):
        self.backend, self.config = backend, config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_calls, self.new_calls, self.hits = max_new_calls, 0, 0
        self.incurred = Usage()

    def generate(self, prompt, temperature, seed, confidence=True):
        key = digest({"schema": 1, "prompt_version": PROMPT_VERSION, "model": self.config,
                      "prompt": prompt, "temperature": temperature, "seed": seed, "confidence": confidence,
                      "endpoint": getattr(self.backend, "host", None), "resolved_backend":getattr(self.backend, "identity", None)})
        path = self.cache_dir / (key + ".json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            row["usage"] = Usage(**row["usage"])
            self.hits += 1
            return replace(Generation(**row), cache_hit=True)
        if self.new_calls >= self.max_new_calls:
            raise RuntimeError("Call budget exhausted; cached responses are saved. Review estimate before increasing budget")
        self.new_calls += 1  # Failed attempts still consume the request budget. Never retry invisibly.
        result = self.backend.generate(prompt, temperature, seed, confidence)
        self.incurred += result.usage
        write_json(path, asdict(result))
        return result

    def usage(self):
        return {"new_calls": self.new_calls, "cache_hits": self.hits, "incurred_usage": asdict(self.incurred),
                "model_identity":getattr(self.backend, "identity", {"backend":"fixture"}),
                "backend_runtime":self.backend.runtime_metrics() if hasattr(self.backend, "runtime_metrics") else {}}


def make_llm(c):
    from adaptive_qa.config import path_for
    mc = c["model"]
    backend = {"fixture": FixtureLLM, "ollama": OllamaLLM, "huggingface": HuggingFaceLLM}[mc["backend"]]
    instance = backend() if mc["backend"] == "fixture" else backend(mc)
    return CachedLLM(instance, mc, path_for(c, c["cache_dir"]) / "llm", c["budget"]["max_new_calls"])
