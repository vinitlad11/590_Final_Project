from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import platform
import subprocess
from adaptive_qa.config import path_for
from adaptive_qa.io import digest, file_hash, write_json


def config_snapshot(c):
    from copy import deepcopy
    snapshot = deepcopy({k:v for k,v in c.items() if not k.startswith("_")})
    # Preserve identity of older configs that used the original two-action tree.
    if snapshot["controller"].get("actions") == ["RETRIEVE", "DELIBERATE"]:
        snapshot["controller"].pop("actions")
    if snapshot.get("evaluation") == {"include_wait": True}:
        snapshot.pop("evaluation")
    return snapshot


def source_hashes(c):
    root = Path(c["_root"])
    return {str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src").rglob("*.py"))}


def manifest(c, out):
    root = Path(c["_root"])
    sources = source_hashes(c)
    files = {"dataset": file_hash(path_for(c, c["dataset"]["path"]))}
    corpus = path_for(c, c["retrieval"]["corpus"])
    if corpus.exists():
        files["corpus"] = file_hash(corpus)
    if c["dataset"]["split_manifest"]:
        files["split_manifest"] = file_hash(path_for(c, c["dataset"]["split_manifest"]))
    fingerprint = digest({"config":config_snapshot(c), "files":files, "source":sources})
    existing = out / "manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous["fingerprint"] != fingerprint and c["model"]["backend"] != "fixture":
            raise ValueError("Experiment inputs/code changed. Use a new experiment_id; keep the original run immutable")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "matplotlib", "PyYAML", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    row = {"experiment_id": c["experiment_id"], "timestamp": datetime.now(timezone.utc).isoformat(),
           "status": "FIXTURE_ONLY_NOT_RESEARCH" if c["model"]["backend"] == "fixture" else "REAL_MODEL_RUN",
           "fingerprint": fingerprint, "files":files, "source_hashes":sources, "git_commit":commit,
           "versions": versions, "config":config_snapshot(c)}
    write_json(existing, row)
    write_json(out / "config_snapshot.json", config_snapshot(c))
    return row


def check_manifest(c, out):
    p = out / "manifest.json"
    if not p.exists():
        raise ValueError("Run baseline/collect first to create a provenance manifest")
    previous = json.loads(p.read_text())
    if previous["config"] != config_snapshot(c):
        raise ValueError("Configuration differs from collected outputs; restore it or use a new experiment_id")
    for name, key in (("dataset", c["dataset"]["path"]), ("corpus", c["retrieval"]["corpus"])):
        if name in previous["files"] and file_hash(path_for(c, key)) != previous["files"][name]:
            raise ValueError(f"{name} changed since collection")
    return previous


def estimate(c):
    questions = c["dataset"]["limit_questions"]
    if questions is None:
        from adaptive_qa.data.load_qanta import load_questions
        questions = len(load_questions(path_for(c, c["dataset"]["path"]), c["dataset"]["text_only"])[0])
    states = questions * len(c["fractions"])
    branching = len(c["controller"]["actions"])
    nodes = sum(branching ** depth for depth in range(c["controller"]["max_steps"] + 1))
    baseline_calls = states * c["model"]["samples"]
    collection_calls = states * (1 + nodes * c["model"]["samples"])
    from adaptive_qa.controller.utility import CostModel
    from adaptive_qa.llm.model import Usage
    u = Usage(collection_calls * c["budget"]["estimated_input_tokens_per_call"],
              collection_calls * c["model"]["max_output_tokens"], collection_calls)
    return {"question_upper_bound":questions, "state_upper_bound":states, "nodes_per_state":nodes,
            "baseline_max_calls":baseline_calls, "collect_max_calls":collection_calls,
            "estimated_collect_input_tokens":u.input_tokens, "collect_output_token_ceiling":u.output_tokens,
            "estimated_collect_usd":CostModel(c["cost"], c["reward"]).dollars(u),
            "max_new_calls": c["budget"]["max_new_calls"],
            "note":"Planning estimate before cache reuse, not measured usage; input estimate is not a hard spending bound."}
