"""Small strict configuration system; paths are relative to the project root."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "experiment_id": "fixture_smoke",
    "seed": 42,
    "dataset": {"path": "data/fixtures/questions.jsonl", "name": "FIXTURE_ONLY",
                "split_manifest": None, "text_only": True, "limit_questions": 50},
    "fractions": [0.25, 0.5, 0.75, 1.0],
    "split_ratios": [0.6, 0.1, 0.1, 0.2],
    "model": {"backend": "fixture", "name": "scripted-fixture-v1", "revision": "fixture-v1",
              "temperature": 0.2, "sample_temperature": 0.7, "samples": 3,
              "max_output_tokens": 128, "max_input_tokens": 4096,
              "timeout_seconds": 120, "device": "cpu"},
    "retrieval": {"corpus": "data/fixtures/corpus.jsonl", "top_k": 3, "max_doc_words": 180},
    "controller": {"max_steps": 1, "estimators": 80, "min_samples_leaf": 3,
                   "actions": ["RETRIEVE", "DELIBERATE"]},
    "evaluation": {"include_wait": True},
    "calibration": {"enabled": False},
    "cost": {"input_tokens_weight": 0.001, "output_tokens_weight": 0.001,
             "llm_calls_weight": 0.0, "retrieval_calls_weight": 0.0,
             "latency_seconds_weight": 0.0, "dollars_weight": 0.0,
             "input_usd_per_million": None, "output_usd_per_million": None},
    "reward": {"correct": 1.0, "incorrect": -1.0, "abstain": 0.0},
    "lambdas": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
    "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "budget": {"max_new_calls": 5000, "estimated_input_tokens_per_call": 500},
    "cache_dir": "data/cache",
    "output_dir": "results/runs",
}


def _merge(base, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    for key, value in supplied.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def validate(c):
    if c["model"]["backend"] not in {"fixture", "ollama", "huggingface"}:
        raise ValueError("backend must be fixture, ollama or huggingface")
    if c["dataset"]["name"] == "FIXTURE_ONLY" and c["model"]["backend"] != "fixture":
        raise ValueError("Use a separately named dataset for real-model runs")
    if c["model"]["backend"] == "fixture" and c["dataset"]["name"] != "FIXTURE_ONLY":
        raise ValueError("Fixture backend cannot produce benchmark results")
    actions = c["controller"]["actions"]
    if actions not in (["RETRIEVE"], ["RETRIEVE", "DELIBERATE"]):
        raise ValueError("controller.actions must be [RETRIEVE] or [RETRIEVE, DELIBERATE]")
    if not isinstance(c["evaluation"]["include_wait"], bool):
        raise ValueError("evaluation.include_wait must be boolean")
    if not isinstance(c["calibration"]["enabled"], bool):
        raise ValueError("calibration.enabled must be boolean")
    fs = c["fractions"]
    if not fs or fs != sorted(set(fs)) or not all(0 < f <= 1 for f in fs) or fs[-1] != 1:
        raise ValueError("fractions must increase in (0,1] and end at 1")
    rs = c["split_ratios"]
    if len(rs) != 4 or any(r <= 0 for r in rs) or not math.isclose(sum(rs), 1):
        raise ValueError("Need positive train/calibration/tune/test ratios summing to 1")
    for key in ("samples", "max_output_tokens", "max_input_tokens", "timeout_seconds"):
        if not isinstance(c["model"][key], int) or c["model"][key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if not isinstance(c["controller"]["max_steps"], int) or not 1 <= c["controller"]["max_steps"] <= 3:
        raise ValueError("max_steps is the number of additional computations, from 1 to 3")
    for section, key in (("controller", "estimators"), ("controller", "min_samples_leaf"),
                         ("budget", "max_new_calls"), ("budget", "estimated_input_tokens_per_call")):
        value = c[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in c["reward"].values()):
        raise ValueError("Rewards must be finite numeric values")
    if any(not math.isfinite(v) or v < 0 for v in c["lambdas"]):
        raise ValueError("lambdas must be finite and nonnegative")
    if not c["lambdas"] or not c["thresholds"] or any(not 0 <= v <= 1 for v in c["thresholds"]):
        raise ValueError("Provide lambdas and thresholds in [0,1]")
    for key, value in c["cost"].items():
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid cost {key}")
    if c["cost"]["dollars_weight"] and any(c["cost"][k] is None for k in
            ("input_usd_per_million", "output_usd_per_million")):
        raise ValueError("Dollar utility requires explicit prices")
    if not isinstance(c["seed"], int) or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(c["model"][k] < 0 or not math.isfinite(c["model"][k]) for k in
           ("temperature", "sample_temperature")):
        raise ValueError("Temperatures must be finite and nonnegative")
    if c["retrieval"]["top_k"] < 1 or c["retrieval"]["max_doc_words"] < 1:
        raise ValueError("Retrieval limits must be positive")
    if c["dataset"]["limit_questions"] is not None and c["dataset"]["limit_questions"] < 10:
        raise ValueError("Use at least 10 questions for the bounded experiment")
    if not c["experiment_id"] or Path(c["experiment_id"]).name != c["experiment_id"] or any(
            x in c["experiment_id"] for x in ("/", "\\", "..")):
        raise ValueError("experiment_id must be a simple directory name")


def load_config(path):
    path = Path(path).resolve()
    c = deepcopy(DEFAULTS)
    _merge(c, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    validate(c)
    c["_root"] = str(path.parent.parent)
    c["_config_path"] = str(path)
    return c


def path_for(c, relative):
    p = Path(relative)
    return p if p.is_absolute() else Path(c["_root"]) / p
