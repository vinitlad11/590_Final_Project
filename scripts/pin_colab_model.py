"""Resolve immutable Hugging Face commit SHAs for Colab config files before inference."""
from pathlib import Path
import argparse
import json
import re
import yaml
from huggingface_hub import HfApi


def pin_configs(root, config_paths=None, model_name=None, force=False):
    root = Path(root).resolve()
    if config_paths:
        paths = [root / p if not Path(p).is_absolute() else Path(p) for p in config_paths]
    else:
        candidates = (
            "configs/colab_qanta2021_pilot10.yaml",
            "configs/colab_qanta2021_pilot50.yaml",
            "configs/stage03_followup10.yaml",
            "configs/stage03_pilot50.yaml",
        )
        paths = [root / p for p in candidates if (root / p).exists()]
    if not paths:
        raise ValueError("No matching config files found")
    configs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths]
    target_model = model_name or configs[0]["model"]["name"]
    existing_revs = {
        c["model"]["revision"]
        for c in configs
        if c["model"]["name"] == target_model
        and c["model"].get("revision") not in (None, "", "SET_COMMIT_SHA", "auto", "main")
        and not force
    }
    if len(existing_revs) > 1:
        raise ValueError("Configs disagree on model revision; pass --force to re-pin")
    revision = next(iter(existing_revs)) if existing_revs else HfApi().model_info(target_model).sha
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("Expected a 40-character immutable checkpoint SHA")
    for path, c in zip(paths, configs):
        if model_name:
            c["model"]["name"] = target_model
        if c["model"].get("revision") != revision or model_name:
            run = root / "results/runs" / c["experiment_id"] / "manifest.json"
            if run.exists() and not force:
                raise ValueError(f"Run already exists for {c['experiment_id']}; change experiment_id or pass --force")
            c["model"]["revision"] = revision
            path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    record = root / "results/logs/colab_model_pin.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"model": target_model, "revision": revision, "configs": [str(p.relative_to(root)) for p in paths]}, indent=2) + "\n", encoding="utf-8")
    return revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", help="Config files to pin")
    parser.add_argument("--model", help="Optional Hugging Face model ID to set and pin (e.g. Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--force", action="store_true", help="Update revision even if already pinned")
    args = parser.parse_args()
    print(pin_configs(Path(__file__).resolve().parents[1], args.configs, args.model, args.force))
