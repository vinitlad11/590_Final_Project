"""Prepare an untouched 100-question Stage 5 holdout set strictly disjoint from Stage 3 pilot50.

Selects 60 untouched train-fold questions from qanta.train.2021.12.20.json and 40 untouched dev-fold
questions from qanta.dev.2021.12.20.json (excluding every question_id and normalized question text
in pilot50.jsonl and selection.json), honoring preprocess.py's held-out split guardrail.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re
from prepare_qanta_public import RAW, ROOT, RELEASE, raw_questions, save_json, sha


def clean_aliases(raw_answer, raw_page):
    aliases = []
    page_clean = " ".join(str(raw_page or "").replace("_", " ").split())
    if page_clean:
        aliases.append(page_clean)
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", page_clean).strip()
        if stripped and stripped not in aliases:
            aliases.append(stripped)
    ans = str(raw_answer or "").strip()
    ans_base = re.split(r"\s*(?:\[|\(|\{|<|;|\bor\b)", ans, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    ans_base = ans_base.strip(chr(34) + chr(39) + " ")
    if ans_base and not re.search(r"\b(accept|reject|prompt)\b", ans_base, flags=re.IGNORECASE):
        if ans_base not in aliases:
            aliases.append(ans_base)
    return list(dict.fromkeys(a for a in aliases if a))


def _pick_untouched(file_path, expected_fold, n, seed, excluded_ids, excluded_texts):
    seen_ids, seen_texts = set(excluded_ids), set(excluded_texts)
    pool = []
    for row in raw_questions(file_path):
        text = row.get("text", row.get("question", ""))
        qid = row.get("qanta_id", row.get("qid", row.get("id")))
        fold = str(row.get("fold") or expected_fold).lower()
        media = any(str(row.get(k, "")).lower() in ("true", "1", "yes") for k in ("has_image", "has_audio"))
        if (
            not isinstance(text, str)
            or not text.strip()
            or qid is None
            or media
            or any(x in text.lower() for x in ("<img", "[image", "<audio", "[audio"))
            or "test" in fold
            or (expected_fold == "train" and ("dev" in fold or "val" in fold))
        ):
            continue
        norm_text = " ".join(text.split())
        full_id = f"qanta2021:{qid}"
        if full_id in seen_ids or norm_text.casefold() in seen_texts:
            continue
        aliases = clean_aliases(row.get("answer", ""), row.get("page", ""))
        if not aliases:
            continue
        seen_ids.add(full_id)
        seen_texts.add(norm_text.casefold())
        pool.append({
            "question_id": full_id,
            "question_text": norm_text,
            "answers": aliases,
            "fold": fold,
            "has_image": False,
            "has_audio": False,
        })
    pool.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['question_id']}".encode()).hexdigest())
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} eligible untouched {expected_fold} questions found; requested {n}")
    return pool[:n]


def build_holdout(
    count=100,
    seed="stage5_holdout_42",
    train_path=None,
    dev_path=None,
    pilot50_path=None,
    out_dir=None,
    overwrite=False,
):
    train_file = Path(train_path) if train_path else RAW / f"qanta.train.{RELEASE}.json"
    dev_file = Path(dev_path) if dev_path else RAW / f"qanta.dev.{RELEASE}.json"
    p50_file = Path(pilot50_path) if pilot50_path else ROOT / "data/processed/qanta2021/pilot50.jsonl"
    ready_dir = Path(out_dir) if out_dir else ROOT / "data/processed/qanta2021_stage5"
    ready_dir.mkdir(parents=True, exist_ok=True)

    excluded_ids, excluded_texts = set(), set()
    if p50_file.exists():
        for line in p50_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                excluded_ids.add(str(row["question_id"]))
                excluded_texts.add(" ".join(str(row["question_text"]).split()).casefold())
    sel_file = ROOT / "data/processed/qanta2021/selection.json"
    if sel_file.exists():
        for row in json.loads(sel_file.read_text(encoding="utf-8")).get("questions", []):
            excluded_ids.add(str(row["question_id"]))
            excluded_texts.add(" ".join(str(row["question_text"]).split()).casefold())

    n_train = int(round(count * 0.6)) if train_file.exists() else 0
    n_dev = count - n_train
    train_sel = _pick_untouched(train_file, "train", n_train, seed, excluded_ids, excluded_texts) if n_train else []
    dev_sel = _pick_untouched(dev_file, "dev", n_dev, seed, excluded_ids | {q["question_id"] for q in train_sel}, excluded_texts | {q["question_text"].casefold() for q in train_sel})
    selected = train_sel + dev_sel

    overlap = {q["question_id"] for q in selected} & excluded_ids
    if overlap:
        raise ValueError(f"Holdout leakage detected: {overlap}")

    splits = {}
    for q in train_sel:
        splits[q["question_id"]] = "train"
    n_cal = max(1, int(round(len(dev_sel) * 0.25))) if dev_sel else 0
    n_tune = max(1, int(round(len(dev_sel) * 0.25))) if len(dev_sel) > 1 else 0
    for q in dev_sel[:n_cal]:
        splits[q["question_id"]] = "calibration"
    for q in dev_sel[n_cal:n_cal + n_tune]:
        splits[q["question_id"]] = "tune"
    for q in dev_sel[n_cal + n_tune:]:
        splits[q["question_id"]] = "test"

    out_jsonl = ready_dir / f"holdout{count}.jsonl"
    out_splits = ready_dir / f"holdout{count}_splits.json"
    out_manifest = ready_dir / f"holdout{count}_manifest.json"

    jsonl_text = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in selected)
    if out_jsonl.exists() and not overwrite and out_jsonl.read_text(encoding="utf-8") != jsonl_text:
        raise ValueError(f"Frozen holdout already exists and differs: {out_jsonl}")

    out_jsonl.write_text(jsonl_text, encoding="utf-8")
    save_json(out_splits, splits)
    save_json(out_manifest, {
        "dataset": f"QANTA_2021_12_20_STAGE5_HOLDOUT_{count}",
        "seed": seed,
        "count": len(selected),
        "train_count": len(train_sel),
        "dev_count": len(dev_sel),
        "excluded_development_qids": len(excluded_ids),
        "overlap_with_stage3_pilot50": 0,
        "holdout_jsonl_sha256": sha(out_jsonl),
        "splits_sha256": sha(out_splits),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Frozen {len(selected)} untouched Stage 5 holdout questions at {out_jsonl} (0 overlap with {len(excluded_ids)} excluded dev questions)")
    return out_jsonl, out_splits, out_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", default="stage5_holdout_42")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_holdout(count=args.count, seed=args.seed, overwrite=args.overwrite)
