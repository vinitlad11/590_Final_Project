"""Collect answers and consistency observations; save raw evidence and sample cost."""
from dataclasses import asdict
from adaptive_qa.config import path_for
from adaptive_qa.data.load_qanta import load_questions
from adaptive_qa.data.preprocess import make_states, split_questions, state_records
from adaptive_qa.evaluation.answers import exact_match, answer_f1
from adaptive_qa.confidence.consistency import candidate_agreement
from adaptive_qa.io import digest, write_json, write_jsonl
from adaptive_qa.llm.answer import parse_answer
from adaptive_qa.llm.model import Usage, make_llm


def prepare(c, out):
    questions, audit = load_questions(path_for(c, c["dataset"]["path"]), c["dataset"]["text_only"])
    # Sampling uses content-stable qid hashing; input file order cannot affect selection.
    questions = sorted(questions, key=lambda q: digest([c["seed"], q.question_id]))
    limit = c["dataset"]["limit_questions"]
    if limit:
        questions = questions[:limit]
    manifest = c["dataset"]["split_manifest"]
    mapping = split_questions(questions, c["seed"], c["split_ratios"], path_for(c, manifest) if manifest else None,
                              require_calibration=c["calibration"]["enabled"])
    audit["selected_questions"] = len(questions)
    write_json(out / "data_audit.json", audit)
    write_json(out / "split_manifest.json", mapping)
    write_jsonl(out / "states.jsonl", state_records(questions, mapping, c["fractions"]))
    return questions, mapping


def seed_for(c, state_id, path, sample):
    return int(digest([c["seed"], state_id, path, sample])[:8], 16) % (2**31)


def collect_candidate(llm, c, state, action="ANSWER", previous="", documents=(), history=""):
    from adaptive_qa.llm.prompts import build_prompt
    prompt = build_prompt(state.question_text, action, previous, documents)
    parsed, generations, usage = [], [], Usage()
    for i in range(c["model"]["samples"]):
        temperature = c["model"]["temperature"] if i == 0 else c["model"]["sample_temperature"]
        generation = llm.generate(prompt, temperature, seed_for(c, state.state_id, history, i))
        record = asdict(generation)
        record.pop("cache_hit")  # Operational cache hits belong in the session ledger, not scientific row identity.
        generations.append(record)
        parsed.append(parse_answer(generation.raw))
        usage += generation.usage
    first = parsed[0]
    return {"prediction": first.answer, "verbal_confidence": first.confidence,
            "self_consistency": candidate_agreement(first.answer, [p.answer for p in parsed]),
            "repeat_agreement": candidate_agreement(first.answer, [p.answer for p in parsed[1:]]) if len(parsed) > 1 else None,
            "signal_kind": "consistency_not_probability",
            "fraction_revealed": state.fraction_revealed, "parse_valid": first.valid,
            "parse_error": first.error, "usage": asdict(usage), "generations": generations,
            "usage_kind": generations[0]["usage_kind"], "history": history}


def run_baseline(c, out):
    questions, mapping = prepare(c, out)
    llm, rows = make_llm(c), []
    for q in questions:
        for state in make_states(q, c["fractions"]):
            candidate = collect_candidate(llm, c, state)
            rows.append({**asdict(state), **candidate, "ground_truth": list(q.answers),
                         "correct": exact_match(candidate["prediction"], q.answers),
                         "answer_f1": answer_f1(candidate["prediction"], q.answers), "split": mapping[q.question_id],
                         "experiment_id": c["experiment_id"], "model": c["model"]["name"],
                         "temperature": c["model"]["temperature"]})
    write_jsonl(out / "baseline_predictions.jsonl", rows)
    # A flat file is convenient to inspect without reading nested generations.
    import csv
    flat = [{k: r[k] for k in ("question_id", "state_id", "fraction_revealed", "prediction", "verbal_confidence", "correct")}
            | r["generations"][0]["usage"] | {"usage_kind": r["usage_kind"], "split":r["split"],
            "ground_truth":" | ".join(r["ground_truth"]), "model":r["generations"][0]["model"],
            "temperature":r["temperature"], "timestamp":r["generations"][0]["timestamp"],
            "experiment_id":r["experiment_id"]} for r in rows]
    with (out / "baseline_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    write_json(out / "baseline_collection_usage.json", llm.usage())
    return rows


def fit_baseline_calibration(rows, out, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95), include_wait=True):
    from adaptive_qa.confidence.calibration import CorrectnessCalibrator
    from adaptive_qa.evaluation.metrics import calibration_metrics
    from adaptive_qa.evaluation.thresholds import fixed_threshold, incremental_wait
    write_json(out / "baseline_thresholds_tune.json", [fixed_threshold(
        [r for r in rows if r["split"] == "tune"], t) for t in thresholds])
    if include_wait:
        write_json(out / "baseline_wait_tune.json", {str(t): incremental_wait(
            [r for r in rows if r["split"] == "tune"], t) for t in thresholds})
    calibrator = CorrectnessCalibrator().fit([r for r in rows if r["split"] == "calibration"])
    report = {}
    for split in ("tune", "test"):
        subset = [r for r in rows if r["split"] == split]
        report[split] = {"verbal": calibration_metrics([r["correct"] for r in subset], [r["verbal_confidence"] for r in subset]),
                         "calibrated": calibration_metrics([r["correct"] for r in subset], [calibrator.predict(r) for r in subset])}
    report["status"] = rows[0]["usage_kind"]
    report["calibration_qids"] = calibrator.fit_qids
    write_json(out / "baseline_calibration.json", report)
    return calibrator
