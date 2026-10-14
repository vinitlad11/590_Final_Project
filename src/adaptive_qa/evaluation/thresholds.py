"""Fixed confidence baseline: abstain at a state, or wait for the next prefix."""
from collections import defaultdict
from dataclasses import asdict
from adaptive_qa.llm.model import Usage
from adaptive_qa.evaluation.metrics import selective_metrics


def fixed_threshold(rows, threshold):
    answered = [r["parse_valid"] and r["verbal_confidence"] >= threshold for r in rows]
    return {"threshold": threshold, **selective_metrics([r["correct"] for r in rows], answered)}


def incremental_wait(rows, threshold, calibrator=None):
    """A question-level baseline. Costs include every prefix actually examined."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_id"]].append(row)
    result = []
    for qid, values in sorted(groups.items()):
        usage, selected = Usage(), None
        for row in sorted(values, key=lambda r: r["fraction_revealed"]):
            # Raw threshold needs only one generation; calibrated threshold uses all samples.
            usage += Usage(**(row["usage"] if calibrator else row["generations"][0]["usage"]))
            confidence = calibrator.predict(row) if calibrator else row["verbal_confidence"]
            if row["parse_valid"] and confidence >= threshold:
                selected = row
                break
        result.append({"question_id": qid, "answered": selected is not None,
                       "prediction": selected["prediction"] if selected else None,
                       "correct": selected["correct"] if selected else 0,
                       "buzz_fraction": selected["fraction_revealed"] if selected else None,
                       "usage": asdict(usage)})
    return result
