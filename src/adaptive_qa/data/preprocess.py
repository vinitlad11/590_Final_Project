"""Question-level splits happen BEFORE prefixes are generated."""
from dataclasses import asdict, dataclass
import json
import math
import random

SPLITS = ("train", "calibration", "tune", "test")


@dataclass(frozen=True)
class Observation:
    """Only information an inference policy may observe. No gold answer."""
    question_id: str
    state_id: str
    question_text: str
    fraction_revealed: float
    requested_fraction: float
    words_revealed: int


def split_questions(questions, seed=42, ratios=(0.6, 0.1, 0.1, 0.2), manifest_path=None, require_calibration=True):
    ids = sorted(q.question_id for q in questions)
    if len(ids) < 10:
        raise ValueError("At least 10 unique questions required for the four-way split")
    if manifest_path:
        mapping = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(mapping) != set(ids) or not set(mapping.values()) <= set(SPLITS):
            raise ValueError("Split manifest must map every included qid exactly once to a supported split")
    else:
        if any(q.official_split for q in questions):
            raise ValueError("Official splits detected: supply a reviewed split_manifest; do not randomly remix them")
        random.Random(seed).shuffle(ids)
        counts = [max(1, int(len(ids) * r)) for r in ratios[:3]]
        counts.append(len(ids) - sum(counts))
        if counts[-1] <= 0:
            raise ValueError("Not enough questions for the requested split")
        mapping, start = {}, 0
        for split, count in zip(SPLITS, counts):
            mapping.update({qid: split for qid in ids[start:start + count]})
            start += count
    required = set(SPLITS) if require_calibration else {"train", "tune", "test"}
    if not required <= set(mapping.values()):
        raise ValueError("Every required split must contain questions")
    for q in questions:
        official = q.official_split.lower()
        if ("test" in official and mapping[q.question_id] != "test") or (
                ("dev" in official or "val" in official) and mapping[q.question_id] == "train"):
            raise ValueError(f"Manifest violates held-out status for {q.question_id}")
    return mapping


def make_states(question, fractions=(0.25, 0.5, 0.75, 1.0)):
    if not fractions or list(fractions) != sorted(set(fractions)) or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Fractions must increase in (0,1]")
    words, result, previous = question.question_text.split(), [], 0
    if not words:
        raise ValueError("Empty question")
    for fraction in fractions:
        count = max(1, math.ceil(len(words) * fraction))
        if count == previous:  # Avoid duplicate decisions on very short questions.
            continue
        result.append(Observation(question.question_id, f"{question.question_id}:w{count}",
                                  " ".join(words[:count]), count / len(words), fraction, count))
        previous = count
    return result


def state_records(questions, mapping, fractions):
    return [dict(asdict(s), split=mapping[q.question_id], ground_truth=list(q.answers))
            for q in questions for s in make_states(q, fractions)]
