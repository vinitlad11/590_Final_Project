"""Import organizer Tossup CSV or canonical JSON(L), rejecting ambiguous answers."""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from adaptive_qa.io import digest, read_jsonl


@dataclass(frozen=True)
class Question:
    question_id: str
    question_text: str
    answers: tuple[str, ...]
    category: str = "unknown"
    official_split: str = ""


def _true(value):
    if value in (None, "", False, 0):
        return False
    if value in (True, 1):
        return True
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid modality flag: {value!r}")


def _answers(row):
    raw = row.get("clean_answers") or row.get("answers") or row.get("aliases")
    if raw:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError("clean_answers/aliases must be a JSON array of accepted strings") from e
        if not isinstance(raw, list) or not raw or not all(isinstance(x, str) and x.strip() for x in raw):
            raise ValueError("Expected nonempty answer strings; do not parse accept/reject directives automatically")
        return tuple(dict.fromkeys(x.strip() for x in raw))
    answer = row.get("answer") or row.get("ground_truth")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Need clean_answers, answers, aliases, or canonical answer. answerline alone is unsafe")
    if any(t in answer.lower() for t in ("accept ", "reject ", "prompt on", "[", "{")):
        raise ValueError("Complex answerline requires a reviewed aliases list")
    return (answer.strip(),)


def load_questions(path, text_only=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DATA_PENDING: place the verified Tossup export at {path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = raw.get("questions", []) if isinstance(raw, dict) else raw
    else:
        raise ValueError("Supported datasets: .csv, .json, .jsonl")
    questions, seen_ids, seen_texts = [], set(), {}
    skipped = 0
    for i, row in enumerate(rows, 1):
        if text_only and (_true(row.get("has_image")) or _true(row.get("has_audio"))):
            skipped += 1
            continue
        qid = row.get("qid", row.get("question_id", row.get("qanta_id", row.get("id"))))
        text = row.get("question", row.get("question_text", row.get("text")))
        if qid is None or not str(qid).strip() or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {i}: missing question ID/text")
        qid, text = str(qid), " ".join(text.split())
        if text_only and any(marker in text.lower() for marker in ("<img", "[image", "<audio", "[audio")):
            skipped += 1
            continue
        text_key = digest(text.casefold())
        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id {qid}; namespace packet IDs before merging")
        if text_key in seen_texts:
            raise ValueError(f"Duplicate question text: {qid} and {seen_texts[text_key]}; deduplicate before splitting")
        seen_ids.add(qid)
        seen_texts[text_key] = qid
        questions.append(Question(qid, text, _answers(row), str(row.get("category") or "unknown"),
                                  str(row.get("split", row.get("fold", "")) or "")))
    if not questions:
        raise ValueError("No usable text-only Tossups")
    return questions, {"input_rows": len(rows), "included": len(questions), "skipped_multimodal": skipped}


def question_records(questions):
    return [asdict(q) for q in questions]
