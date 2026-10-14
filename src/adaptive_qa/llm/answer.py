"""Fail-closed structured answer parsing; malformed generations remain recorded."""
from dataclasses import dataclass
import json
import math
import re


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    confidence: float | None
    valid: bool
    error: str | None = None


def parse_answer(raw, confidence=True):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or not isinstance(obj.get("answer"), str) or not obj["answer"].strip():
            raise ValueError("answer must be a nonempty string")
        value = obj.get("confidence") if confidence else None
        if confidence and (isinstance(value, bool) or not isinstance(value, (float, int)) or
                           not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("confidence must be a finite number from 0 to 1")
        return ParsedAnswer(obj["answer"].strip(), float(value) if confidence else None, True)
    except (ValueError, TypeError) as e:
        return ParsedAnswer("", 0.0 if confidence else None, False, str(e))
