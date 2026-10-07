"""Conservative alias-based matching, not official QANTA adjudication."""
from collections import Counter
import re
import unicodedata


def normalize(text, remove_articles=True):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    # Preserve symbols: C++ != C, -3 != 3. Strip ordinary punctuation only.
    text = "".join(" " if unicodedata.category(c).startswith("P") and c not in "+-#" else c
                   for c in text)
    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, aliases):
    p = normalize(prediction)
    return int(bool(p) and any(p == normalize(a) for a in aliases if normalize(a)))


def answer_f1(prediction, aliases):
    p = Counter(normalize(prediction).split())
    best = 0.0
    for alias in aliases:
        a = Counter(normalize(alias).split())
        overlap = sum((p & a).values())
        if overlap and p and a:
            precision, recall = overlap / sum(p.values()), overlap / sum(a.values())
            best = max(best, 2 * precision * recall / (precision + recall))
    return best
