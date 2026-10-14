"""Agreement with the actual selected candidate, not a different majority answer."""
from adaptive_qa.evaluation.answers import normalize


def candidate_agreement(candidate, sample_answers):
    if not sample_answers:
        raise ValueError("Need at least one sample")
    candidate = normalize(candidate)
    if not candidate:
        return 0.0
    return sum(bool(normalize(x)) and normalize(x) == candidate for x in sample_answers) / len(sample_answers)
