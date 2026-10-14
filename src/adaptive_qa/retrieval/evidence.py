"""Cheap lexical proxies. These features do NOT verify factual entailment."""
from adaptive_qa.evaluation.answers import normalize


def evidence_features(prediction, documents):
    scores = [d["retrieval_score"] for d in documents]
    answer = normalize(prediction)
    support = [bool(answer) and (" " + answer + " ") in (" " + normalize(d["text"]) + " ") for d in documents]
    return {"evidence_available": float(bool(documents)), "retrieval_score": scores[0] if scores else 0.0,
            "retrieval_margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
            "evidence_support": sum(support) / len(documents) if documents else 0.0,
            "top_answer_agreement": float(support[0]) if support else 0.0}
