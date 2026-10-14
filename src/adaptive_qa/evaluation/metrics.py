"""Deterministic calibration and selective prediction metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def calibration_metrics(labels, probabilities, n_bins=10):
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or n_bins < 1:
        raise ValueError("Invalid calibration inputs")
    if not len(y):
        return {"ece": None, "brier": None, "auroc": None, "bins": []}
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Need binary labels and finite probabilities in [0,1]")
    bins, ece = [], 0.0
    indices = np.minimum((p * n_bins).astype(int), n_bins - 1)
    for b in range(n_bins):
        selected = indices == b
        if selected.any():
            confidence, accuracy = float(p[selected].mean()), float(y[selected].mean())
            count = int(selected.sum())
            bins.append({"bin": b, "count": count, "confidence": confidence, "accuracy": accuracy})
            ece += count / len(y) * abs(confidence - accuracy)
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None, "bins": bins}


def risk_coverage(labels, scores):
    """Discrete AURC = mean risk at coverages k/n, k=1..n.

    Within equal-score groups, expected risk under random tie order is used,
    so qid order cannot make a tied policy appear better.
    """
    y, p = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid risk-coverage inputs")
    if not len(y):
        return {"aurc": None, "points": []}
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    points, seen, errors = [], 0, 0.0
    for score in sorted(set(p), reverse=True):
        group = y[p == score]
        rate = float(1 - group.mean())
        for k in range(1, len(group) + 1):
            points.append({"coverage": (seen + k) / len(y), "risk": (errors + k * rate) / (seen + k),
                           "score_threshold": float(score)})
        seen += len(group)
        errors += len(group) * rate
    return {"aurc": float(np.mean([r["risk"] for r in points])), "points": points}


def selective_metrics(correct, answered):
    y, a = np.asarray(correct, dtype=float), np.asarray(answered, dtype=bool)
    if len(y) != len(a) or not len(y):
        raise ValueError("Need nonempty matching arrays")
    accuracy = float(y[a].mean()) if a.any() else None
    return {"exact_match_all_states": float(np.mean(y * a)), "coverage": float(a.mean()),
            "selective_accuracy": accuracy, "risk": 1 - accuracy if accuracy is not None else None}
