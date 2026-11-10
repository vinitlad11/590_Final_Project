"""Logistic correctness calibration fitted only on designated calibration qids."""
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

CALIBRATION_FEATURES = ("verbal_confidence", "self_consistency", "fraction_revealed")


class _NumpyCalibratorModel:
    """Exact L2-regularized logistic regression with standard scaling when scikit-learn is unavailable."""
    def __init__(self, c=1.0, max_iter=500):
        self.c = c
        self.max_iter = max_iter

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        xs = (x - self.mean_) / self.scale_
        n, d = xs.shape
        X = np.column_stack([np.ones(n), xs])
        w = np.zeros(d + 1, dtype=float)
        reg = np.diag([0.0] + [1.0 / self.c] * d)
        for _ in range(self.max_iter):
            z = np.clip(X @ w, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = X.T @ (p - y) + reg @ w
            w_diag = np.maximum(p * (1.0 - p), 1e-6)
            hess = (X.T * w_diag) @ X + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.w_ = w
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        xs = (x - self.mean_) / self.scale_
        X = np.column_stack([np.ones(len(xs)), xs])
        z = np.clip(X @ self.w_, -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])


class CorrectnessCalibrator:
    def __init__(self, feature_names=CALIBRATION_FEATURES):
        self.feature_names = tuple(feature_names)

    def fit(self, rows):
        if not rows or any(r["split"] != "calibration" for r in rows):
            raise ValueError("Calibrator may fit only calibration rows")
        self.fit_qids = sorted({r["question_id"] for r in rows})
        x = np.array([[r[k] for k in self.feature_names] for r in rows], dtype=float)
        y = np.array([r["correct"] for r in rows], dtype=int)
        if len(set(y)) < 2:
            raise ValueError("Calibration needs both correct and incorrect outcomes; collect more data")
        if _HAS_SKLEARN:
            self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        else:
            self.model = _NumpyCalibratorModel(c=1.0, max_iter=500)
        self.model.fit(x, y)
        return self

    def predict(self, row):
        if not row.get("parse_valid", True):
            return 0.0
        return float(self.model.predict_proba([[row[k] for k in self.feature_names]])[0, 1])
