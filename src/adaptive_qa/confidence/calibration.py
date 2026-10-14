"""Logistic correctness calibration fitted only on designated calibration qids."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CALIBRATION_FEATURES = ("verbal_confidence", "self_consistency", "fraction_revealed")


class CorrectnessCalibrator:
    def __init__(self, feature_names=CALIBRATION_FEATURES):
        self.feature_names = tuple(feature_names)

    def fit(self, rows):
        if not rows or any(r["split"] != "calibration" for r in rows):
            raise ValueError("Calibrator may fit only calibration rows")
        self.fit_qids = sorted({r["question_id"] for r in rows})
        x = np.array([[r[k] for k in self.feature_names] for r in rows])
        y = np.array([r["correct"] for r in rows])
        if len(set(y)) < 2:
            raise ValueError("Calibration needs both correct and incorrect outcomes; collect more data")
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        self.model.fit(x, y)
        return self

    def predict(self, row):
        if not row.get("parse_valid", True):
            return 0.0
        return float(self.model.predict_proba([[row[k] for k in self.feature_names]])[0, 1])
