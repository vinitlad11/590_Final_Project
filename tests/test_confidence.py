import unittest
from adaptive_qa.confidence.consistency import candidate_agreement
from adaptive_qa.confidence.calibration import CorrectnessCalibrator
from adaptive_qa.evaluation.metrics import calibration_metrics, risk_coverage, selective_metrics


class ConfidenceTests(unittest.TestCase):
    def test_consistency_matches_selected_answer(self):
        self.assertEqual(candidate_agreement("Einstein", ["Newton", "Newton", "Newton", "Einstein", "Curie"]), 0.2)
        self.assertEqual(candidate_agreement("Einstein", ["Einstein", "Newton", "Einstein"]), 2 / 3)
        self.assertEqual(candidate_agreement("", ["", ""]), 0)

    def test_metrics_against_hand_calculation(self):
        m = calibration_metrics([1, 0], [0.75, 0.25], n_bins=2)
        self.assertAlmostEqual(m["brier"], 0.0625)
        self.assertAlmostEqual(m["ece"], 0.25)
        self.assertEqual(m["auroc"], 1)
        self.assertEqual(calibration_metrics([1], [1])["ece"], 0)
        self.assertIsNone(calibration_metrics([1], [1])["auroc"])
        self.assertEqual(risk_coverage([1, 0], [0.9, 0.1])["aurc"], 0.25)
        self.assertEqual(risk_coverage([1, 0], [0.5, 0.5])["aurc"], 0.5)
        self.assertEqual(risk_coverage([0, 1], [0.5, 0.5])["aurc"], 0.5)
        self.assertIsNone(selective_metrics([1, 0], [False, False])["selective_accuracy"])

    def test_calibrator_rejects_leakage_and_degenerate_labels(self):
        with self.assertRaises(ValueError):
            CorrectnessCalibrator().fit([{"split": "test"}])
        with self.assertRaises(ValueError):
            CorrectnessCalibrator().fit([{"split":"calibration", "question_id":"q", "correct":1,
                                        "verbal_confidence":0.9, "self_consistency":1., "fraction_revealed":1.}])
