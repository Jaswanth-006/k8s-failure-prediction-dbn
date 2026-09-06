import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.goal6_evaluator import Goal6Evaluator

class TestGoal6Metrics(unittest.TestCase):
    def test_metrics_calculation(self):
        evaluator = Goal6Evaluator()
        
        # Exp 1: Positive, TP, Correct RC
        evaluator.record_experiment("exp1", is_positive=True, t_fault=100, t_detect=110, actual_rc="svc-a", predicted_rc="svc-a")

        # Exp 2: Positive, TP, Incorrect RC
        evaluator.record_experiment("exp2", is_positive=True, t_fault=200, t_detect=220, actual_rc="svc-b", predicted_rc="svc-c")

        # Exp 3: Positive, FN (No detection)
        evaluator.record_experiment("exp3", is_positive=True, t_fault=300, t_detect=None, actual_rc="svc-d", predicted_rc=None)

        # Exp 4: Positive, TP, Correct RC
        evaluator.record_experiment("exp4", is_positive=True, t_fault=400, t_detect=430, actual_rc="svc-e", predicted_rc="svc-e")

        # Exp 5: Negative, FP occurred
        evaluator.record_experiment("exp5", is_positive=False, fp_occurred=True)

        # Exp 6: Negative, No FP (TN)
        evaluator.record_experiment("exp6", is_positive=False, fp_occurred=False)

        # Exp 7: Negative, No FP (TN)
        evaluator.record_experiment("exp7", is_positive=False, fp_occurred=False)

        # Exp 8: Negative, No FP (TN)
        evaluator.record_experiment("exp8", is_positive=False, fp_occurred=False)

        metrics = evaluator.compute_metrics()

        # TPs = 3 (exp1, exp2, exp4)
        # FNs = 1 (exp3)
        # FPs = 1 (exp5)
        # TNs = 3 (exp6, exp7, exp8)

        self.assertEqual(metrics["true_positives"], 3)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 3)

        # Latency: (10 + 20 + 30) / 3 = 20
        self.assertAlmostEqual(metrics["detection_latency"], 20.0)

        # Precision: TP / (TP + FP) = 3 / (3 + 1) = 0.75
        self.assertAlmostEqual(metrics["precision"], 0.75)

        # Recall: TP / (TP + FN) = 3 / (3 + 1) = 0.75
        self.assertAlmostEqual(metrics["recall"], 0.75)

        # F1: 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
        self.assertAlmostEqual(metrics["f1_score"], 0.75)

        # FPR: FP / (FP + TN) = 1 / (1 + 3) = 0.25
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.25)
        
        # RC Accuracy: 2 correct (exp1, exp4) out of 3 TPs = 2/3
        self.assertAlmostEqual(metrics["root_cause_accuracy"], 2/3)

if __name__ == "__main__":
    unittest.main()
