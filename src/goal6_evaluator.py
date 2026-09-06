import numpy as np

class Goal6Evaluator:
    """
    Evaluates the failure-prediction and root-cause system (Goal 6).
    Computes Detection Latency, Precision, Recall, F1 Score, False Positive Rate, and Root Cause Accuracy
    based on a series of controlled experiments.
    """
    
    def __init__(self):
        self.experiments = []
        
    def record_experiment(self, experiment_id, is_positive, t_fault=None, t_detect=None, actual_rc=None, predicted_rc=None, fp_occurred=False):
        """
        Record the results of a single experiment.
        
        :param experiment_id: str or int
        :param is_positive: bool (True if a fault was injected, False if perfectly healthy)
        :param t_fault: float or int (timestamp of fault injection, if positive)
        :param t_detect: float or int (timestamp of detection, or None if missed)
        :param actual_rc: str (the injected root cause, if positive)
        :param predicted_rc: str (the localized root cause, or None)
        :param fp_occurred: bool (did a false positive alarm trigger? only applicable for negative trials)
        """
        self.experiments.append({
            "id": experiment_id,
            "is_positive": is_positive,
            "t_fault": t_fault,
            "t_detect": t_detect,
            "actual_rc": actual_rc,
            "predicted_rc": predicted_rc,
            "fp_occurred": fp_occurred
        })
        
    def compute_metrics(self):
        """
        Calculates all six metrics.
        Returns a dictionary of the aggregated metrics.
        """
        if not self.experiments:
            return {}
            
        total = len(self.experiments)
        positive_exps = [exp for exp in self.experiments if exp["is_positive"]]
        negative_exps = [exp for exp in self.experiments if not exp["is_positive"]]
        
        tp = sum(1 for exp in positive_exps if exp["t_detect"] is not None)
        fn = sum(1 for exp in positive_exps if exp["t_detect"] is None)
        fp = sum(1 for exp in negative_exps if exp["fp_occurred"])
        tn = sum(1 for exp in negative_exps if not exp["fp_occurred"])
        
        # Detection Latency (Average across TPs)
        latencies = [exp["t_detect"] - exp["t_fault"] for exp in positive_exps if exp["t_detect"] is not None]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score = 2 * P * R / (P + R)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # False Positive Rate = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Root Cause Accuracy = Correct RCs / TPs (only among positive trials)
        correct_rcs = sum(1 for exp in positive_exps if exp["t_detect"] is not None and exp["predicted_rc"] == exp["actual_rc"])
        rc_accuracy = correct_rcs / tp if tp > 0 else 0.0
        
        return {
            "total_experiments": total,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "detection_latency": avg_latency,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "root_cause_accuracy": rc_accuracy
        }

