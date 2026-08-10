import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prometheus_client import REGISTRY
from src.decision_policy import DecisionPolicy
from src.metrics import export_decision_metrics

class TestObservabilityMetrics(unittest.TestCase):
    
    def get_metric_value(self, name, labels=None):
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == name:
                    if not labels or all(sample.labels.get(k) == v for k, v in labels.items()):
                        return sample.value
        return None
        
    def test_metrics_export(self):
        policy = DecisionPolicy(debounce_ticks=1, cooldown_seconds=0)
        
        # 1. Healthy State Export
        healthy_ddn = {"root_cause": "None", "posteriors": {}}
        decision = policy.evaluate(healthy_ddn)
        export_decision_metrics(decision, shadow_mode=True, rate_limit_len=0, elapsed_time=0.01)
        
        self.assertEqual(self.get_metric_value('preface_risk_score'), 0.0)
        self.assertEqual(self.get_metric_value('preface_intervention_eligible'), 0.0)
        
        # 2. Fault State Export (WOULD_EXECUTE)
        fault_ddn = {
            "root_cause": "test-service",
            "posteriors": {"test-service": {"Critical": 0.99}},
            "expected_utilities": {
                "test-service": {
                    "Reschedule_Pod": 50.0,
                    "Restart_Pod": 20.0,
                    "Do_Nothing": -50.0
                }
            }
        }
        
        # Initial counter value
        would_execute_before = self.get_metric_value('preface_would_execute_total', {"action": "Reschedule_Pod"}) or 0.0
        executed_before = self.get_metric_value('preface_executed_total', {"action": "Reschedule_Pod"}) or 0.0
        
        decision = policy.evaluate(fault_ddn)
        export_decision_metrics(decision, shadow_mode=True, rate_limit_len=0, elapsed_time=0.02)
        
        # Verify Risk & Root Cause
        self.assertEqual(self.get_metric_value('preface_risk_score'), 0.99)
        self.assertEqual(self.get_metric_value('preface_probability_critical'), 0.99)
        self.assertEqual(self.get_metric_value('preface_root_cause_probability', {"service": "test-service"}), 0.99)
        
        # Verify Utilities
        self.assertEqual(self.get_metric_value('preface_expected_utility', {"action": "Reschedule_Pod"}), 50.0)
        self.assertEqual(self.get_metric_value('preface_delta_eu'), 30.0)
        
        # Verify State & Eligibility
        self.assertEqual(self.get_metric_value('preface_intervention_eligible'), 1.0)
        
        # Verify Shadow Mode counter incremented
        would_execute_after = self.get_metric_value('preface_would_execute_total', {"action": "Reschedule_Pod"}) or 0.0
        self.assertEqual(would_execute_after, would_execute_before + 1.0)
        
        # Verify Live Execution counter did NOT increment
        executed_after = self.get_metric_value('preface_executed_total', {"action": "Reschedule_Pod"}) or 0.0
        self.assertEqual(executed_after, executed_before)
        
        # 3. Blocked State Export
        # Simulate Cooldown
        policy.record_action("test-service")
        policy.cooldown_seconds = 300 # Reset to active cooldown
        decision = policy.evaluate(fault_ddn)
        
        blocked_before = self.get_metric_value('preface_blocked_total', {"reason": "Cooldown Active"}) or 0.0
        export_decision_metrics(decision, shadow_mode=True, rate_limit_len=1, elapsed_time=0.01)
        blocked_after = self.get_metric_value('preface_blocked_total', {"reason": "Cooldown Active"}) or 0.0
        
        self.assertEqual(blocked_after, blocked_before + 1.0)
        self.assertEqual(self.get_metric_value('preface_intervention_eligible'), 0.0)
        self.assertEqual(self.get_metric_value('preface_rate_limit_usage'), 1.0)

if __name__ == '__main__':
    unittest.main()
