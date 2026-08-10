import sys
import os
import unittest
from unittest.mock import patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision_policy import DecisionPolicy
from src.action_executor import ActionExecutor

class TestSafetyRails(unittest.TestCase):
    
    def create_ddn(self, root_cause="test-service"):
        return {
            "root_cause": root_cause,
            "posteriors": {root_cause: {"Critical": 0.99}},
            "expected_utilities": {
                root_cause: {
                    "Reschedule_Pod": 50.0,
                    "Restart_Pod": 10.0,
                    "Do_Nothing": -50.0
                }
            }
        }

    def test_shadow_mode_blocks_mutation(self):
        executor = ActionExecutor()
        
        # shadow_mode=True returns "WOULD_EXECUTE"
        res = executor.execute("Restart_Pod", "test-service", shadow_mode=True)
        self.assertEqual(res, "WOULD_EXECUTE")
        
        # shadow_mode=False raises RuntimeError in Sprint 4.3
        with self.assertRaises(RuntimeError):
            executor.execute("Restart_Pod", "test-service", shadow_mode=False)

    @patch('time.time')
    def test_persistence_survives_restart(self, mock_time):
        mock_time.return_value = 1000.0
        
        policy = DecisionPolicy(debounce_ticks=11)
        ddn = self.create_ddn()
        
        # Tick 1 to 7
        for _ in range(7):
            decision = policy.evaluate(ddn)
            self.assertEqual(decision["state"], "PENDING")
            
        self.assertEqual(decision["persistence_count"], 7)
        
        # Checkpoint export
        checkpoint = policy.export_state()
        
        # Simulate restart
        new_policy = DecisionPolicy(debounce_ticks=11)
        new_policy.import_state(checkpoint)
        
        # Tick 8 to 10
        for _ in range(3):
            decision = new_policy.evaluate(ddn)
            self.assertEqual(decision["state"], "PENDING")
            
        # Tick 11
        decision = new_policy.evaluate(ddn)
        self.assertEqual(decision["state"], "INTERVENE")
        self.assertEqual(decision["persistence_count"], 11)

    @patch('time.time')
    def test_cooldown_survives_restart(self, mock_time):
        mock_time.return_value = 1000.0
        
        policy = DecisionPolicy(debounce_ticks=1, cooldown_seconds=300)
        ddn = self.create_ddn()
        
        decision = policy.evaluate(ddn)
        self.assertEqual(decision["state"], "INTERVENE")
        
        # Record action
        policy.record_action("test-service")
        
        # Export checkpoint
        checkpoint = policy.export_state()
        
        # Restart
        new_policy = DecisionPolicy(debounce_ticks=1, cooldown_seconds=300)
        new_policy.import_state(checkpoint)
        
        # Time passes (200s, still in cooldown)
        mock_time.return_value = 1200.0
        decision = new_policy.evaluate(ddn)
        self.assertEqual(decision["state"], "COOLDOWN")
        
        # Time passes (350s, cooldown over)
        mock_time.return_value = 1350.0
        decision = new_policy.evaluate(ddn)
        self.assertEqual(decision["state"], "INTERVENE")

    @patch('time.time')
    def test_rate_limiting(self, mock_time):
        mock_time.return_value = 1000.0
        
        policy = DecisionPolicy(debounce_ticks=1, cooldown_seconds=0)
        config = {"rate_limit_max": 3, "rate_limit_window": 3600}
        ddn = self.create_ddn()
        
        # Execute 3 times
        for i in range(3):
            decision = policy.evaluate(ddn, config)
            self.assertEqual(decision["state"], "INTERVENE")
            policy.record_action("test-service")
            
        # 4th time should be blocked by rate limit
        decision = policy.evaluate(ddn, config)
        self.assertEqual(decision["state"], "RATE_LIMITED")
        
        # Export
        checkpoint = policy.export_state()
        
        # Restart
        new_policy = DecisionPolicy(debounce_ticks=1, cooldown_seconds=0)
        new_policy.import_state(checkpoint)
        
        # Still blocked
        decision = new_policy.evaluate(ddn, config)
        self.assertEqual(decision["state"], "RATE_LIMITED")
        
        # Time passes beyond window
        mock_time.return_value = 5000.0
        decision = new_policy.evaluate(ddn, config)
        self.assertEqual(decision["state"], "INTERVENE")

    def test_root_cause_change_resets_persistence(self):
        policy = DecisionPolicy(debounce_ticks=11)
        
        # Tick 1-5 for service A
        ddn_a = self.create_ddn("service-A")
        for _ in range(5):
            policy.evaluate(ddn_a)
            
        self.assertEqual(policy.current_root_cause, "service-A")
        
        # Tick 6 changes to service B
        ddn_b = self.create_ddn("service-B")
        decision = policy.evaluate(ddn_b)
        
        self.assertEqual(decision["persistence_count"], 1)
        self.assertEqual(policy.current_root_cause, "service-B")

    def test_healthy_reset(self):
        policy = DecisionPolicy(debounce_ticks=11)
        
        ddn = self.create_ddn()
        policy.evaluate(ddn)
        self.assertEqual(policy.current_root_cause, "test-service")
        
        healthy_ddn = {"root_cause": "None"}
        decision = policy.evaluate(healthy_ddn)
        
        self.assertEqual(decision["state"], "HEALTHY")
        self.assertEqual(decision["persistence_count"], 0)
        self.assertEqual(policy.current_root_cause, "None")
        
if __name__ == '__main__':
    unittest.main()
