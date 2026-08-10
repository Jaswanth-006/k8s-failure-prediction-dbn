import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controller import KubernetesActionController

def run_regression_tests():
    print("--- Running Decision Policy Regression Tests ---")

    # We will subclass to track actions
    class TestController(KubernetesActionController):
        def __init__(self, shadow_mode=True, cooldown=300):
            super().__init__(shadow_mode=shadow_mode, cooldown_seconds=cooldown)
            self.executed_actions = []

        def _execute_action(self, service_name, action, delta_eu):
            # Track but rely on the base class for shadow mode logging
            super()._execute_action(service_name, action, delta_eu)
            if not self.shadow_mode:
                self.executed_actions.append((service_name, action, delta_eu))
            else:
                self.executed_actions.append((service_name, action, delta_eu, "SHADOW"))

    def create_mock_ddn(root_cause="None", p_crit=0.99):
        if root_cause == "None":
            return {
                "posteriors": {},
                "root_cause": "None",
                "expected_utilities": {}
            }
        
        # Simulate MEU behavior (Reschedule > Restart)
        eu_reschedule = 50.0
        eu_restart = 20.0
        return {
            "posteriors": {root_cause: {"Critical": p_crit}},
            "root_cause": root_cause,
            "expected_utilities": {
                root_cause: {
                    "Do_Nothing": -100.0,
                    "Reschedule_Pod": eu_reschedule,
                    "Restart_Pod": eu_restart,
                    "Scale_Out": 10.0,
                    "Traffic_Shift": 5.0
                }
            }
        }

    # Test 1: 10 consecutive anomalous ticks -> no intervention
    controller = TestController(shadow_mode=True, cooldown=300)
    print("\n[Test 1] 10 consecutive ticks...")
    for i in range(10):
        controller.reconcile_tick(create_mock_ddn("ts-train-service"))
    assert len(controller.executed_actions) == 0, "Failed: Action executed before 11 ticks."
    print("PASS: No intervention after 10 ticks.")

    # Test 2: 11 consecutive ticks -> intervention eligible
    print("\n[Test 2] 11th tick...")
    controller.reconcile_tick(create_mock_ddn("ts-train-service"))
    assert len(controller.executed_actions) == 1, "Failed: Action not executed on 11th tick."
    target, action, delta, shadow = controller.executed_actions[0]
    assert target == "ts-train-service"
    assert action == "Reschedule_Pod", "Failed: MEU action selection changed!"
    assert shadow == "SHADOW", "Failed: Shadow mode not enforced!"
    print("PASS: Intervention eligible on 11th tick, MEU=Reschedule, Shadow Mode enforced.")

    # Test 3: Cooldown blocks repeated actions
    print("\n[Test 3] Cooldown blocks repeated actions...")
    # Send another anomalous tick immediately
    controller.reconcile_tick(create_mock_ddn("ts-train-service"))
    assert len(controller.executed_actions) == 1, "Failed: Action executed during cooldown."
    print("PASS: Cooldown blocked repeated action.")

    # Test 4: Root-cause change resets persistence
    print("\n[Test 4] Root-cause change resets persistence...")
    controller = TestController(shadow_mode=True, cooldown=300)
    for i in range(10):
        controller.reconcile_tick(create_mock_ddn("ts-train-service"))
    # Change root cause on 11th tick
    controller.reconcile_tick(create_mock_ddn("ts-order-service"))
    assert len(controller.executed_actions) == 0, "Failed: Root cause change did not reset persistence."
    print("PASS: Root cause change reset persistence.")

    # Test 5: Healthy tick resets persistence
    print("\n[Test 5] Healthy tick resets persistence...")
    controller = TestController(shadow_mode=True, cooldown=300)
    for i in range(10):
        controller.reconcile_tick(create_mock_ddn("ts-train-service"))
    # Healthy tick
    controller.reconcile_tick(create_mock_ddn("None"))
    # Followed by anomalous tick
    controller.reconcile_tick(create_mock_ddn("ts-train-service"))
    assert len(controller.executed_actions) == 0, "Failed: Healthy tick did not reset persistence."
    print("PASS: Healthy tick reset persistence.")

    print("\nSUCCESS: All regression tests passed! Exact Phase 3 semantics are preserved in the shared DecisionPolicy.")

if __name__ == "__main__":
    run_regression_tests()
