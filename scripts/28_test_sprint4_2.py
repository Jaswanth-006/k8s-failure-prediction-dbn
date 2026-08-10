import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision_policy import DecisionPolicy

def test_sprint4_2():
    policy = DecisionPolicy(debounce_ticks=1, cooldown_seconds=0)

    print("--- Sprint 4.2 Utility Decision Tests ---")

    def create_ddn(eu_reschedule, eu_restart, eu_scale):
        return {
            "root_cause": "test-service",
            "posteriors": {"test-service": {"Critical": 0.99}},
            "expected_utilities": {
                "test-service": {
                    "Reschedule_Pod": eu_reschedule,
                    "Restart_Pod": eu_restart,
                    "Scale_Out": eu_scale,
                    "Traffic_Shift": -10.0,
                    "Do_Nothing": -50.0
                }
            }
        }

    # Test 1: Tie-breaking behavior
    # Reschedule and Restart have exact same EU. Precedence says Reschedule > Restart.
    print("\n[Test 1] Tie-breaking behavior...")
    ddn = create_ddn(50.0, 50.0, 20.0)
    decision = policy.evaluate(ddn)
    assert decision["best_action"] == "Reschedule_Pod", f"Tie-break failed, selected {decision['best_action']}"
    print("PASS: Tie-breaking correctly selected Reschedule over Restart.")

    # Test 2: Node-host degradation (Reschedule has higher EU)
    print("\n[Test 2] Node-host degradation...")
    ddn = create_ddn(80.0, 30.0, 20.0)
    decision = policy.evaluate(ddn)
    assert decision["best_action"] == "Reschedule_Pod"
    print("PASS: Selected Reschedule_Pod when EU was highest.")

    # Test 3: Targeted degradation (Restart has higher EU)
    print("\n[Test 3] Targeted degradation...")
    ddn = create_ddn(30.0, 80.0, 20.0)
    decision = policy.evaluate(ddn)
    assert decision["best_action"] == "Restart_Pod"
    print("PASS: Selected Restart_Pod when EU was highest.")

    # Test 4: Action filtering (Reschedule disabled, despite highest EU)
    print("\n[Test 4] Action filtering...")
    ddn = create_ddn(80.0, 30.0, 20.0)
    config = {"enabled_actions": ["Restart_Pod", "Scale_Out", "Do_Nothing"]}
    decision = policy.evaluate(ddn, config)
    assert decision["best_action"] == "Restart_Pod", f"Failed filtering, selected {decision['best_action']}"
    print("PASS: Disabled action was correctly ignored.")

    # Test 5: All actions disabled -> Do_Nothing
    print("\n[Test 5] All actions disabled...")
    ddn = create_ddn(80.0, 80.0, 80.0)
    config = {"enabled_actions": []}
    decision = policy.evaluate(ddn, config)
    assert decision["best_action"] == "Do_Nothing", f"Failed empty filtering, selected {decision['best_action']}"
    print("PASS: Empty enabled_actions safely defaulted to Do_Nothing.")
    
    # Test 6: Delta EU Calculation
    print("\n[Test 6] Delta EU Calculation...")
    ddn = create_ddn(50.0, 20.0, 0.0)
    decision = policy.evaluate(ddn)
    assert decision["delta_eu"] == 30.0, f"Failed delta EU calculation, got {decision['delta_eu']}"
    print("PASS: Delta EU correctly exported.")

    print("\nSUCCESS: All Sprint 4.2 Utility Decision Tests Passed!")

if __name__ == "__main__":
    test_sprint4_2()
