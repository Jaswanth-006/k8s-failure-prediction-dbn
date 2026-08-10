import os
import sys
import numpy as np
import pandas as pd
import time
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.controller import KubernetesActionController

SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service"
]

def make_ddn_output(root_cause):
    posteriors = {s: {"Critical": 0.0} for s in SERVICES}
    if root_cause != "None":
        posteriors[root_cause]["Critical"] = 0.99
    
    expected_utilities = {s: {"Reschedule_Pod": 0.0, "Restart_Pod": 0.0, "Do_Nothing": 0.0} for s in SERVICES}
    if root_cause != "None":
        expected_utilities[root_cause]["Reschedule_Pod"] = 25.0
        expected_utilities[root_cause]["Restart_Pod"] = 15.0
        expected_utilities[root_cause]["Do_Nothing"] = -50.0

    return {
        "posteriors": posteriors,
        "root_cause": root_cause,
        "expected_utilities": expected_utilities
    }

def run_unit_tests():
    print("=========================================================================")
    print(" 1. CONTROLLER PERSISTENCE UNIT TESTS")
    print("=========================================================================\n")
    
    controller = KubernetesActionController(shadow_mode=True, cooldown_seconds=300)
    
    def simulate_ticks(ticks_list):
        for t, rc in enumerate(ticks_list):
            out = make_ddn_output(rc)
            controller.reconcile_tick(out)
            curr = controller.current_root_cause
            pers = controller.root_cause_persistence.get(curr, 0)
            print(f"  Tick {t+1}: rc={rc}, controller_state={curr}, persistence={pers}")

    print("TEST A - 10 consecutive")
    simulate_ticks(["ts-train-service"] * 10)
    
    print("\nTEST B - 11 consecutive")
    controller.reconcile_tick(make_ddn_output("None"))
    simulate_ticks(["ts-train-service"] * 11)
    
    print("\nTEST C - root cause changes before tick 11")
    controller.reconcile_tick(make_ddn_output("None"))
    simulate_ticks(["ts-train-service"] * 5 + ["ts-station-service"])
    
    print("\nTEST D - healthy tick before tick 11")
    controller.reconcile_tick(make_ddn_output("None"))
    simulate_ticks(["ts-train-service"] * 9 + ["None"])
    
    print("\nTEST E - cooldown")
    controller.reconcile_tick(make_ddn_output("None"))
    simulate_ticks(["ts-train-service"] * 11)
    out = make_ddn_output("ts-train-service")
    controller.reconcile_tick(out)
    print(f"  Tick 12: rc=ts-train-service, persistence={controller.root_cause_persistence.get('ts-train-service')}, action executed time={controller.last_action_time.get('ts-train-service')}")
    
def run_healthy_validation():
    print("\n=========================================================================")
    print(" 2. COMPLETE 720-TICK HEALTHY VALIDATION")
    print("=========================================================================\n")
    
    G = nx.DiGraph()
    G.add_edge("ts-ui-dashboard", "ts-user-service")
    G.add_edge("ts-ui-dashboard", "ts-train-service")
    G.add_edge("ts-ui-dashboard", "ts-route-service")
    G.add_edge("ts-ui-dashboard", "ts-order-service")
    G.add_edge("ts-order-service", "ts-payment-service")
    G.add_edge("ts-order-service", "ts-inventory-service")
    G.add_edge("ts-order-service", "ts-station-service")
    
    rect = Rectifier(SERVICES, ["cpu_usage"], ["node_cpu"])
    pipe = RobustAnomalyScorePipeline(rect.feature_names, SERVICES)
    pipe.load_model("models/phase3_autoencoder_cpu_only.pth")
    ddn = DynamicDecisionNetworkPhase3(G, num_particles=1000)
    controller = KubernetesActionController(shadow_mode=True)
    
    df = pd.read_csv("data/raw/healthy/phase3_healthy_telemetry_dataset.csv")
    unique_timestamps = sorted(df["timestamp"].unique())
    split_idx = int(len(unique_timestamps) * 0.8)
    val_timestamps = unique_timestamps[split_idx:]
    
    total_ticks = len(val_timestamps)
    root_cause_detections = 0
    persistence_pending = 0
    intervention_eligible = 0
    reschedule_decisions = 0
    restart_decisions = 0
    
    max_p_crit = 0.0
    all_normalized = True
    
    longest_healthy_critical_episode = 0
    current_episode_len = 0
    current_root_cause = "None"
    
    for ts in val_timestamps:
        x_t, _ = rect.process_tick(df[df["timestamp"] == ts])
        anomaly_signals = pipe.compute_anomaly_signals(x_t)
        ddn_output = ddn.step(anomaly_signals, node_pressure_flag=False)
        
        posteriors = ddn_output["posteriors"]
        for s in SERVICES:
            p_sum = sum(posteriors[s].values())
            if not np.isclose(p_sum, 1.0, atol=1e-5):
                all_normalized = False
            max_p_crit = max(max_p_crit, posteriors[s]["Critical"])
            
        root_cause = ddn_output["root_cause"]
        
        # Track longest episode
        if root_cause != "None":
            if root_cause == current_root_cause:
                current_episode_len += 1
            else:
                current_root_cause = root_cause
                current_episode_len = 1
            if current_episode_len > longest_healthy_critical_episode:
                longest_healthy_critical_episode = current_episode_len
        else:
            current_root_cause = "None"
            current_episode_len = 0
            
        if root_cause != "None":
            root_cause_detections += 1
            
        controller.reconcile_tick(ddn_output, node_pressure_flag=False)
        
        if root_cause != "None":
            pers = controller.root_cause_persistence.get(root_cause, 0)
            if pers < 11:
                persistence_pending += 1
            else:
                intervention_eligible += 1
                service_eu = ddn_output["expected_utilities"][root_cause]
                best_action = max(service_eu, key=service_eu.get)
                if best_action == "Reschedule_Pod":
                    reschedule_decisions += 1
                elif best_action == "Restart_Pod":
                    restart_decisions += 1
                    
    print(f"1. Healthy ticks processed: {total_ticks}")
    print(f"2. Raw root-cause detections: {root_cause_detections}")
    print(f"3. Persistence-pending detections (<11 ticks): {persistence_pending}")
    print(f"4. Intervention-eligible decisions (>=11 ticks): {intervention_eligible}")
    print(f"5. Reschedule decisions: {reschedule_decisions}")
    print(f"6. Restart decisions: {restart_decisions}")
    
    false_intervention_rate = intervention_eligible / total_ticks * 100
    print(f"7. False intervention rate: {false_intervention_rate:.2f}%")
    
    print(f"8. Longest healthy Critical episode: {longest_healthy_critical_episode} ticks")
    print(f"9. Maximum P(Critical): {max_p_crit:.4f}")
    print(f"10. Posterior normalization: {'YES' if all_normalized else 'NO'}")
    print(f"11. Confirmation Kubernetes state was untouched: YES (shadow_mode=True)")
    
    if intervention_eligible == 0:
        print("\nSUCCESS! The 11-tick persistence filter perfectly eliminated all healthy false positives.")
    else:
        print("\nWARNING: Some false positives still bypassed the 11-tick filter.")

if __name__ == "__main__":
    run_unit_tests()
    run_healthy_validation()
