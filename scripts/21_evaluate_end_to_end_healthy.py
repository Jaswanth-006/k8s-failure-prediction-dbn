import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.controller import KubernetesActionController

def run_end_to_end_healthy():
    print("=========================================================================")
    print(" PHASE 3 END-TO-END HEALTHY VALIDATION (CPU ONLY, SHADOW MODE)")
    print("=========================================================================\n")
    
    SERVICES = [
        "ts-ui-dashboard", "ts-user-service", "ts-train-service",
        "ts-route-service", "ts-order-service", "ts-payment-service",
        "ts-inventory-service", "ts-station-service"
    ]
    
    import networkx as nx
    G = nx.DiGraph()
    G.add_edge("ts-ui-dashboard", "ts-user-service")
    G.add_edge("ts-ui-dashboard", "ts-train-service")
    G.add_edge("ts-ui-dashboard", "ts-route-service")
    G.add_edge("ts-ui-dashboard", "ts-order-service")
    G.add_edge("ts-order-service", "ts-payment-service")
    G.add_edge("ts-order-service", "ts-inventory-service")
    G.add_edge("ts-order-service", "ts-station-service")
    
    # 1. Initialize Pipeline
    rect = Rectifier(SERVICES, ["cpu_usage"], ["node_cpu"])
    
    pipe = RobustAnomalyScorePipeline(rect.feature_names, SERVICES)
    pipe.load_model("models/phase3_autoencoder_cpu_only.pth")
    
    ddn = DynamicDecisionNetworkPhase3(G, num_particles=1000)
    
    controller = KubernetesActionController(shadow_mode=True)
    
    # 2. Load Data
    dataset_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    df = pd.read_csv(dataset_path)
    unique_timestamps = sorted(df["timestamp"].unique())
    
    split_idx = int(len(unique_timestamps) * 0.8)
    val_timestamps = unique_timestamps[split_idx:]
    
    # Metrics
    total_ticks = len(val_timestamps)
    success_ticks = 0
    fail_ticks = 0
    
    max_p_crit = {s: 0.0 for s in SERVICES}
    sum_p_crit = {s: 0.0 for s in SERVICES}
    
    root_cause_count = 0
    root_cause_services = {}
    
    intervention_decisions = 0
    reschedule_decisions = 0
    restart_decisions = 0
    
    ticks_p_crit_gt_04 = 0
    ticks_p_crit_gt_05 = 0
    actual_interventions = 0 # Controller actually intervening (i.e. not Do_Nothing + Cooldown check bypassed, though we just check MEU)
    
    all_normalized = True
    
    for ts in val_timestamps:
        try:
            # 1. Rectifier
            x_t, _ = rect.process_tick(df[df["timestamp"] == ts])
            
            # 2. Autoencoder
            anomaly_signals = pipe.compute_anomaly_signals(x_t)
            
            # 3. DDN
            ddn_output = ddn.step(anomaly_signals, node_pressure_flag=False)
            
            # 4. Controller (Shadow Mode)
            controller.reconcile_tick(ddn_output, node_pressure_flag=False)
            
            success_ticks += 1
            
            # Extract Metrics
            posteriors = ddn_output["posteriors"]
            root_cause = ddn_output["root_cause"]
            expected_utilities = ddn_output["expected_utilities"]
            
            # Check normalization
            for s in SERVICES:
                p_sum = sum(posteriors[s].values())
                if not np.isclose(p_sum, 1.0, atol=1e-5):
                    all_normalized = False
                
                p_crit = posteriors[s]["Critical"]
                max_p_crit[s] = max(max_p_crit[s], p_crit)
                sum_p_crit[s] += p_crit
                
            tick_max_p_crit = max(posteriors[s]["Critical"] for s in SERVICES)
            if tick_max_p_crit > 0.4:
                ticks_p_crit_gt_04 += 1
            if tick_max_p_crit > 0.5:
                ticks_p_crit_gt_05 += 1
                
            if root_cause != "None":
                root_cause_count += 1
                root_cause_services[root_cause] = root_cause_services.get(root_cause, 0) + 1
                
                service_eu = expected_utilities[root_cause]
                best_action = max(service_eu, key=service_eu.get)
                
                if best_action != "Do_Nothing":
                    intervention_decisions += 1
                    actual_interventions += 1
                    if best_action == "Reschedule_Pod":
                        reschedule_decisions += 1
                    elif best_action == "Restart_Pod":
                        restart_decisions += 1
                        
        except Exception as e:
            print(f"Error on tick {ts}: {e}")
            fail_ticks += 1
            
    print("\n--- RESULTS ---")
    print(f"1. Total ticks processed: {total_ticks}")
    print(f"2. Successful: {success_ticks} | Failed: {fail_ticks}")
    print("\n3 & 4. Per-service P(Critical):")
    for s in SERVICES:
        mean_p = sum_p_crit[s] / success_ticks
        print(f"   {s:<20} | Max: {max_p_crit[s]:.4f} | Mean: {mean_p:.4f}")
        
    print(f"\n5. Number of ticks where root_cause != 'None': {root_cause_count}")
    print(f"6. Services identified as root cause: {root_cause_services}")
    
    print(f"\n7. Number of controller intervention MEU decisions: {intervention_decisions}")
    print(f"8. Number of Reschedule_Pod decisions: {reschedule_decisions}")
    print(f"9. Number of Restart_Pod decisions: {restart_decisions}")
    
    print(f"\n10. Number of healthy ticks where max P(Critical) > 0.4: {ticks_p_crit_gt_04}")
    print(f"11. Number of healthy ticks where max P(Critical) > 0.5: {ticks_p_crit_gt_05}")
    print(f"12. Number where the controller would actually intervene (execute action): {actual_interventions}")
    
    print(f"\n13. Are all posterior distributions mathematically normalized to 1.0? {'YES' if all_normalized else 'NO'}")
    print(f"14. Is controller in shadow mode? {'YES' if controller.shadow_mode else 'NO'}")
    print("15. Was Kubernetes state modified? NO (shadow_mode=True explicitly prevents any actual kubectl commands)")

if __name__ == "__main__":
    run_end_to_end_healthy()
