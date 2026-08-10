import sys
import os
import pandas as pd
import networkx as nx

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rectifier import Rectifier
from src.autoencoder import AnomalyScorePipeline
from src.ddn_core import DynamicDecisionNetwork
from src.controller import KubernetesActionController

SERVICES = [
    "ts-ui-dashboard",
    "ts-user-service",
    "ts-train-service",
    "ts-route-service",
    "ts-order-service",
    "ts-payment-service",
    "ts-inventory-service",
    "ts-station-service"
]
POD_KPIS = ["cpu_usage", "memory_bytes"]
NODE_KPIS = ["node_cpu"]

def build_sample_service_graph() -> nx.DiGraph:
    """Builds TrainTicket core service call graph (DAG)."""
    G = nx.DiGraph()
    G.add_edge("ts-ui-dashboard", "ts-user-service")
    G.add_edge("ts-ui-dashboard", "ts-train-service")
    G.add_edge("ts-ui-dashboard", "ts-route-service")
    G.add_edge("ts-ui-dashboard", "ts-order-service")
    G.add_edge("ts-order-service", "ts-payment-service")
    G.add_edge("ts-order-service", "ts-inventory-service")
    G.add_edge("ts-order-service", "ts-station-service")
    return G

def main():
    print("================================================================")
    print(" PREFACE-DDN: Read-Only Inference Mode")
    print("================================================================\n")

    model_path = "models/phase1_autoencoder.pth"
    fault_dataset_path = "data/raw/faults/cpu_train_service_telemetry_dataset.csv"

    # 1. Initialize Pipeline Modules
    rectifier = Rectifier(SERVICES, POD_KPIS, NODE_KPIS)
    
    # Autoencoder initialization with feature names
    autoencoder_pipe = AnomalyScorePipeline(rectifier.feature_names, SERVICES)
    
    # Load model weights AND normalizers
    print(f"[INFO] Loading trained model from {model_path}...")
    autoencoder_pipe.load_model(model_path)
    print(f"[INFO] Model loaded successfully.")

    service_graph = build_sample_service_graph()
    ddn_engine = DynamicDecisionNetwork(service_graph)
    controller = KubernetesActionController(shadow_mode=True, cooldown_seconds=60)

    # 2. Read Dataset
    print(f"[INFO] Reading fault dataset from {fault_dataset_path}...")
    dataset_df = pd.read_csv(fault_dataset_path)
    
    unique_timestamps = sorted(dataset_df["timestamp"].unique())
    print(f"[INFO] Found {len(unique_timestamps)} unique timestamps.")

    # 3. Simulate Telemetry Ticks in Inference Mode
    print("\n--- Starting DDN Inference Loop ---\n")

    successful_ticks = 0
    failed_ticks = 0
    
    # For summary reporting
    max_anomaly = {s: -999.0 for s in SERVICES}
    sum_anomaly = {s: 0.0 for s in SERVICES}
    max_p_crit = {s: -999.0 for s in SERVICES}
    root_cause_counts = {}
    action_counts = {}

    for tick, ts in enumerate(unique_timestamps):
        try:
            df_tick = dataset_df[dataset_df["timestamp"] == ts]
            
            # Step 1: RECTIFIER
            x_t, imputed = rectifier.process_tick(df_tick)

            # Step 2: Autoencoder Anomaly Signals
            anomaly_signals = autoencoder_pipe.compute_anomaly_signals(x_t)

            # Step 3: DDN Particle Filter State Update & Utility Calculation
            ddn_output = ddn_engine.step(anomaly_signals, node_pressure_flag=False)

            # Step 4: K8s Action Controller Reconciliation
            action = controller.reconcile_tick(ddn_output, node_pressure_flag=False)

            # Metrics for summary
            successful_ticks += 1
            
            root_cause = ddn_output.get("root_cause", "None")
            posteriors = ddn_output.get("posteriors", {})
            expected_utilities = ddn_output.get("expected_utilities", {})
            
            if root_cause != "None":
                root_cause_counts[root_cause] = root_cause_counts.get(root_cause, 0) + 1
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            for s in SERVICES:
                sig = anomaly_signals[s]
                max_anomaly[s] = max(max_anomaly[s], sig)
                sum_anomaly[s] += sig
                
                # Get P(Critical) from DDN posterior
                s_posteriors = posteriors.get(s, {})
                p_crit = s_posteriors.get('Critical', 0.0)
                max_p_crit[s] = max(max_p_crit[s], p_crit)

            # Log concise info
            print(f"[{ts}] Root Cause: {root_cause}")
            if root_cause != "None":
                rc_posteriors = posteriors.get(root_cause, {})
                p_n = rc_posteriors.get('Normal', 0.0)
                p_d = rc_posteriors.get('Degrading', 0.0)
                p_c = rc_posteriors.get('Critical', 0.0)
                print(f"  {root_cause} Posteriors: P(Normal)={p_n:.2f}, P(Degrading)={p_d:.2f}, P(Critical)={p_c:.2f}")
                
            if action:
                rc_eu = expected_utilities.get(root_cause, {})
                eu_restart = rc_eu.get("Restart", 0.0)
                eu_reschedule = rc_eu.get("Reschedule", 0.0)
                delta_eu = eu_reschedule - eu_restart
                print(f"  Action: {action} (Delta EU Reschedule-Restart: {delta_eu:.2f})")
                
        except Exception as e:
            print(f"[{ts}] ERROR: {e}")
            failed_ticks += 1

    # 4. Final Summary
    print("\n================================================================")
    print(" INFERENCE SUMMARY")
    print("================================================================")
    print(f"Total Timestamps Processed: {len(unique_timestamps)}")
    print(f"Successful Ticks: {successful_ticks}")
    print(f"Failed Ticks: {failed_ticks}")
    print(f"Ticks with root_cause != 'None': {sum(root_cause_counts.values())}")
    
    most_frequent_rc = max(root_cause_counts, key=root_cause_counts.get) if root_cause_counts else "None"
    print(f"Most frequently identified root cause: {most_frequent_rc}")
    
    print("\nService-level Metrics:")
    for s in SERVICES:
        mean_sig = sum_anomaly[s] / len(unique_timestamps) if unique_timestamps else 0
        print(f"  {s:<20} | Max Anomaly: {max_anomaly[s]:>6.2f} | Mean Anomaly: {mean_sig:>6.2f} | Max P(Critical): {max_p_crit[s]:>4.2f}")

    print("\nShadow Mode Controller Decisions Observed:")
    if action_counts:
        for act, count in action_counts.items():
            print(f"  {act}: {count} times")
    else:
        print("  None")

    print("\n[CONFIRMATION]")
    print("- Model was loaded from disk (no training occurred).")
    print("- Healthy baseline was not modified.")
    print("- Model file was not modified.")
    print("- Kubernetes state was not modified (Shadow Mode = True).")
    print("- No Chaos resources were created or changed.")
    
if __name__ == "__main__":
    main()
