"""
PREFACE-DDN End-to-End Pipeline Demonstration Script
Loads/generates metric snapshots, runs RECTIFIER -> Autoencoder -> DDN Core -> K8s Controller.
Demonstrates the Intervention Utility Test (Reschedule vs Restart) decision loop.
"""

import sys
import os
import numpy as np
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
    print(" PREFACE-DDN: End-to-End Pipeline Execution Demonstration")
    print("================================================================")

    # 1. Initialize Pipeline Modules
    rectifier = Rectifier(SERVICES, POD_KPIS, NODE_KPIS)
    autoencoder_pipe = AnomalyScorePipeline(rectifier.feature_names, SERVICES)
    service_graph = build_sample_service_graph()
    ddn_engine = DynamicDecisionNetwork(service_graph)
    controller = KubernetesActionController(shadow_mode=True, cooldown_seconds=60)

    # 2. Generate Synthetic Baseline Data & Train Autoencoder
    print(f"\n[Stage 1 & 2] Training Autoencoder on healthy baseline features (Dim = {rectifier.input_width})...")
    
    # Load real telemetry dataset
    dataset_path = "data/raw/healthy/trainticket_telemetry_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return
        
    dataset_df = pd.read_csv(dataset_path)
    
    # Process all ticks through rectifier to build the training set
    baseline_features = []
    for ts, df_tick in dataset_df.groupby("timestamp"):
        x_t, imputed = rectifier.process_tick(df_tick)
        baseline_features.append(x_t)
        
    healthy_data = np.array(baseline_features, dtype=np.float32)
    autoencoder_pipe.train_autoencoder(healthy_data, epochs=10)
    print("Autoencoder trained successfully!")

    # 3. Simulate Telemetry Ticks (Healthy -> Degrading -> Fault Injected)
    print("\n[Stage 3 & 4] Running Monitoring Ticks with DDN & Intervention Utility Test...")

    # Just run a few ticks from the dataset to demonstrate the pipeline
    for tick, (ts, raw_df) in enumerate(dataset_df.groupby("timestamp")):
        if tick >= 5:
            break
            
        print(f"\n--- Monitoring Tick {tick+1} ---")

        # Step 1: RECTIFIER
        x_t, imputed = rectifier.process_tick(raw_df)

        # Step 2: Autoencoder Anomaly Signals
        anomaly_signals = autoencoder_pipe.compute_anomaly_signals(x_t)

        # Step 3: DDN Particle Filter State Update & Utility Calculation
        node_pressure_flag = (tick >= 4) # Simulate physical node pressure on tick 4+
        ddn_output = ddn_engine.step(anomaly_signals, node_pressure_flag=node_pressure_flag)

        # Step 4: K8s Action Controller Reconciliation & Intervention Utility Test
        controller.reconcile_tick(ddn_output, node_pressure_flag=node_pressure_flag)

    print("\n================================================================")
    print(" Pipeline Demonstration Completed Successfully!")
    print("================================================================")

if __name__ == "__main__":
    main()
