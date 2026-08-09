import sys
import os
import random
import numpy as np
import pandas as pd
import torch

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rectifier import Rectifier
from src.autoencoder import AnomalyScorePipeline

def set_seed(seed: int = 42):
    """Ensure reproducibility for the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    print("================================================================")
    print(" PREFACE-DDN: Phase 2 Autoencoder Training")
    print("================================================================\n")

    # Define paths
    dataset_path = "data/raw/healthy/trainticket_telemetry_dataset.csv"
    model_output_path = "models/phase1_autoencoder.pth"

    print(f"[INFO] Looking for dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Required dataset not found: {dataset_path}")
        sys.exit(1)

    # 1. Initialize Rectifier
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
    
    rectifier = Rectifier(SERVICES, POD_KPIS, NODE_KPIS)
    print(f"[INFO] Rectifier initialized. Expected feature vector width: {rectifier.input_width}")

    # 2. Process Dataset into Vectors
    dataset_df = pd.read_csv(dataset_path)
    # Ensure stable ordering of timestamps
    unique_timestamps = sorted(dataset_df["timestamp"].unique())
    print(f"[INFO] Found {len(unique_timestamps)} unique timestamps.")

    if len(unique_timestamps) < 2:
        print("[ERROR] Not enough timestamps for a train/validation split.")
        sys.exit(1)

    # Convert timestamps into fixed-width feature vectors
    X_all = []
    for ts in unique_timestamps:
        df_tick = dataset_df[dataset_df["timestamp"] == ts]
        x_t, imputed = rectifier.process_tick(df_tick)
        X_all.append(x_t)

    X_all = np.array(X_all, dtype=np.float32)

    # 3. Validation Split (80/20)
    split_idx = int(len(unique_timestamps) * 0.8)
    X_train = X_all[:split_idx]
    X_val = X_all[split_idx:]
    
    print(f"\n[INFO] Split dataset sequentially (NO shuffling of timestamps):")
    print(f"       - Training:   {len(X_train)} timestamps")
    print(f"       - Validation: {len(X_val)} timestamps")
    print(f"       (Note: This is a small pipeline-validation split for integration testing, not a statistically significant model evaluation due to small N.)\n")

    # 4. Train Autoencoder
    print(f"[INFO] Instantiating Autoencoder...")
    autoencoder_pipe = AnomalyScorePipeline(rectifier.feature_names, SERVICES)
    
    print(f"[INFO] Training Autoencoder on healthy baseline features (Dim={rectifier.input_width})...")
    # Loss is logged implicitly or we can just let it run
    autoencoder_pipe.train_autoencoder(X_train, epochs=50, batch_size=16)
    print("[INFO] Autoencoder trained successfully!")

    # 5. Save the Model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    autoencoder_pipe.save_model(model_output_path)
    print(f"[INFO] Saved PyTorch model to: {model_output_path}")

    # 6. Load the Model (Sanity Check)
    print(f"\n[INFO] Reloading model from disk to verify persistence...")
    loaded_pipe = AnomalyScorePipeline(rectifier.feature_names, SERVICES)
    loaded_pipe.load_model(model_output_path)
    print(f"[INFO] Model successfully loaded!\n")

    # 7. Evaluate Anomaly Scores
    def evaluate_split(split_name, data_vectors):
        print(f"--- {split_name} Evaluation ---")
        # Aggregated anomaly scores for reporting
        scores = {s: [] for s in SERVICES}
        for x_t in data_vectors:
            signals = loaded_pipe.compute_anomaly_signals(x_t)
            for s, val in signals.items():
                scores[s].append(val)
        
        # Report Mean, Std, Min, Max
        for s in SERVICES:
            s_arr = np.array(scores[s])
            mean_val = np.mean(s_arr)
            std_val = np.std(s_arr)
            min_val = np.min(s_arr)
            max_val = np.max(s_arr)
            print(f"  {s:<20} | Mean: {mean_val:>6.3f} | Std: {std_val:>5.3f} | Min: {min_val:>6.3f} | Max: {max_val:>6.3f}")

    evaluate_split("A. Training-data Sanity Check (Expected ~ 0.0)", X_train)
    print("")
    evaluate_split("B. Held-out Healthy Validation (Expected near 0.0)", X_val)

    print("\n================================================================")
    print(" Phase 2 Autoencoder Training Completed Successfully!")
    print("================================================================")

if __name__ == "__main__":
    main()
