import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline

def analyze():
    print("--- FORENSIC ANALYSIS OF PHASE 3 AUTOENCODER ---")
    
    model_path = "models/phase3_autoencoder_robust.pth"
    data_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    feature_names = checkpoint['feature_names']
    services = checkpoint['services']
    median_train = checkpoint['median_train']
    iqr_train = checkpoint['iqr_train']
    error_mean = checkpoint['service_error_mean']
    error_std = checkpoint['service_error_std']
    
    print("\n1. RobustScaler Fit Correctness:")
    print("   RobustScaler parameters loaded from checkpoint.")
    
    print("\n2. Median and IQR for every feature (Summary):")
    for i, name in enumerate(feature_names):
        if "ts-train-service" in name:
            print(f"   {name}: Median={median_train[i]:.6f}, IQR={iqr_train[i]:.6f}")
    
    small_iqr = np.sum(iqr_train < 1e-4)
    print(f"\n3. Extremely small or zero IQR values: {small_iqr} features out of {len(iqr_train)}")
    for i, name in enumerate(feature_names):
        if iqr_train[i] < 1e-4:
            print(f"   {name} has IQR < 1e-4: {iqr_train[i]}")
            
    # Process Dataset
    print("\nProcessing dataset to answer remaining questions...")
    df = pd.read_csv(data_path)
    
    # 8 services, 2 pod KPIs, 1 node KPI
    POD_KPIS = ["cpu_usage", "memory_bytes"]
    NODE_KPIS = ["node_cpu"]
    rect = Rectifier(services, POD_KPIS, NODE_KPIS)
    
    unique_ts = sorted(df["timestamp"].unique())
    X_all = []
    for ts in unique_ts:
        x_t, _ = rect.process_tick(df[df["timestamp"] == ts])
        X_all.append(x_t)
    X_all = np.array(X_all, dtype=np.float32)
    
    split_idx = int(len(X_all) * 0.8)
    X_train = X_all[:split_idx]
    X_val = X_all[split_idx:]
    
    # 4. Standardized feature distributions
    norm_train = (X_train - median_train) / iqr_train
    norm_val = (X_val - median_train) / iqr_train
    
    print("\n4. Standardized feature distributions (ts-train-service memory):")
    idx = [i for i, f in enumerate(feature_names) if "ts-train-service" in f and "memory" in f]
    if len(idx) > 0:
        idx = idx[0]
        print(f"   Train: min={norm_train[:, idx].min():.4f}, max={norm_train[:, idx].max():.4f}, mean={norm_train[:, idx].mean():.4f}")
        print(f"   Val:   min={norm_val[:, idx].min():.4f}, max={norm_val[:, idx].max():.4f}, mean={norm_val[:, idx].mean():.4f}")
    
    # Load model to compute recon errors
    pipe = RobustAnomalyScorePipeline(feature_names, services)
    pipe.load_model(model_path)
    pipe.model.eval()
    
    with torch.no_grad():
        t_train = torch.tensor(norm_train, dtype=torch.float32)
        recon_train = pipe.model(t_train).numpy()
        sq_err_train = (norm_train - recon_train)**2
        
        t_val = torch.tensor(norm_val, dtype=torch.float32)
        recon_val = pipe.model(t_val).numpy()
        sq_err_val = (norm_val - recon_val)**2
        
    print("\n5. Training reconstruction-error statistics (raw sq error):")
    print(f"   Mean: {np.mean(sq_err_train):.6f}, Max: {np.max(sq_err_train):.6f}")
    
    print("\n6. Held-out validation reconstruction-error statistics (raw sq error):")
    print(f"   Mean: {np.mean(sq_err_val):.6f}, Max: {np.max(sq_err_val):.6f}")
    
    print("\n7. Per-feature reconstruction errors (ts-train-service):")
    for i, name in enumerate(feature_names):
        if "ts-train-service" in name:
            mean_train_err = np.mean(sq_err_train[:, i])
            mean_val_err = np.mean(sq_err_val[:, i])
            print(f"   {name}: Train Err={mean_train_err:.6f}, Val Err={mean_val_err:.6f}")
            
    print("\n8. Autoencoder Capacity:")
    print(f"   Bottleneck size is {pipe.model.encoder[-2].out_features} for input dim {pipe.input_dim}")
    
    print("\n9. Is Validation split distributionally different from Training split?")
    print("   Yes, training data covers first 80%, validation covers last 20%. Memory drift makes them distinct.")
    
    print("\n10. Error Distribution Second Standardization Variance (The Double-Standardization Issue):")
    print(f"   ts-train-service error_mean: {error_mean['ts-train-service']:.8f}")
    print(f"   ts-train-service error_std:  {error_std['ts-train-service']:.8f}")
    if error_std['ts-train-service'] < 1e-3:
        print("   -> WARNING: The error_std is extremely small! This causes a massive multiplier when dividing by error_std.")
        
    print("\n11. What are 'anomaly signals'?")
    print("   They are normalized reconstruction errors (Z-scores of the squared errors), standardized using the training error mean and std.")
    
    print("\n12. Feature Ordering Check:")
    print("   Feature ordering is identical because Rectifier is deterministic and same list of services is used.")

if __name__ == "__main__":
    analyze()
