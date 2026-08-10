import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline

def analyze_raw_mse():
    print("--- RAW RECONSTRUCTION ERROR ANALYSIS ---")
    
    model_path = "models/phase3_autoencoder_robust.pth"
    data_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    
    checkpoint = torch.load(model_path, weights_only=False)
    feature_names = checkpoint['feature_names']
    services = checkpoint['services']
    
    df = pd.read_csv(data_path)
    rect = Rectifier(services, ["cpu_usage", "memory_bytes"], ["node_cpu"])
    
    unique_ts = sorted(df["timestamp"].unique())
    X_all = []
    for ts in unique_ts:
        x_t, _ = rect.process_tick(df[df["timestamp"] == ts])
        X_all.append(x_t)
    X_all = np.array(X_all, dtype=np.float32)
    
    split_idx = int(len(X_all) * 0.8)
    X_val = X_all[split_idx:]
    
    pipe = RobustAnomalyScorePipeline(feature_names, services)
    pipe.load_model(model_path)
    pipe.model.eval()
    
    # We will compute the raw MSE exactly as the class does, but skip the Z-score
    median_train = pipe.median_train
    iqr_train = pipe.iqr_train
    
    norm_val = (X_val - median_train) / iqr_train
    
    with torch.no_grad():
        t_val = torch.tensor(norm_val, dtype=torch.float32)
        recon_val = pipe.model(t_val).numpy()
        sq_err_val = (norm_val - recon_val)**2
        
    raw_scores = {s: [] for s in services}
    
    for i in range(len(X_val)):
        for s in services:
            s_indices = [j for j, srv in enumerate(pipe.feature_service_map) if srv == s]
            raw_s_error = float(np.mean(sq_err_val[i, s_indices]))
            raw_scores[s].append(raw_s_error)
            
    print("\nPer-service mean/std/min/max of raw reconstruction error (Validation Set):")
    max_all = 0.0
    for s in services:
        arr = np.array(raw_scores[s])
        mean_v, std_v, min_v, max_v = np.mean(arr), np.std(arr), np.min(arr), np.max(arr)
        max_all = max(max_all, max_v)
        
        highlight = "  <-- ts-train-service" if s == "ts-train-service" else ""
        print(f"  {s:<20} | Mean: {mean_v:>8.3f} | Std: {std_v:>8.3f} | Min: {min_v:>8.3f} | Max: {max_v:>8.3f}{highlight}")
        
    print(f"\nAll-services maximum raw MSE: {max_all:.3f}")
    
    print("\nDDN Parameters Check:")
    print("  DDN Normal mu = 0.0")
    print("  DDN Critical mu = 5.0")
    print("\nAnalysis:")
    if max_all > 9.5: # Critical mu + 3*sigma
        print("  Clipping at 10.0 would STILL cause healthy observations to become Critical (10.0 is closest to Critical mu=5.0).")
    
if __name__ == "__main__":
    analyze_raw_mse()
