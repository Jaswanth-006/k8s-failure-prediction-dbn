import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline

def run_analysis():
    print("--- PHASE 3 CALIBRATION METHOD FEASIBILITY ANALYSIS ---")
    
    model_path = "models/phase3_autoencoder_robust.pth"
    data_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    
    checkpoint = torch.load(model_path, weights_only=False)
    feature_names = checkpoint['feature_names']
    services = checkpoint['services']
    median_train = checkpoint['median_train']
    iqr_train = checkpoint['iqr_train']
    
    df = pd.read_csv(data_path)
    rect = Rectifier(services, ["cpu_usage", "memory_bytes"], ["node_cpu"])
    
    unique_ts = sorted(df["timestamp"].unique())
    X_all = []
    for ts in unique_ts:
        x_t, _ = rect.process_tick(df[df["timestamp"] == ts])
        X_all.append(x_t)
    X_all = np.array(X_all, dtype=np.float32)
    
    # 70 / 15 / 15 Split
    total_ticks = len(X_all)
    idx_70 = int(total_ticks * 0.70)
    idx_85 = int(total_ticks * 0.85)
    
    # We only need Calibration (15%) and Validation (15%)
    X_calib = X_all[idx_70:idx_85]
    X_val = X_all[idx_85:]
    
    pipe = RobustAnomalyScorePipeline(feature_names, services)
    pipe.load_model(model_path)
    pipe.model.eval()
    
    def get_raw_mse(X_data):
        norm_data = (X_data - median_train) / iqr_train
        with torch.no_grad():
            t_data = torch.tensor(norm_data, dtype=torch.float32)
            recon = pipe.model(t_data).numpy()
            sq_err = (norm_data - recon)**2
            
        raw_scores = {s: [] for s in services}
        for i in range(len(X_data)):
            for s in services:
                s_indices = [j for j, srv in enumerate(pipe.feature_service_map) if srv == s]
                raw_s_error = float(np.mean(sq_err[i, s_indices]))
                raw_scores[s].append(raw_s_error)
        return raw_scores

    calib_raw = get_raw_mse(X_calib)
    val_raw = get_raw_mse(X_val)
    
    # Fit Calibration Parameters on Calibration Set (Method A, B, C)
    calib_params = {}
    for s in services:
        arr = np.array(calib_raw[s])
        calib_params[s] = {
            'mean': np.mean(arr),
            'std': np.std(arr) if np.std(arr) > 1e-6 else 1.0,
            'median': np.median(arr),
            'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
            'p99': np.percentile(arr, 99)
        }
        if calib_params[s]['iqr'] < 1e-6:
            calib_params[s]['iqr'] = 1.0
            
    # Apply and Evaluate on Validation Set
    results_A = {s: [] for s in services}
    results_B = {s: [] for s in services}
    results_C = {s: [] for s in services}
    
    for s in services:
        arr_val = np.array(val_raw[s])
        
        # Method A: Z-score
        z_a = (arr_val - calib_params[s]['mean']) / calib_params[s]['std']
        results_A[s] = np.clip(z_a, 0.0, None)  # DDN signals are typically >= 0
        
        # Method B: Robust Z-score
        z_b = (arr_val - calib_params[s]['median']) / (calib_params[s]['iqr'] / 1.349)
        results_B[s] = np.clip(z_b, 0.0, None)
        
        # Method C: Percentile (Scale 99th percentile to 2.5 (Degrading threshold))
        # This aligns the "worst" healthy data roughly with the start of degradation
        scale_c = 2.5 / (calib_params[s]['p99'] if calib_params[s]['p99'] > 1e-6 else 1.0)
        z_c = arr_val * scale_c
        results_C[s] = np.clip(z_c, 0.0, None)

    methods = [("A (Z-score)", results_A), ("B (Robust Z-score)", results_B), ("C (P99 scaled to 2.5)", results_C)]
    
    for method_name, results in methods:
        print(f"\n=======================================================")
        print(f" METHOD: {method_name}")
        print(f"=======================================================")
        
        exceed_5 = 0
        exceed_10 = 0
        total_obs = len(X_val) * len(services)
        
        for s in services:
            arr = results[s]
            mean_v, std_v = np.mean(arr), np.std(arr)
            min_v, max_v = np.min(arr), np.max(arr)
            p95, p99 = np.percentile(arr, 95), np.percentile(arr, 99)
            
            exceed_5 += np.sum(arr > 5.0)
            exceed_10 += np.sum(arr > 10.0)
            
            print(f" {s:<20} | Mean:{mean_v:>6.2f} | Std:{std_v:>6.2f} | Min:{min_v:>6.2f} | Max:{max_v:>7.2f} | P95:{p95:>6.2f} | P99:{p99:>6.2f}")
            
        print(f"\n Threshold Exceedance (Total Observations = {total_obs}):")
        print(f"   > 5.0 (Critical DDN Threshold): {exceed_5} / {total_obs}")
        print(f"   > 10.0 (Hard Clip):             {exceed_10} / {total_obs}")
        if exceed_5 > total_obs * 0.05:
            print("   -> CONCLUSION: Too many False Positives. Unsafe for DDN.")
        elif exceed_5 > 0:
            print("   -> CONCLUSION: Some False Positives. Marginally unsafe.")
        else:
            print("   -> CONCLUSION: Zero False Positives on healthy data. Compatible with DDN.")

if __name__ == "__main__":
    run_analysis()
