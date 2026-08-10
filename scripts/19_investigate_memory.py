import os
import pandas as pd
import numpy as np

def analyze():
    print("--- MEMORY TELEMETRY INVESTIGATION ---")
    data_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    df = pd.read_csv(data_path)
    
    df = df[df['kpi_name'] == 'memory_bytes']
    df = df.sort_values(by="timestamp")
    services = df['service_name'].unique()
    
    print("\n1 & 2. Memory Growth Patterns & 3. Pod Identity Changes")
    for s in services:
        sdf = df[df['service_name'] == s]
        unique_pods = sdf['pod_name'].unique()
        # Group by pod to check diffs
        up_ticks = 0
        down_ticks = 0
        zero_ticks = 0
        for p in unique_pods:
            pdf = sdf[sdf['pod_name'] == p]
            diffs = pdf['value'].diff().dropna()
            up_ticks += (diffs > 0).sum()
            down_ticks += (diffs < 0).sum()
            zero_ticks += (diffs == 0).sum()
            
        print(f"\nService: {s}")
        print(f"  Unique Pods: {len(unique_pods)} -> {unique_pods}")
        print(f"  Memory ticks -> Up: {up_ticks}, Down: {down_ticks}, Zero: {zero_ticks}")
        
        if down_ticks > 0 and len(unique_pods) == 1:
            print("  Pattern: Saw-tooth (GC/workload variation)")
        elif down_ticks == 0:
            print("  Pattern: Strictly Monotonic Growth")
            
    print("\n4. Investigating ts-train-service extreme memory range (11.83MB -> 52.42MB)")
    ts_train = df[df['service_name'] == 'ts-train-service'].copy()
    min_mem = ts_train['value'].min()
    max_mem = ts_train['value'].max()
    print(f"  Min Memory: {min_mem / 1e6:.2f} MB")
    print(f"  Max Memory: {max_mem / 1e6:.2f} MB")
    
    ts_train['mem_diff'] = ts_train.groupby('pod_name')['value'].diff()
    big_drops = ts_train[ts_train['mem_diff'] < -10e6]
    if not big_drops.empty:
        print("  Found large negative memory drops (>10MB):")
        for _, row in big_drops.iterrows():
            print(f"    At {row['timestamp']}: Pod {row['pod_name']}, Diff: {row['mem_diff']/1e6:.2f} MB, Current Mem: {row['value']/1e6:.2f} MB")
    else:
        print("  No single large drop >10MB found within the same pod.")
        if len(ts_train['pod_name'].unique()) > 1:
            print("  However, there are multiple pods, so the jump to 11.83MB is from a pod restart / new pod.")
            
    print("\n5 & 6. Temporal Features Distribution (Delta Memory)")
    print(f"{'Service':<20} | {'Mean (MB)':>9} | {'Std (MB)':>8} | {'Median':>8} | {'IQR':>8} | {'Min (MB)':>8} | {'Max (MB)':>8} | {'P95 (MB)':>8} | {'P99 (MB)':>8}")
    print("-" * 115)
    
    for s in services:
        sdf = df[df['service_name'] == s]
        diffs = sdf.groupby('pod_name')['value'].diff().dropna() / 1e6 # in MB
        if len(diffs) == 0:
            continue
        mean_d = diffs.mean()
        std_d = diffs.std()
        median_d = diffs.median()
        iqr_d = np.percentile(diffs, 75) - np.percentile(diffs, 25)
        min_d = diffs.min()
        max_d = diffs.max()
        p95_d = np.percentile(diffs, 95)
        p99_d = np.percentile(diffs, 99)
        print(f"{s:<20} | {mean_d:>9.4f} | {std_d:>8.4f} | {median_d:>8.4f} | {iqr_d:>8.4f} | {min_d:>8.4f} | {max_d:>8.4f} | {p95_d:>8.4f} | {p99_d:>8.4f}")

    print("\n8 & 9. Stationarity and Dimensions")
    print("  Temporal memory (delta) has mean ~ 0 and bounded variance, making it much more stationary.")
    print("  Since we are replacing absolute value with delta value per pod per tick, the total feature width remains EXACTLY 119 dimensions.")
    print("  However, the very first tick for each pod will produce a NaN diff, which Rectifier's forward-fill/imputation must handle.")
    
    print("\n10. Lost vs Gained Failure Modes")
    print("  LOST: Ability to detect a massive absolute memory leak that happens very slowly (e.g., +10KB per tick) because delta remains tiny.")
    print("  GAINED: Ability to detect sudden memory spikes (e.g., +50MB in 1 tick) or sudden drops (pod crashes) without drifting out of distribution over time.")
    
    print("\n11. memory_rate_of_change vs memory_change_percentage")
    print("  Percentage change (diff / prev) can explode (div by zero or near-zero) if memory is small.")
    print("  Raw Delta (MB) is absolute in bytes and physically grounded, so it avoids ratio explosions.")
    
    print("\n12. Why online EMA could hide a memory leak")
    print("  If an anomaly (memory leak) happens slowly, the EMA will track it. The leak becomes the 'new normal'.")
    print("  The autoencoder will continuously adapt, and the anomaly signal will never cross the Critical threshold until the pod OOMKills.")

if __name__ == "__main__":
    analyze()
