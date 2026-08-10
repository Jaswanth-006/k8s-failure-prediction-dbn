import pandas as pd
import numpy as np

def generate_report():
    dataset_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    print("================================================================")
    print(" PHASE 3 HEALTHY BASELINE COLLECTION REPORT")
    print("================================================================\n")
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
        
    timestamps = sorted(df['timestamp'].unique())
    num_timestamps = len(timestamps)
    start_ts = timestamps[0] if num_timestamps > 0 else "N/A"
    end_ts = timestamps[-1] if num_timestamps > 0 else "N/A"
    row_count = len(df)
    
    # Calculate missing timestamps (expecting 1 second intervals roughly)
    # Since network latency can vary, we just check if we have roughly 3600 timestamps
    missing = max(0, 3600 - num_timestamps)
    
    # Check for NaN/Inf
    nan_count = df['value'].isna().sum()
    inf_count = np.isinf(df['value']).sum()
    
    print(f"Total Timestamps Collected: {num_timestamps} (Target: 3600)")
    print(f"Start Timestamp: {start_ts}")
    print(f"End Timestamp:   {end_ts}")
    print(f"Total Rows:      {row_count}")
    print(f"Missing Ticks:   ~{missing}")
    print(f"NaN Values:      {nan_count}")
    print(f"Inf Values:      {inf_count}")
    
    print("\n--- Per-Service Statistics ---\n")
    services = df['service_name'].unique()
    kpis = df['kpi_name'].unique()
    
    for srv in sorted(services):
        if srv == 'unknown': continue
        print(f"Service: {srv}")
        srv_df = df[df['service_name'] == srv]
        for kpi in kpis:
            kpi_df = srv_df[srv_df['kpi_name'] == kpi]
            if len(kpi_df) > 0:
                vals = kpi_df['value'].values
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                min_v = np.min(vals)
                max_v = np.max(vals)
                
                if 'memory' in kpi:
                    print(f"  {kpi:<12} | Mean: {mean_v/1e6:>8.2f} MB | Std: {std_v/1e6:>8.2f} MB | Min: {min_v/1e6:>8.2f} MB | Max: {max_v/1e6:>8.2f} MB")
                else:
                    print(f"  {kpi:<12} | Mean: {mean_v:>8.4f} | Std: {std_v:>8.4f} | Min: {min_v:>8.4f} | Max: {max_v:>8.4f}")
        print("")
        
if __name__ == "__main__":
    generate_report()
