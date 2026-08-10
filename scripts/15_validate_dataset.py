import pandas as pd
import numpy as np

def validate_dataset():
    path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    df = pd.read_csv(path)
    
    unique_ts = df["timestamp"].nunique()
    print(f"1. Unique collector timestamps: {unique_ts}")
    
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df["value"]).sum()
    print(f"2. NaN values: {nan_count}, Inf values: {inf_count}")
    
    services = df["service_name"].unique()
    expected_services = [
        "ts-ui-dashboard", "ts-user-service", "ts-train-service", 
        "ts-route-service", "ts-order-service", "ts-payment-service", 
        "ts-inventory-service", "ts-station-service"
    ]
    missing_services = [s for s in expected_services if s not in services]
    print(f"3. All 8 expected services represented: {len(missing_services) == 0} (Missing: {missing_services})")
    
    # Check timestamps
    ts_list = sorted(df["timestamp"].unique())
    start = ts_list[0]
    end = ts_list[-1]
    print(f"4. Timestamp span:")
    print(f"   Start: {start}")
    print(f"   End:   {end}")
    
    # 5. We will verify 119 dimensions by instantiating Rectifier
    from src.rectifier import Rectifier
    rect = Rectifier(expected_services, ["cpu_usage", "memory_bytes"], ["node_cpu"])
    print(f"5. Rectifier feature vector width: {rect.input_width} dimensions")

if __name__ == "__main__":
    validate_dataset()
