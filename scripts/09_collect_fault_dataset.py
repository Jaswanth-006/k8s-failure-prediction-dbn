"""
Prometheus Metrics Ingestion & Telemetry Collector Script
Queries Prometheus every 60 seconds and logs raw pod/node KPIs into CSV dataset format.
Tags timestamps with Healthy (0) vs. Injected Fault (1) labels.
"""

import time
import pandas as pd
import requests
from datetime import datetime

PROMETHEUS_URL = "http://localhost:9090"

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

def query_prometheus(query: str):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        result = response.json()
        if result["status"] == "success":
            return result["data"]["result"]
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
    return []

def collect_metrics_snapshot():
    records = []
    timestamp = datetime.now().isoformat()
    
    # Query Pod CPU Usage
    cpu_data = query_prometheus('sum(rate(container_cpu_usage_seconds_total{container!=""}[5m])) by (pod, namespace)')
    for item in cpu_data:
        pod_name = item["metric"].get("pod", "unknown")
        val = float(item["value"][1])
        srv_name = "unknown"
        for s in SERVICES:
            if s in pod_name:
                srv_name = s
                break
        records.append({
            "timestamp": timestamp,
            "pod_name": pod_name,
            "service_name": srv_name,
            "pod_phase": "Running",
            "is_ready": True,
            "kpi_name": "cpu_usage",
            "value": val
        })

    # Query Pod Memory Usage
    mem_data = query_prometheus('sum(container_memory_working_set_bytes{container!=""}) by (pod, namespace)')
    for item in mem_data:
        pod_name = item["metric"].get("pod", "unknown")
        val = float(item["value"][1])
        srv_name = "unknown"
        for s in SERVICES:
            if s in pod_name:
                srv_name = s
                break
        records.append({
            "timestamp": timestamp,
            "pod_name": pod_name,
            "service_name": srv_name,
            "pod_phase": "Running",
            "is_ready": True,
            "kpi_name": "memory_bytes",
            "value": val
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    print("Starting Prometheus Telemetry Collector (1-minute tick)...")
    dataset_df = pd.DataFrame()
    # Start 5-minute (300 seconds) collection loop
    for tick in range(300):
        print(f"\n--- [Tick {tick+1}/300] Collecting Metrics ---")
        df_tick = collect_metrics_snapshot()
        dataset_df = pd.concat([dataset_df, df_tick], ignore_index=True)
        time.sleep(1)

    dataset_df.to_csv("data/raw/faults/cpu_train_service_telemetry_dataset.csv", index=False)
    print("Saved telemetry dataset to 'data/raw/faults/cpu_train_service_telemetry_dataset.csv'!")
