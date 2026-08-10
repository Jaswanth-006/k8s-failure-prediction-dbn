"""
Prometheus Metrics Ingestion & Telemetry Collector Script (Phase 3)
Queries Prometheus every 60 seconds and logs raw pod/node KPIs into CSV dataset format.
Target duration: 60 minutes (3600 ticks at 1-second intervals, representing 60 minutes of scrape points).
Saves progressively to prevent data loss if connection drops.
Automatically reconnects the Prometheus port-forward if it drops.
"""

import time
import pandas as pd
import requests
from datetime import datetime
import os
import subprocess
import threading

PROMETHEUS_URL = "http://localhost:9090"
PORT_FORWARD_CMD = ["kubectl", "port-forward", "-n", "monitoring", "svc/prometheus-server", "9090:80"]

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

pf_process = None

def start_port_forward():
    global pf_process
    print(f"[{datetime.now().isoformat()}] Starting Prometheus port-forward...")
    if pf_process is not None:
        try:
            pf_process.terminate()
        except:
            pass
    # Use subprocess.Popen to run in background
    pf_process = subprocess.Popen(
        PORT_FORWARD_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3) # Give it time to bind

def check_and_reconnect_prometheus():
    try:
        # Check if Prometheus is reachable
        requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=3)
    except requests.RequestException:
        print(f"[{datetime.now().isoformat()}] Connection to Prometheus lost. Reconnecting...")
        start_port_forward()

def query_prometheus(query: str):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
        result = response.json()
        if result.get("status") == "success":
            return result["data"]["result"]
    except Exception as e:
        pass 
    return None

def collect_metrics_snapshot():
    records = []
    timestamp = datetime.now().isoformat()
    
    cpu_data = query_prometheus('sum(rate(container_cpu_usage_seconds_total{container!=""}[5m])) by (pod, namespace)')
    if cpu_data is None:
        return None
        
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

    mem_data = query_prometheus('sum(container_memory_working_set_bytes{container!=""}) by (pod, namespace)')
    if mem_data is None:
        return None
        
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
    print("Starting Prometheus Telemetry Collector for Phase 3 (1-second tick loop for 60 minutes)...")
    out_path = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Check existing data to resume
    existing_ticks = 0
    if os.path.exists(out_path):
        try:
            df_existing = pd.read_csv(out_path)
            existing_ticks = df_existing['timestamp'].nunique()
            print(f"Resuming collection. Found {existing_ticks} existing timestamps in dataset.")
        except Exception as e:
            print("Could not read existing dataset, starting fresh.")
            pd.DataFrame(columns=["timestamp", "pod_name", "service_name", "pod_phase", "is_ready", "kpi_name", "value"]).to_csv(out_path, index=False)
    else:
        pd.DataFrame(columns=["timestamp", "pod_name", "service_name", "pod_phase", "is_ready", "kpi_name", "value"]).to_csv(out_path, index=False)
    
    total_target = 3600
    if existing_ticks >= total_target:
        print("Target already reached.")
        exit(0)
        
    start_port_forward()
    
    successful_ticks = existing_ticks
    
    try:
        while successful_ticks < total_target:
            df_tick = collect_metrics_snapshot()
            if df_tick is not None and not df_tick.empty:
                df_tick.to_csv(out_path, mode='a', header=False, index=False)
                successful_ticks += 1
                if successful_ticks % 60 == 0:
                    print(f"\n--- [Tick {successful_ticks}/{total_target}] Collected Metrics ({(successful_ticks//60)} minutes total) ---")
            else:
                # Failure detected
                print(f"[{datetime.now().isoformat()}] Failed to collect metrics. Checking connection...")
                check_and_reconnect_prometheus()
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nCollection stopped manually.")
    finally:
        if pf_process is not None:
            pf_process.terminate()
            
    print(f"Collection complete. Dataset has {successful_ticks} timestamps.")
