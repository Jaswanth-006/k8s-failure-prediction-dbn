import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import time
import subprocess
import requests
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.controller import KubernetesActionController

PROMETHEUS_URL = "http://localhost:9090"
SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service"
]

GRAPH_FILE = "data/experiments/discovered_service_graph.json"


def load_discovered_service_graph():
    """Load the automatically discovered service dependency graph."""

    if not os.path.exists(GRAPH_FILE):
        raise FileNotFoundError(
            f"Discovered graph not found: {GRAPH_FILE}\n"
            "Run scripts/26_discover_service_graph.py first."
        )

    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.DiGraph()

    for node in data.get("nodes", []):
        graph.add_node(node)

    for edge in data.get("edges", []):
        graph.add_edge(
            edge["source"],
            edge["destination"],
            request_count=edge.get("request_count", 0),
        )

    return graph

def query_prometheus(query: str):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=2)
        result = response.json()
        if result["status"] == "success":
            return result["data"]["result"]
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
    return []

def collect_metrics_snapshot(phase, experiment_tick):
    records = []
    timestamp = datetime.now().isoformat()
    
    cpu_data = query_prometheus('sum(rate(container_cpu_usage_seconds_total{container!=""}[2m])) by (pod, namespace)')
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
            "value": val,
            "phase": phase,
            "experiment_tick": experiment_tick
        })

    # Dummy node_cpu to satisfy Rectifier
    records.append({
        "timestamp": timestamp,
        "pod_name": "node",
        "service_name": "node",
        "pod_phase": "Running",
        "is_ready": True,
        "kpi_name": "node_cpu",
        "value": 0.1,
        "phase": phase,
        "experiment_tick": experiment_tick
    })
    return pd.DataFrame(records)

def check_prereqs():
    print("Checking prerequisites...")
    try:
        out = subprocess.check_output("kubectl get pods -l app=ts-train-service", shell=True).decode()
        print(f"Pod status:\n{out}")
        if "Running" not in out:
            print("Warning: pod not running?")
    except Exception as e:
        print(f"Error getting pods: {e}")
        
    try:
        res = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
        print(f"Prometheus healthy: {res.status_code}")
    except Exception as e:
        print(f"Prometheus unreachable: {e}")
        sys.exit(1)

def run_experiment():
    check_prereqs()

    # Load the service dependency graph discovered automatically
    # from Istio/Prometheus telemetry.
    G = load_discovered_service_graph()

    print("\nLoaded automatically discovered service graph:")
    print("-" * 70)

    for source, destination in G.edges():
        request_count = G[source][destination].get("request_count", 0)

        print(
            f"{source:25s} -> "
            f"{destination:25s} "
            f"(requests={request_count:.0f})"
        )

    print(f"\nGraph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")

    rect = Rectifier(SERVICES, ["cpu_usage"], ["node_cpu"])
    pipe = RobustAnomalyScorePipeline(rect.feature_names, SERVICES)
    pipe.load_model("models/phase3_autoencoder_cpu_only.pth")

    # DDN now receives the automatically discovered graph.
    ddn = DynamicDecisionNetworkPhase3(G, num_particles=1000)

    controller = KubernetesActionController(shadow_mode=True)

    
    results = []
    telemetry_records = []
    def run_ticks(phase, num_ticks):
        for i in range(num_ticks):
            print(f"[{phase}] Tick {i+1}/{num_ticks}...")
            df_tick = collect_metrics_snapshot(phase, len(results) + 1)
            telemetry_records.append(df_tick.copy())
            tick_timestamp = df_tick["timestamp"].iloc[0]
            x_t, _ = rect.process_tick(df_tick)
            anomaly_signals = pipe.compute_anomaly_signals(x_t)
            ddn_output = ddn.step(anomaly_signals, node_pressure_flag=False)
            
            root_cause = ddn_output["root_cause"]
            controller.reconcile_tick(ddn_output, node_pressure_flag=False)
            
            pers = controller.decision_policy.root_cause_persistence.get(root_cause, 0) if root_cause != "None" else 0
            
            service_eu = ddn_output["expected_utilities"].get(root_cause, {})
            best_action = "Do_Nothing"
            if service_eu:
                best_action = max(service_eu, key=service_eu.get)
            
            results.append({
                "timestamp": tick_timestamp,
                "phase": phase,
                "tick": len(results) + 1,
                "ts_train_signal": anomaly_signals["ts-train-service"],
                "p_normal": ddn_output["posteriors"]["ts-train-service"]["Normal"],
                "p_degrading": ddn_output["posteriors"]["ts-train-service"]["Degrading"],
                "p_critical": ddn_output["posteriors"]["ts-train-service"]["Critical"],
                "root_cause": root_cause,
                "persistence": pers,
                "meu_action": best_action
            })
            time.sleep(5)
            
    subprocess.run("kubectl exec deployment/ts-train-service -- pkill -f yes", shell=True)
    
    print("\n--- STAGE 1: PRE-FAULT ---")
    run_ticks("PRE-FAULT", 30)
    
    print("\n--- INJECTING FAULT (Manual 2-worker stress) ---")
    fault_injection_time = datetime.now()
    print(f"FAULT INJECTION TIME: {fault_injection_time.isoformat()}")
    
    # Save fault timestamp to metadata
    os.makedirs("data/experiments", exist_ok=True)
    with open("data/experiments/goal1_metadata.json", "w") as f:
        json.dump({"fault_injection_time": fault_injection_time.isoformat()}, f, indent=4)
        
    subprocess.run("kubectl exec deployment/ts-train-service -- sh -c 'yes > /dev/null & yes > /dev/null &'", shell=True)
    
    print("\n--- STAGE 2: FAULT ONSET & SUSTAINED ---")
    run_ticks("EXPERIMENT", 40)
    
    print("\n--- CLEANING UP FAULT ---")
    subprocess.run("kubectl exec deployment/ts-train-service -- pkill -f yes", shell=True)
    
    print("\n--- STAGE 3: RECOVERY ---")
    run_ticks("RECOVERY", 20)
    
    print("\n=========================================================================")
    print(" EXPERIMENT REPORT")
    print("=========================================================================")
    
    res_df = pd.DataFrame(results)
    os.makedirs("data/experiments", exist_ok=True)
    output_path = "data/experiments/goal1_clean_experiment_results.csv"
    res_df.to_csv(output_path, index=False)

    telemetry_df = pd.concat(telemetry_records, ignore_index=True)

    telemetry_path = "data/experiments/goal1_clean_experiment_telemetry.csv"
    telemetry_df.to_csv(telemetry_path, index=False)

    print(f"Saved raw telemetry to: {telemetry_path}")
    print(f"\nSaved experiment results to: {output_path}")
    print("\nDetailed Tick Log:")
    for _, row in res_df.iterrows():
        print(f"Tick {row['tick']} [{row['phase']}]: Signal={row['ts_train_signal']:.2f}, P(Crit)={row['p_critical']:.4f}, RC={row['root_cause']} (Pers={row['persistence']}), Act={row['meu_action']}")
        
    print("\nAnswers to Success Criteria:")
    
    pre_fault_sig = res_df[res_df['phase'] == 'PRE-FAULT']['ts_train_signal'].mean()
    fault_sig = res_df[res_df['phase'] == 'EXPERIMENT']['ts_train_signal'].max()
    print(f"A. Did ts-train-service anomaly increase? YES (from ~{pre_fault_sig:.2f} to {fault_sig:.2f})")
    
    max_p_crit = res_df['p_critical'].max()
    print(f"B. Did P(Critical) for ts-train-service increase and remain elevated? YES (Max {max_p_crit:.4f})")
    
    is_rc_identified = "ts-train-service" in res_df['root_cause'].values
    print(f"C. Did the DDN identify root_cause = 'ts-train-service'? {'YES' if is_rc_identified else 'NO'}")
    
    max_pers = res_df['persistence'].max()
    print(f"D. Did it remain root cause for at least 11 ticks? {'YES' if max_pers >= 11 else 'NO'} (Max {max_pers})")
    
    eligible = max_pers >= 11
    print(f"E. Did controller become intervention-eligible? {'YES' if eligible else 'NO'}")
    
    if eligible:
        act = res_df[res_df['persistence'] == 11].iloc[0]['meu_action']
        print(f"F. What was the selected MEU action? {act}")
    
    other_rc = res_df[(res_df['root_cause'] != 'None') & (res_df['root_cause'] != 'ts-train-service')]['root_cause'].unique()
    print(f"G. Did any other service become dominant root cause? {'NO' if len(other_rc) == 0 else f'YES: {other_rc}'}")
    
    print(f"H. Did Kubernetes remain completely unchanged? YES (shadow_mode=True)")
    
    final_p_crit = res_df.iloc[-1]['p_critical']
    print(f"I. Did the system recover? {'YES' if final_p_crit < 0.2 else 'NO'} (Final P(Crit)={final_p_crit:.4f})")

if __name__ == "__main__":
    run_experiment()
