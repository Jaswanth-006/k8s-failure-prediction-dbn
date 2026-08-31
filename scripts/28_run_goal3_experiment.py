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
        pass
    return []

def collect_metrics_snapshot(phase, experiment_tick):
    records = []
    timestamp = datetime.now().isoformat()
    
    cpu_data = query_prometheus('sum(rate(container_cpu_usage_seconds_total{container!=""}[2m])) by (pod, namespace)')
    if cpu_data:
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

def run_experiment():
    G = load_discovered_service_graph()

    print("\nLoaded automatically discovered service graph:")
    print("-" * 70)
    for source, destination in G.edges():
        request_count = G[source][destination].get("request_count", 0)
        print(f"{source:25s} -> {destination:25s} (requests={request_count:.0f})")

    rect = Rectifier(SERVICES, ["cpu_usage"], ["node_cpu"])
    pipe = RobustAnomalyScorePipeline(rect.feature_names, SERVICES)
    pipe.load_model("models/phase3_autoencoder_cpu_only.pth")

    ddn = DynamicDecisionNetworkPhase3(G, num_particles=1000)
    controller = KubernetesActionController(shadow_mode=True)
    
    results = []
    
    # Check if Prometheus is up. If not, fallback to playback mode using Goal 1 data.
    use_playback = False
    try:
        res = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
        if res.status_code != 200:
            use_playback = True
    except:
        use_playback = True

    if use_playback:
        playback_file = "data/experiments/goal1_clean_experiment_telemetry.csv"
        print(f"\n[!] Prometheus unreachable. Entering OFFLINE PLAYBACK MODE using: {playback_file}")
        if not os.path.exists(playback_file):
            print("Playback file not found. Cannot proceed.")
            sys.exit(1)
        playback_df = pd.read_csv(playback_file)
        max_ticks = playback_df["experiment_tick"].max()
    else:
        print("\n[+] Prometheus is reachable. Running LIVE experiment.")
        # Ensure clean state
        subprocess.run("kubectl exec deployment/ts-train-service -- pkill -f yes", shell=True)

    def process_tick_data(df_tick, phase, tick_num):
        tick_timestamp = df_tick["timestamp"].iloc[0]
        
        x_t, _ = rect.process_tick(df_tick)
        anomaly_signals = pipe.compute_anomaly_signals(x_t)
        ddn_output = ddn.step(anomaly_signals, node_pressure_flag=False)
        
        root_cause = ddn_output["root_cause"]
        controller.reconcile_tick(ddn_output, node_pressure_flag=False)
        
        pers = controller.decision_policy.root_cause_persistence.get(root_cause, 0) if root_cause != "None" else 0
        
        # Print Goal 3 Diagnostics
        print(f"\n[{phase}] Tick {tick_num} Diagnostics:")
        print("Service health (Critical or highly anomalous):")
        for s in SERVICES:
            post = ddn_output["posteriors"][s]
            anomaly = anomaly_signals.get(s, 0.0)
            if anomaly > 1.5 or post["Critical"] > 0.1 or post["Degrading"] > 0.1:
                print(f"  {s}: anomaly={anomaly:.2f} P(Normal)={post['Normal']:.2f} P(Degrading)={post['Degrading']:.2f} P(Critical)={post['Critical']:.2f}")

        causal_data = ddn_output.get("causal_data")
        if causal_data:
            print("\nCausal analysis:")
            for item in causal_data["ranked"]:
                s = item[0]
                score_data = item[1]
                if score_data["classification"] != "NORMAL":
                    print(f"  {s}:")
                    print(f"    classification={score_data['classification']}")
                    print(f"    intrinsic_score={score_data['intrinsic_evidence']:.2f}")
                    print(f"    upstream_causal_score={score_data['upstream_causal_evidence']:.2f}")
                    print(f"    victim_score={score_data['victim_evidence']:.2f}")

            print("\nRoot cause ranking:")
            for idx, item in enumerate(causal_data["ranked"][:3]):
                print(f"  {idx+1}. {item[0]} (score={item[1]['root_cause_score']:.2f})")
            
            print(f"\nFinal RCA Selection: {root_cause} (Persistence: {pers})")
            print("-" * 50)
        
        best_action = "Do_Nothing"
        service_eu = ddn_output["expected_utilities"].get(root_cause, {})
        if service_eu:
            best_action = max(service_eu, key=service_eu.get)
        
        results.append({
            "timestamp": tick_timestamp,
            "phase": phase,
            "tick": tick_num,
            "ts_train_signal": anomaly_signals.get("ts-train-service", 0.0),
            "ts_route_signal": anomaly_signals.get("ts-route-service", 0.0),
            "root_cause": root_cause,
            "persistence": pers,
            "meu_action": best_action
        })

    if use_playback:
        for tick_num in range(1, max_ticks + 1):
            df_tick = playback_df[playback_df["experiment_tick"] == tick_num]
            if df_tick.empty:
                continue
            phase = df_tick["phase"].iloc[0]
            process_tick_data(df_tick, phase, tick_num)
    else:
        # Live Run
        def run_ticks(phase, num_ticks):
            for i in range(num_ticks):
                tick_num = len(results) + 1
                df_tick = collect_metrics_snapshot(phase, tick_num)
                process_tick_data(df_tick, phase, tick_num)
                time.sleep(5)
                
        print("\n--- STAGE 1: PRE-FAULT ---")
        run_ticks("PRE-FAULT", 10)
        
        print("\n--- INJECTING FAULT (Manual 2-worker stress) ---")
        subprocess.run("kubectl exec deployment/ts-train-service -- sh -c 'yes > /dev/null & yes > /dev/null &'", shell=True)
        
        print("\n--- STAGE 2: FAULT ONSET & SUSTAINED ---")
        run_ticks("EXPERIMENT", 20)
        
        print("\n--- CLEANING UP FAULT ---")
        subprocess.run("kubectl exec deployment/ts-train-service -- pkill -f yes", shell=True)
        
        print("\n--- STAGE 3: RECOVERY ---")
        run_ticks("RECOVERY", 10)
    
    print("\n=========================================================================")
    print(" EXPERIMENT REPORT (Goal 3)")
    print("=========================================================================")
    
    res_df = pd.DataFrame(results)
    os.makedirs("data/experiments", exist_ok=True)
    output_path = "data/experiments/goal3_causal_experiment_results.csv"
    res_df.to_csv(output_path, index=False)

    print(f"\nSaved experiment results to: {output_path}")
    
    # Analyze if Goal 3 worked
    # ts-train-service should be ROOT_CAUSE during experiment
    # ts-route-service might become PROPAGATED_VICTIM
    print("\nGoal 3 Verification:")
    exp_data = res_df[res_df["phase"] == "EXPERIMENT"]
    root_causes = exp_data["root_cause"].value_counts()
    print("Root Cause counts during fault injection phase:")
    print(root_causes)
    if "ts-train-service" in root_causes and root_causes["ts-train-service"] > 0:
        print("-> SUCCESS: ts-train-service was successfully identified as the root cause despite potential downstream victims.")
    else:
        print("-> FAILED: ts-train-service was not identified as the root cause.")

if __name__ == "__main__":
    run_experiment()
