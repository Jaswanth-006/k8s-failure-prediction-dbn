"""
PREFACE-DBN Goal 4: Multi-Signal Telemetry & Evidence-Rich RCA Experiment
===========================================================================
scripts/29_run_goal4_experiment.py

Modes
-----
MODE 1 (offline playback):
  Used when Prometheus is not reachable.
  Simulates a realistic fault scenario using synthetic but deterministic data
  derived from the Goal 1 telemetry dataset structure. Goal 2 discovered graph
  is loaded from disk.

MODE 2 (live Prometheus):
  Connects to http://localhost:9090.
  Discovers available metrics, collects service + edge telemetry, runs full
  Goal 3 + Goal 4 enriched RCA pipeline, prints structured snapshot output.

Output format
-------------
  SERVICE TELEMETRY table
  EDGE TELEMETRY table
  CAUSAL RCA table
  Final root cause
  Results saved to data/experiments/goal4_experiment_results.csv

The fault experiment demonstrates why multiple signals improve discrimination:
  HEALTHY: low CPU, normal memory, normal requests, low error rate
  FAULT:   CPU spikes on ts-train-service, error rate rises
           ts-route-service degrades but is classified PROPAGATED_VICTIM
  RECOVERY: metrics return toward normal, RCA returns to None/NORMAL
"""

import os
import sys
import json
import time
import math
import random
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.controller import KubernetesActionController
from src.causal_rca import DirectionalCausalAnalyzer
from src.telemetry_schema import ServiceTelemetry, EdgeTelemetry, TelemetrySnapshot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMETHEUS_URL = "http://localhost:9090"
GRAPH_FILE     = "data/experiments/discovered_service_graph.json"
OUTPUT_FILE    = "data/experiments/goal4_experiment_results.csv"
MODEL_PATH     = "models/phase3_autoencoder_cpu_only.pth"
PLAYBACK_FILE  = "data/experiments/goal1_clean_experiment_telemetry.csv"

SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service"
]

COL_WIDTH_SVC  = 28
COL_WIDTH_EDGE = 40


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------
def load_discovered_graph() -> nx.DiGraph:
    if not os.path.exists(GRAPH_FILE):
        raise FileNotFoundError(
            f"Discovered graph not found: {GRAPH_FILE}\n"
            "Run scripts/26_discover_service_graph.py first."
        )
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data.get("nodes", []):
        G.add_node(node)
    for edge in data.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["destination"],
            request_count=edge.get("request_count", 0.0),
        )
    return G


# ---------------------------------------------------------------------------
# Prometheus helpers (live mode)
# ---------------------------------------------------------------------------
def is_prometheus_available() -> bool:
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def prom_query(query: str) -> list:
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        data = r.json()
        if data.get("status") == "success":
            return data["data"]["result"]
    except Exception:
        pass
    return []


def prom_metric_exists(metric_name: str) -> bool:
    results = prom_query(f"count({metric_name})")
    return len(results) > 0


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _match_service(pod_name: str) -> str:
    for s in SERVICES:
        if s in pod_name:
            return s
    return "unknown"


# ---------------------------------------------------------------------------
# Live telemetry collection
# ---------------------------------------------------------------------------
def collect_live_snapshot(G: nx.DiGraph) -> TelemetrySnapshot:
    """Query Prometheus for all Goal 4 signals and build a TelemetrySnapshot."""
    from src.telemetry_collector import PrometheusTelemetryCollector
    collector = PrometheusTelemetryCollector(
        prometheus_url=PROMETHEUS_URL,
        services=SERVICES,
        service_graph=G,
    )
    snap = collector.collect()
    if snap is None:
        snap = TelemetrySnapshot()
    return snap


# ---------------------------------------------------------------------------
# Live CPU metrics for Rectifier (existing pipeline)
# ---------------------------------------------------------------------------
def collect_cpu_df(phase: str, tick_num: int) -> pd.DataFrame:
    records = []
    timestamp = datetime.now().isoformat()

    cpu_data = prom_query(
        'sum(rate(container_cpu_usage_seconds_total{container!="",container!="POD"}[2m])) by (pod)'
    )
    for item in cpu_data:
        pod  = item["metric"].get("pod", "unknown")
        val  = _safe_float(item["value"][1])
        svc  = _match_service(pod)
        records.append({
            "timestamp": timestamp, "pod_name": pod, "service_name": svc,
            "pod_phase": "Running", "is_ready": True,
            "kpi_name": "cpu_usage", "value": val,
            "phase": phase, "experiment_tick": tick_num,
        })

    # Dummy node_cpu required by Rectifier
    records.append({
        "timestamp": timestamp, "pod_name": "node", "service_name": "node",
        "pod_phase": "Running", "is_ready": True,
        "kpi_name": "node_cpu", "value": 0.1,
        "phase": phase, "experiment_tick": tick_num,
    })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Synthetic offline telemetry (MODE 1)
# ---------------------------------------------------------------------------
def build_synthetic_snapshot(phase: str, G: nx.DiGraph) -> TelemetrySnapshot:
    """
    Build a deterministic synthetic TelemetrySnapshot based on the experiment phase.

    HEALTHY  : all services operating normally.
    FAULT    : ts-train-service has high CPU, high error rate.
               ts-route-service shows elevated error rate (downstream propagation).
    RECOVERY : all metrics trending back toward normal.
    """
    snap = TelemetrySnapshot()

    for svc in SERVICES:
        if phase == "HEALTHY":
            cpu   = 0.05 + random.uniform(0, 0.03)
            mem   = 50_000_000 + random.randint(0, 10_000_000)
            rrate = 2.0 + random.uniform(-0.5, 0.5)
            erate = 0.0
        elif phase == "FAULT":
            if svc == "ts-train-service":
                cpu   = 0.80 + random.uniform(0, 0.15)
                mem   = 400_000_000 + random.randint(0, 100_000_000)
                rrate = 50.0 + random.uniform(-5, 5)
                erate = 0.45 + random.uniform(0, 0.15)
            elif svc == "ts-route-service":
                cpu   = 0.15 + random.uniform(0, 0.05)
                mem   = 80_000_000 + random.randint(0, 20_000_000)
                rrate = 30.0 + random.uniform(-5, 5)
                erate = 0.20 + random.uniform(0, 0.10)
            else:
                cpu   = 0.05 + random.uniform(0, 0.02)
                mem   = 50_000_000 + random.randint(0, 5_000_000)
                rrate = 2.0 + random.uniform(-0.5, 0.5)
                erate = 0.0
        else:  # RECOVERY
            if svc == "ts-train-service":
                cpu   = 0.20 + random.uniform(0, 0.10)
                mem   = 150_000_000 + random.randint(0, 50_000_000)
                rrate = 10.0 + random.uniform(-2, 2)
                erate = 0.05 + random.uniform(0, 0.05)
            elif svc == "ts-route-service":
                cpu   = 0.08 + random.uniform(0, 0.03)
                mem   = 60_000_000 + random.randint(0, 10_000_000)
                rrate = 5.0 + random.uniform(-1, 1)
                erate = 0.02 + random.uniform(0, 0.02)
            else:
                cpu   = 0.05 + random.uniform(0, 0.02)
                mem   = 50_000_000 + random.randint(0, 5_000_000)
                rrate = 2.0 + random.uniform(-0.3, 0.3)
                erate = 0.0

        snap.services[svc] = ServiceTelemetry(
            service=svc, cpu_rate=cpu, memory_bytes=mem,
            request_rate=rrate, error_rate=erate, available=True,
        )

    for src, dst, data_attr in G.edges(data=True):
        req_count = data_attr.get("request_count", 40.0)
        if phase == "FAULT" and src == "ts-train-service" and dst == "ts-route-service":
            erate = 0.30 + random.uniform(0, 0.10)
            lat   = 3500.0 + random.uniform(0, 1000)
            lat_avail = True
        elif phase == "RECOVERY" and src == "ts-train-service" and dst == "ts-route-service":
            erate = 0.05 + random.uniform(0, 0.05)
            lat   = 500.0 + random.uniform(0, 200)
            lat_avail = True
        else:
            erate = 0.0
            lat   = 50.0 + random.uniform(0, 30)
            lat_avail = True

        snap.edges[(src, dst)] = EdgeTelemetry(
            source=src, destination=dst,
            request_rate=req_count / 60.0,
            request_count=req_count,
            error_rate=erate,
            latency_p95=lat,
            latency_available=lat_avail,
        )

    return snap


def build_synthetic_cpu_df(phase: str, tick_num: int) -> pd.DataFrame:
    """
    Build synthetic CPU DataFrame for the existing Rectifier pipeline.
    CPU anomaly signal drives the autoencoder score (same as Goal 1).
    """
    records = []
    timestamp = datetime.now().isoformat()
    for svc in SERVICES:
        if phase == "FAULT" and svc == "ts-train-service":
            cpu_val = 0.80
        elif phase == "RECOVERY" and svc == "ts-train-service":
            cpu_val = 0.25
        else:
            cpu_val = 0.05
        records.append({
            "timestamp": timestamp, "pod_name": f"{svc}-pod-abc",
            "service_name": svc, "pod_phase": "Running", "is_ready": True,
            "kpi_name": "cpu_usage", "value": cpu_val,
            "phase": phase, "experiment_tick": tick_num,
        })
    records.append({
        "timestamp": timestamp, "pod_name": "node", "service_name": "node",
        "pod_phase": "Running", "is_ready": True,
        "kpi_name": "node_cpu", "value": 0.1,
        "phase": phase, "experiment_tick": tick_num,
    })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_service_table(snapshot: TelemetrySnapshot):
    print()
    print("SERVICE TELEMETRY")
    print("-" * 70)
    hdr = f"{'Service':<{COL_WIDTH_SVC}}  {'CPU':>6}  {'Mem(MB)':>8}  {'Req/s':>7}  {'Error%':>7}"
    print(hdr)
    print("-" * 70)
    for svc in SERVICES:
        st = snapshot.services.get(svc)
        if st is None:
            print(f"  {svc:<{COL_WIDTH_SVC}}  {'N/A':>6}")
            continue
        mem_mb = st.memory_bytes / 1_000_000
        print(
            f"  {svc:<{COL_WIDTH_SVC}}"
            f"  {st.cpu_rate:>6.3f}"
            f"  {mem_mb:>8.1f}"
            f"  {st.request_rate:>7.2f}"
            f"  {st.error_rate * 100:>6.1f}%"
        )


def print_edge_table(snapshot: TelemetrySnapshot):
    print()
    print("EDGE TELEMETRY")
    print("-" * 75)
    hdr = f"{'Source -> Destination':<{COL_WIDTH_EDGE}}  {'Req/s':>7}  {'Error%':>7}  {'P95(ms)':>8}  {'Lat?':>4}"
    print(hdr)
    print("-" * 75)
    for (src, dst), et in snapshot.edges.items():
        label = f"{src} -> {dst}"
        lat_str = f"{et.latency_p95:>8.0f}" if et.latency_available else "     N/A"
        lat_avail_str = "Y" if et.latency_available else "N"
        print(
            f"  {label:<{COL_WIDTH_EDGE}}"
            f"  {et.request_rate:>7.2f}"
            f"  {et.error_rate * 100:>6.1f}%"
            f"  {lat_str}"
            f"  {lat_avail_str:>4}"
        )


def print_rca_table(causal_data: dict):
    print()
    print("CAUSAL RCA")
    print("-" * 60)
    print(f"  {'Service':<{COL_WIDTH_SVC}}  {'Score':>7}  {'Classification':<20}")
    print("-" * 60)
    for svc, score_data in causal_data["scores"].items():
        cls = score_data["classification"]
        score = score_data["root_cause_score"]
        print(f"  {svc:<{COL_WIDTH_SVC}}  {score:>7.3f}  {cls:<20}")


# ---------------------------------------------------------------------------
# Core tick processing
# ---------------------------------------------------------------------------
def process_tick(
    tick_num: int,
    phase: str,
    df_tick: pd.DataFrame,
    snapshot: TelemetrySnapshot,
    rect: Rectifier,
    pipe: RobustAnomalyScorePipeline,
    ddn: DynamicDecisionNetworkPhase3,
    controller: KubernetesActionController,
    results: list,
):
    # 1. Existing CPU pipeline
    x_t, _ = rect.process_tick(df_tick)
    anomaly_signals = pipe.compute_anomaly_signals(x_t)

    # 2. Enrich anomaly signals from multi-signal service telemetry
    # (Goal 4: memory and error rate are incorporated into causal scoring,
    #  not into the autoencoder anomaly signal, preserving the trained model)
    service_telemetry = snapshot.services if snapshot.services else None
    edge_telemetry    = snapshot.edges    if snapshot.edges    else None

    # 3. DDN step (unchanged — only takes anomaly_signals)
    ddn_output = ddn.step(
        anomaly_signals,
        node_pressure_flag=False,
        edge_telemetry=edge_telemetry,
        service_telemetry=service_telemetry,
    )   
    root_cause = ddn_output["root_cause"]

    # 4. Goal 3 + Goal 4 enriched causal analysis
    # The DDN already runs causal_analyzer internally, but we run an
    # enriched pass here to get the full Goal 4 evidence details for display.
    enriched_causal_data = None
    if ddn.causal_analyzer:
        enriched_causal_data = ddn_output["causal_data"]
        root_cause = ddn_output["root_cause"]

    # 5. Controller
    ddn_output["root_cause"] = root_cause
    controller.reconcile_tick(ddn_output, node_pressure_flag=False)
    pers = controller.decision_policy.root_cause_persistence.get(root_cause, 0) if root_cause != "None" else 0

    # 6. Record results
    best_action = "Do_Nothing"
    if root_cause != "None":
        eu = ddn_output["expected_utilities"].get(root_cause, {})
        if eu:
            best_action = max(eu, key=eu.get)

    results.append({
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "tick": tick_num,
        "ts_train_cpu_anomaly": anomaly_signals.get("ts-train-service", 0.0),
        "ts_route_cpu_anomaly": anomaly_signals.get("ts-route-service", 0.0),
        "ts_train_error_rate":  snapshot.services.get("ts-train-service", ServiceTelemetry("x")).error_rate,
        "ts_route_error_rate":  snapshot.services.get("ts-route-service", ServiceTelemetry("x")).error_rate,
        "root_cause": root_cause,
        "persistence": pers,
        "best_action": best_action,
    })

    return enriched_causal_data, anomaly_signals, root_cause


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment():
    random.seed(42)  # deterministic offline mode
    np.random.seed(42)

    print("=" * 65)
    print("PREFACE-DBN - GOAL 4")
    print("Multi-Signal Telemetry & Evidence-Rich RCA")
    print("=" * 65)

    # Load discovered graph
    G = load_discovered_graph()
    print(f"\nDiscovered graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    for src, dst in G.edges():
        rc = G[src][dst].get("request_count", 0)
        print(f"    {src} -> {dst}  (request_count={rc:.0f})")

    # Initialize pipeline
    rect = Rectifier(SERVICES, ["cpu_usage"], ["node_cpu"])
    pipe = RobustAnomalyScorePipeline(rect.feature_names, SERVICES)

    if not os.path.exists(MODEL_PATH):
        print(f"\n[!] Model not found at {MODEL_PATH}. Using untrained model (anomaly signals will be zero).")
    else:
        pipe.load_model(MODEL_PATH)

    ddn        = DynamicDecisionNetworkPhase3(G, num_particles=1000)
    controller = KubernetesActionController(shadow_mode=True)
    results    = []

    # Detect mode
    live_mode = is_prometheus_available()
    if live_mode:
        print("\n[+] Prometheus is reachable. Running LIVE experiment.")
    else:
        print(f"\n[!] Prometheus unreachable at {PROMETHEUS_URL}. Running OFFLINE PLAYBACK MODE.")

    # -----------------------------------------------------------------------
    # Experiment phases
    # -----------------------------------------------------------------------
    phases = [
        ("HEALTHY",   10),
        ("FAULT",     20),
        ("RECOVERY",  10),
    ]

    if live_mode:
        # Inject fault in ts-train-service before FAULT phase
        subprocess.run(
            "kubectl exec deployment/ts-train-service -- sh -c 'yes > /dev/null & yes > /dev/null &'",
            shell=True, capture_output=True
        )

    tick_num = 0
    for phase, num_ticks in phases:
        print(f"\n{'='*65}")
        print(f"  PHASE: {phase} ({num_ticks} ticks)")
        print(f"{'='*65}")

        # Inject fault ONLY when entering FAULT phase
        if live_mode and phase == "FAULT":
            print("\n[!] Injecting CPU stress into ts-train-service...")
            subprocess.run(
                "kubectl exec deployment/ts-train-service -- "
                "sh -c 'yes > /dev/null & yes > /dev/null &'",
                shell=True,
                capture_output=True
            )

        # Remove fault when entering RECOVERY phase
        if live_mode and phase == "RECOVERY":
            print("\n[+] Removing CPU stress from ts-train-service...")
            subprocess.run(
                "kubectl exec deployment/ts-train-service -- pkill -f yes",
                shell=True,
                capture_output=True
            )

        for _ in range(num_ticks):
            tick_num += 1
            print(f"\n[Tick {tick_num}] Phase: {phase}")

            if live_mode:
                df_tick  = collect_cpu_df(phase, tick_num)
                snapshot = collect_live_snapshot(G)
                time.sleep(5)
            else:
                df_tick  = build_synthetic_cpu_df(phase, tick_num)
                snapshot = build_synthetic_snapshot(phase, G)

            # Print telemetry snapshot for this tick
            print_service_table(snapshot)
            print_edge_table(snapshot)

            # Process tick
            enriched_causal_data, anomaly_signals, root_cause = process_tick(
                tick_num, phase, df_tick, snapshot,
                rect, pipe, ddn, controller, results
            )

            # Print RCA
            if enriched_causal_data:
                print_rca_table(enriched_causal_data)
            print(f"\n  Final root cause: {root_cause}")
            print("-" * 65)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  GOAL 4 EXPERIMENT REPORT")
    print("=" * 65)

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    fault_df = res_df[res_df["phase"] == "FAULT"]
    healthy_df = res_df[res_df["phase"] == "HEALTHY"]

    print("\nGoal 4 Verification:")
    if len(fault_df) > 0:
        rc_counts = fault_df["root_cause"].value_counts()
        print(f"\n  Root cause counts during FAULT phase:")
        for svc, cnt in rc_counts.items():
            print(f"    {svc}: {cnt} ticks")

        
        expected_root_cause = "ts-train-service"

        correct_ticks = int(
            (fault_df["root_cause"] == expected_root_cause).sum()
            )

        none_ticks = int(
            (fault_df["root_cause"] == "None").sum()
        )

        wrong_ticks = len(fault_df) - correct_ticks - none_ticks

        accuracy = (
            correct_ticks / len(fault_df)
            if len(fault_df) > 0
            else 0.0
        )

        print(f"\n  Expected root cause: {expected_root_cause}")
        print(f"  FAULT ticks: {len(fault_df)}")
        print(f"  Correct root-cause ticks: {correct_ticks}")
        print(f"  No root cause: {none_ticks}")
        print(f"  Incorrect root cause: {wrong_ticks}")
        print(f"  Root-cause accuracy: {accuracy * 100:.1f}%")

        if accuracy >= 0.80:
            print("\n  SUCCESS: Goal 4 root-cause accuracy >= 80%.")
        else:
            print("\n  WARNING: Goal 4 root-cause accuracy < 80%.")
        print(f"\n  Average ts-train-service error rate during FAULT: "
          f"{fault_df['ts_train_error_rate'].mean() * 100:.1f}%")
        print(f"  Average ts-train-service error rate during HEALTHY: "
          f"{healthy_df['ts_train_error_rate'].mean() * 100:.1f}%")


if __name__ == "__main__":
    run_experiment()
