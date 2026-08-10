import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preface_baseline import PrefaceBaseline
from src.benchmark_metrics import MetricsEvaluator

PROMETHEUS_URL = "http://localhost:9090"

def get_historical_telemetry(timestamp_iso):
    query = 'sum(rate(container_cpu_usage_seconds_total{container!=""}[2m])) by (pod, namespace)'

    # Assuming timestamps in results.csv are naive local times (IST +05:30)
    # The runner used datetime.now().isoformat()
    prom_time = f"{timestamp_iso}+05:30"

    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query, "time": prom_time}, timeout=2)
    result = response.json()

    records = []
    services = [
        "ts-ui-dashboard", "ts-user-service", "ts-train-service",
        "ts-route-service", "ts-order-service", "ts-payment-service",
        "ts-inventory-service", "ts-station-service"
    ]

    if result.get("status") == "success":
        for item in result["data"]["result"]:
            pod_name = item["metric"].get("pod", "unknown")
            val = float(item["value"][1])
            srv_name = "unknown"
            for s in services:
                if s in pod_name:
                    srv_name = s
                    break
            records.append({
                "timestamp": timestamp_iso,
                "pod_name": pod_name,
                "service_name": srv_name,
                "pod_phase": "Running",
                "is_ready": True,
                "kpi_name": "cpu_usage",
                "value": val
            })

    records.append({
        "timestamp": timestamp_iso,
        "pod_name": "node",
        "service_name": "node",
        "pod_phase": "Running",
        "is_ready": True,
        "kpi_name": "node_cpu",
        "value": 0.1
    })

    return pd.DataFrame(records)

def main():
    exp_dir = os.path.abspath("data/experiments/phase5/cpu/single/pilot_cpu_01")
    results_csv = os.path.join(exp_dir, "results.csv")

    if not os.path.exists(results_csv):
        print("Missing results.csv")
        sys.exit(1)

    dbn_df = pd.read_csv(results_csv)

    baseline = PrefaceBaseline(
        ae_model_path="models/phase3_autoencoder_cpu_only.pth",
        healthy_dataset_path="data/raw/healthy/phase3_healthy_telemetry_dataset.csv"
    )

    print("Baseline initialized.")
    print(f"Total Error m_e: {baseline.m_e:.4f}, s_e: {baseline.s_e:.4f}, Threshold: {baseline.threshold:.4f}")

    baseline_results = []

    print("Running Baseline on historical telemetry...")
    for idx, row in dbn_df.iterrows():
        ts = row['timestamp']
        phase = row['phase']
        tick = row['tick']

        df_tick = get_historical_telemetry(ts)
        x_t, _ = baseline.rectifier.process_tick(df_tick)

        is_anomaly, root_cause, max_signal = baseline.predict(x_t)

        # Original PREFACE is memoryless, no persistence or cooldown
        decision_state = "INTERVENE" if is_anomaly else "HEALTHY"
        # Original PREFACE didn't have mitigation actions, but we track "Alarm"
        selected_action = "Alarm" if is_anomaly else "Do_Nothing"

        baseline_results.append({
            "timestamp": ts,
            "phase": phase,
            "tick": tick,
            "root_cause": root_cause,
            "p_critical": 1.0 if is_anomaly else 0.0,
            "persistence": 0,
            "decision_state": decision_state,
            "selected_action": selected_action,
            "delta_eu": 0.0
        })

    baseline_df = pd.DataFrame(baseline_results)
    baseline_out = os.path.join(exp_dir, "results_baseline.csv")
    baseline_df.to_csv(baseline_out, index=False)

    # Evaluate DBN
    eval_dbn = MetricsEvaluator(exp_dir)
    eval_dbn.load()
    dbn_metrics = eval_dbn.evaluate()

    # Evaluate Baseline
    eval_base = MetricsEvaluator(exp_dir)
    eval_base.load()
    eval_base.results_path = baseline_out
    eval_base.df = pd.read_csv(baseline_out)
    eval_base.df['timestamp'] = pd.to_datetime(eval_base.df['timestamp'])
    base_metrics = eval_base.evaluate()

    # Output comparison
    out_dir = os.path.abspath("data/experiments/phase5/results")
    os.makedirs(out_dir, exist_ok=True)

    comparison = {
        "experiment_id": "pilot_cpu_01",
        "preface_baseline": base_metrics,
        "preface_dbn": dbn_metrics
    }

    with open(os.path.join(out_dir, "preface_vs_dbn_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=4)

    comp_df = pd.DataFrame({
        "Metric": ["Detection Time (Td)", "Reaction Interval (s)", "First Correct Root Cause",
                   "Strong Localization", "Weak Localization", "Overall Localization",
                   "Intervention Time (Te)", "Detection Lead Time (s)", "Eligibility Lead Time (s)"],
        "PREFACE Baseline": [
            base_metrics.get("Td"),
            base_metrics.get("reaction_interval"),
            base_metrics.get("Trc"),
            base_metrics.get("strong_localization"),
            base_metrics.get("weak_localization"),
            base_metrics.get("overall_localization"),
            base_metrics.get("Te"),
            base_metrics.get("lead_time", {}).get("detection_lead_time"),
            base_metrics.get("lead_time", {}).get("eligibility_lead_time"),
        ],
        "PREFACE-DBN": [
            dbn_metrics.get("Td"),
            dbn_metrics.get("reaction_interval"),
            dbn_metrics.get("Trc"),
            dbn_metrics.get("strong_localization"),
            dbn_metrics.get("weak_localization"),
            dbn_metrics.get("overall_localization"),
            dbn_metrics.get("Te"),
            dbn_metrics.get("lead_time", {}).get("detection_lead_time"),
            dbn_metrics.get("lead_time", {}).get("eligibility_lead_time"),
        ]
    })
    comp_df.to_csv(os.path.join(out_dir, "preface_vs_dbn_comparison.csv"), index=False)

    print("Comparison complete. Check results in data/experiments/phase5/results/")
    print(comp_df.to_string())

if __name__ == "__main__":
    main()
