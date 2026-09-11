"""
Collect healthy-baseline telemetry for autoencoder training.

Writes the CSV schema `20_audit_and_train_cpu_only.py` expects:
    timestamp, pod_name, service_name, pod_phase, is_ready, kpi_name, value

Deliberately reuses the same Prometheus query as scripts/37_record_runs.py. If
training features and inference features came from different queries, the
autoencoder's idea of "normal" would not match what it is later shown, and every
anomaly score would be biased by that mismatch rather than by real anomalies.

The autoencoder scores each tick independently - there is no recurrence - so the
sampling cadence here only controls how many samples are gathered, not the
temporal reasoning that happens later at inference time. A shorter interval is
therefore a legitimate way to build a training set faster, though consecutive
samples overlap through the rate() window and carry less independent information
than their count suggests.

Usage
-----
    python scripts/40_collect_healthy.py --minutes 20 --interval 15
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
OUT_PATH = "data/raw/healthy/phase3_healthy_telemetry_dataset.csv"

COLUMNS = ["timestamp", "pod_name", "service_name", "pod_phase",
           "is_ready", "kpi_name", "value"]

SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service",
]


def cpu_query(rate_window):
    return ('sum(rate(container_cpu_usage_seconds_total'
            '{container!="",container!="POD"}[%s])) by (pod)' % rate_window)


NODE_CPU_QUERY = (
    '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[%s]))'
)


def query(promql):
    r = requests.get("%s/api/v1/query" % PROMETHEUS_URL,
                     params={"query": promql}, timeout=15)
    data = r.json()
    if data.get("status") != "success":
        return []
    return data["data"]["result"]


def match_service(pod_name):
    for s in SERVICES:
        if s in pod_name:
            return s
    return "unknown"


def collect_tick(rate_window):
    """One snapshot, as rows in the training CSV schema."""
    timestamp = datetime.now().isoformat()
    rows = []

    for item in query(cpu_query(rate_window)):
        pod = item["metric"].get("pod", "unknown")
        service = match_service(pod)
        if service == "unknown":
            # Skip infrastructure pods; the Rectifier only models app services.
            continue
        rows.append({
            "timestamp": timestamp, "pod_name": pod, "service_name": service,
            "pod_phase": "Running", "is_ready": True,
            "kpi_name": "cpu_usage", "value": float(item["value"][1]),
        })

    # Real node utilisation, now that node-exporter is installed. Previously this
    # slot carried a hardcoded 0.1, which made the node feature a constant.
    node_result = query(NODE_CPU_QUERY % rate_window)
    node_value = float(node_result[0]["value"][1]) if node_result else 0.0
    rows.append({
        "timestamp": timestamp, "pod_name": "node", "service_name": "node",
        "pod_phase": "Running", "is_ready": True,
        "kpi_name": "node_cpu", "value": node_value,
    })

    return rows


def main():
    ap = argparse.ArgumentParser(description="Collect healthy baseline telemetry")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--rate-window", default="2m")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--append", action="store_true",
                    help="add to an existing dataset instead of replacing it")
    args = ap.parse_args()

    try:
        if requests.get("%s/-/healthy" % PROMETHEUS_URL, timeout=5).status_code != 200:
            raise SystemExit("Prometheus is not healthy at %s" % PROMETHEUS_URL)
    except requests.RequestException as exc:
        raise SystemExit("Prometheus unreachable at %s (%s)" % (PROMETHEUS_URL, exc))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not args.append or not os.path.exists(args.out):
        pd.DataFrame(columns=COLUMNS).to_csv(args.out, index=False)

    total = int(args.minutes * 60 / args.interval)
    print("Collecting %d ticks every %.0fs (%.0f min) -> %s"
          % (total, args.interval, args.minutes, args.out))
    print("The workload must stay HEALTHY throughout; do not inject faults now.\n")

    written = 0
    for tick in range(total):
        try:
            rows = collect_tick(args.rate_window)
        except Exception as exc:
            print("  [tick %d] query failed (%s); skipping" % (tick, exc.__class__.__name__))
            time.sleep(args.interval)
            continue

        if not rows:
            print("  [tick %d] no samples returned; skipping" % tick)
            time.sleep(args.interval)
            continue

        pd.DataFrame(rows)[COLUMNS].to_csv(args.out, mode="a", header=False, index=False)
        written += 1

        pods = sum(1 for r in rows if r["kpi_name"] == "cpu_usage")
        node = next((r["value"] for r in rows if r["kpi_name"] == "node_cpu"), 0.0)
        print("  [tick %3d/%d] %d pod samples, node_cpu=%.3f"
              % (tick + 1, total, pods, node))

        if tick < total - 1:
            time.sleep(args.interval)

    df = pd.read_csv(args.out)
    print("\nWrote %d ticks (%d rows) to %s" % (written, len(df), args.out))
    print("Distinct timestamps: %d" % df["timestamp"].nunique())
    print("\nTrain with:\n    python scripts/20_audit_and_train_cpu_only.py")


if __name__ == "__main__":
    main()
