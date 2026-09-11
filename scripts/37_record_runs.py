"""
Record experiment runs for the Goal 6 evaluation.

This is the data-collection half of the split introduced with src/run_dataset.py.
It produces one JSON file per run containing per-tick anomaly signals plus the
ground truth, which 36_evaluate_goal6.py then replays. Collection and scoring
stay in separate processes so a result can never be produced by the same code
that produced its data.

Modes
-----
  --mode live       Drive the real cluster. Each tick queries Prometheus, runs
                    the Rectifier and the autoencoder, and stores the resulting
                    anomaly signals. Faulty runs inject a real fault partway
                    through. Runs are stamped source="live" and are the only
                    ones the evaluator will report as a system evaluation.

  --mode synthetic  Generate runs from a simulator. Stamped source="synthetic",
                    so the evaluator refuses to report them as evidence. Useful
                    for exercising this script and the evaluator without a
                    cluster.

Tick interval
-------------
Defaults to 60s, matching the PREFACE paper's 1-minute cadence and the project
design docs. Earlier experiments in this repo used roughly 5s ticks, which is
shorter than the 2-minute window of the `rate()` queries feeding them, so
consecutive ticks shared almost all of their underlying data. Keep this at or
above the rate window unless you have a specific reason not to.

Examples
--------
    # Check the plan without touching anything
    python scripts/37_record_runs.py --mode live --dry-run

    # 6 healthy + 6 faulty runs against the cluster
    python scripts/37_record_runs.py --mode live --healthy-runs 6 --faulty-runs 6

    # Exercise the pipeline with no cluster
    python scripts/37_record_runs.py --mode synthetic --healthy-runs 3 --faulty-runs 3 \
        --out data/experiments/runs_synthetic
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Must precede pandas: pandas pulls in pyarrow, whose native libraries stop
# torch's c10.dll from initialising on Windows. See src/dll_compat.py.
import src.dll_compat  # noqa: F401

import numpy as np
import pandas as pd

from src.run_dataset import RunRecord, save_run, SOURCE_LIVE, SOURCE_SYNTHETIC
from src.disruption import detect_disruption

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
MODEL_PATH = "models/phase3_autoencoder_cpu_only.pth"
DEFAULT_OUT = "data/experiments/runs"

DEFAULT_SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service",
]

# Services eligible for fault injection. The UI dashboard is excluded because it
# is the topology root; injecting there gives the localizer nothing to localize.
DEFAULT_TARGETS = ["ts-train-service", "ts-route-service", "ts-order-service"]

def cpu_query(rate_window="2m"):
    return ('sum(rate(container_cpu_usage_seconds_total'
            '{container!="",container!="POD"}[%s])) by (pod)' % rate_window)


def node_cpu_query(rate_window="2m"):
    """Real node utilisation, from node-exporter."""
    return '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[%s]))' % rate_window

# User-facing signals, from Istio's telemetry at the mesh level. These are what
# a user actually experiences, and what src.disruption tests to find the moment
# the failure became visible. Without them earliness cannot be computed.
WORKLOAD_QUERIES = {
    "p95_latency_ms": (
        'histogram_quantile(0.95, sum(rate('
        'istio_request_duration_milliseconds_bucket{reporter="destination"}[2m])) by (le))'
    ),
    "error_rate": (
        'sum(rate(istio_requests_total{reporter="destination",response_code=~"5.."}[2m]))'
        ' / clamp_min(sum(rate(istio_requests_total{reporter="destination"}[2m])), 0.001)'
    ),
    "request_rate": (
        'sum(rate(istio_requests_total{reporter="destination"}[2m]))'
    ),
}


# ---------------------------------------------------------------------------
# Fault injectors
# ---------------------------------------------------------------------------
class ExecInjector:
    """
    Burn CPU inside the target pod with busy loops.

    Simple and dependency-free, but it saturates instantly, which is unlike most
    real degradation. Prefer ChaosMeshInjector when Chaos Mesh is installed.
    """

    name = "exec"

    def __init__(self, workers=2):
        self.workers = workers

    def start(self, service):
        loops = " ".join(["yes > /dev/null &"] * self.workers)
        _kubectl(["exec", "deployment/%s" % service, "--", "sh", "-c", loops])

    def stop(self, service):
        _kubectl(["exec", "deployment/%s" % service, "--", "pkill", "-f", "yes"], check=False)


class ChaosMeshInjector:
    """Inject CPU stress through a Chaos Mesh StressChaos resource."""

    name = "chaos"

    MANIFEST = """apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: preface-cpu-{service}
  namespace: default
spec:
  mode: all
  selector:
    namespaces: [default]
    labelSelectors:
      app: {service}
  stressors:
    cpu:
      workers: {workers}
      load: {load}
  duration: '{duration}'
"""

    def __init__(self, workers=2, load=80, duration="30m"):
        self.workers = workers
        self.load = load
        self.duration = duration

    def _manifest(self, service):
        return self.MANIFEST.format(
            service=service, workers=self.workers,
            load=self.load, duration=self.duration,
        )

    def start(self, service):
        _kubectl(["apply", "-f", "-"], stdin=self._manifest(service))

    def stop(self, service):
        _kubectl(
            ["delete", "stresschaos", "preface-cpu-%s" % service, "--ignore-not-found"],
            check=False,
        )


def _kubectl(args, stdin=None, check=True):
    cmd = ["kubectl"] + args
    result = subprocess.run(
        cmd, input=stdin, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "kubectl %s failed:\n%s" % (" ".join(args), result.stderr.strip())
        )
    return result


# ---------------------------------------------------------------------------
# Live collection
# ---------------------------------------------------------------------------
class LiveSignalSource:
    """Prometheus -> Rectifier -> autoencoder -> per-service anomaly signals."""

    def __init__(self, services, model_path=MODEL_PATH, prometheus_url=PROMETHEUS_URL,
                 rate_window="2m"):
        # Imported here so synthetic mode does not need torch installed.
        import requests
        from src.rectifier import Rectifier
        from src.autoencoder_phase3 import RobustAnomalyScorePipeline

        self._requests = requests
        self.services = services
        self.prometheus_url = prometheus_url
        self.rate_window = rate_window

        self.rectifier = Rectifier(services, ["cpu_usage"], ["node_cpu"])
        self.pipeline = RobustAnomalyScorePipeline(self.rectifier.feature_names, services)

        if not os.path.exists(model_path):
            raise SystemExit(
                "\nNo trained autoencoder at %s.\n"
                "Train one first (scripts/20_audit_and_train_cpu_only.py); without it "
                "every anomaly signal is meaningless.\n" % model_path
            )
        self.pipeline.load_model(model_path)

    def check_prometheus(self):
        try:
            r = self._requests.get("%s/-/healthy" % self.prometheus_url, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _query(self, promql):
        r = self._requests.get(
            "%s/api/v1/query" % self.prometheus_url,
            params={"query": promql}, timeout=10,
        )
        data = r.json()
        if data.get("status") != "success":
            return []
        return data["data"]["result"]

    def _match(self, pod_name):
        for s in self.services:
            if s in pod_name:
                return s
        return "unknown"

    def workload(self):
        """
        One tick of user-facing metrics.

        Returns {} when Istio telemetry is absent, which the caller treats as
        "earliness not measurable for this run" rather than as healthy. Silently
        reporting zeros here would fabricate a disruption timestamp.
        """
        out = {}
        for name, promql in WORKLOAD_QUERIES.items():
            try:
                result = self._query(promql)
            except Exception:
                continue
            if not result:
                continue
            try:
                value = float(result[0]["value"][1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if value != value:  # NaN, which histogram_quantile returns with no data
                continue
            out[name] = value
        return out

    def signals(self):
        """One tick of anomaly signals, keyed by service."""
        timestamp = datetime.now().isoformat()
        records = []
        for item in self._query(cpu_query(self.rate_window)):
            pod = item["metric"].get("pod", "unknown")
            records.append({
                "timestamp": timestamp,
                "pod_name": pod,
                "service_name": self._match(pod),
                "pod_phase": "Running",
                "is_ready": True,
                "kpi_name": "cpu_usage",
                "value": float(item["value"][1]),
            })

        # Real node utilisation from node-exporter. This slot used to carry a
        # hardcoded 0.1; the training collector now records the true value, so a
        # constant here would mean the autoencoder sees a feature at inference
        # that never appeared during training.
        node_result = self._query(node_cpu_query(self.rate_window))
        node_value = float(node_result[0]["value"][1]) if node_result else 0.0
        records.append({
            "timestamp": timestamp, "pod_name": "node", "service_name": "node",
            "pod_phase": "Running", "is_ready": True,
            "kpi_name": "node_cpu", "value": node_value,
        })

        x_t, _ = self.rectifier.process_tick(pd.DataFrame(records))
        return self.pipeline.compute_anomaly_signals(x_t)


def record_live_run(run_id, source_obj, services, is_positive, target,
                    pre_ticks, post_ticks, interval, injector, fault_type):
    """Drive one run against the cluster and return it as a RunRecord."""
    run = RunRecord(
        run_id=run_id,
        source=SOURCE_LIVE,
        is_positive=is_positive,
        services=services,
        t_fault=pre_ticks if is_positive else None,
        injected_service=target if is_positive else None,
        fault_type=fault_type if is_positive else None,
    )

    total = pre_ticks + post_ticks if is_positive else pre_ticks + post_ticks
    injected = False
    try:
        for tick in range(total):
            if is_positive and tick == pre_ticks:
                print("    [tick %d] injecting %s into %s" % (tick, fault_type, target))
                injector.start(target)
                injected = True

            run.add_tick(tick, source_obj.signals(), source_obj.workload())

            last = run.ticks[-1]
            top = max(last.anomaly_signals.items(), key=lambda kv: kv[1])
            lat = last.workload.get("p95_latency_ms")
            extra = "" if lat is None else "  p95=%.0fms err=%.1f%%" % (
                lat, 100.0 * last.workload.get("error_rate", 0.0))
            print("    [tick %2d/%d] max signal %s=%.3f%s"
                  % (tick + 1, total, top[0], top[1], extra))

            if tick < total - 1:
                time.sleep(interval)
    finally:
        if injected:
            print("    stopping fault on %s" % target)
            injector.stop(target)

    return run


# ---------------------------------------------------------------------------
# Synthetic collection
# ---------------------------------------------------------------------------
def record_synthetic_run(run_id, services, is_positive, target,
                         pre_ticks, post_ticks, rng):
    """
    Generate one run from a simulator.

    Deliberately crude: this exists to exercise the recorder and evaluator, not
    to stand in for telemetry. Runs are stamped source="synthetic" so the
    evaluator will not report them as a system evaluation.
    """
    run = RunRecord(
        run_id=run_id,
        source=SOURCE_SYNTHETIC,
        is_positive=is_positive,
        services=services,
        t_fault=pre_ticks if is_positive else None,
        injected_service=target if is_positive else None,
        fault_type="cpu" if is_positive else None,
    )
    upstream = services[0]
    for tick in range(pre_ticks + post_ticks):
        signals = {}
        faulting = is_positive and tick >= pre_ticks
        for s in services:
            if faulting and s == target:
                signals[s] = float(rng.normal(5.0, 0.5))
            elif faulting and s == upstream:
                signals[s] = float(rng.normal(1.5, 0.5))
            else:
                signals[s] = float(max(0.0, rng.normal(0.1, 0.1)))

        # User-facing metrics degrade only after the fault, and gradually, so
        # there is a real error interval between the anomaly appearing and the
        # disruption. A fault that hurt users instantly would leave nothing to
        # predict.
        elapsed = (tick - pre_ticks) if faulting else -1
        if elapsed >= 0:
            latency = 50.0 + max(0, elapsed - 2) * 40.0 + rng.normal(0, 3)
            err = 0.0 if elapsed < 5 else min(0.5, (elapsed - 4) * 0.05)
        else:
            latency = 50.0 + rng.normal(0, 3)
            err = 0.0
        workload = {
            "p95_latency_ms": float(max(1.0, latency)),
            "error_rate": float(err),
            "request_rate": float(max(0.0, rng.normal(20.0, 2.0))),
        }
        run.add_tick(tick, signals, workload)
    return run


# ---------------------------------------------------------------------------
# Disruption annotation
# ---------------------------------------------------------------------------
def annotate_disruption(run, baseline_ticks, interval_seconds):
    """
    Find and store the tick at which the run became user-visibly disrupted.

    Runs it on healthy runs too: a healthy run that reports a disruption means
    the detector is firing on noise, which is worth knowing before the number is
    used anywhere.
    """
    run.interval_seconds = float(interval_seconds)

    if not run.has_workload():
        run.t_disruption = None
        print("      no workload data - earliness will not be computable")
        return

    result = detect_disruption(run.workload_series(), baseline_ticks=baseline_ticks)
    run.t_disruption = result.t_disruption

    if result.t_disruption is None:
        print("      no disruption detected (%s)" % result.reason)
    elif run.is_positive:
        lead = result.t_disruption - run.t_fault
        print("      disruption at tick %d, %d ticks (%.0fs) after injection"
              % (result.t_disruption, lead, lead * interval_seconds))
    else:
        print("      WARNING: disruption at tick %d on a HEALTHY run - "
              "detector may be firing on noise" % result.t_disruption)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight_live(services, targets, injector_name, out_dir, interval,
                   pre_ticks, post_ticks, healthy, faulty):
    """Report everything that must be true before a live recording session."""
    problems = []

    print("\n--- Preflight ---")

    have_kubectl = shutil.which("kubectl") is not None
    cluster_reachable = False
    if not have_kubectl:
        problems.append("kubectl is not on PATH")
        print("  kubectl        MISSING")
    else:
        r = subprocess.run(["kubectl", "get", "nodes"], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("kubectl cannot reach a cluster")
            print("  cluster        UNREACHABLE")
        else:
            cluster_reachable = True
            print("  cluster        ok (%d nodes)" % (len(r.stdout.strip().splitlines()) - 1))

    if not os.path.exists(MODEL_PATH):
        problems.append("no trained autoencoder at %s" % MODEL_PATH)
        print("  autoencoder    MISSING (%s)" % MODEL_PATH)
    else:
        print("  autoencoder    ok")

    try:
        import requests
        ok = requests.get("%s/-/healthy" % PROMETHEUS_URL, timeout=3).status_code == 200
        print("  prometheus     %s (%s)" % ("ok" if ok else "UNHEALTHY", PROMETHEUS_URL))
        if not ok:
            problems.append("Prometheus not healthy at %s" % PROMETHEUS_URL)
    except Exception as exc:
        problems.append("Prometheus unreachable at %s (%s)" % (PROMETHEUS_URL, exc.__class__.__name__))
        print("  prometheus     UNREACHABLE (%s)" % PROMETHEUS_URL)

    if injector_name == "chaos":
        if not cluster_reachable:
            print("  chaos mesh     UNKNOWN (needs a reachable cluster)")
        else:
            r = subprocess.run(
                ["kubectl", "get", "crd", "stresschaos.chaos-mesh.org"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                problems.append(
                    "Chaos Mesh is not installed (no StressChaos CRD); "
                    "install it or pass --injector exec"
                )
                print("  chaos mesh     NOT INSTALLED")
            else:
                print("  chaos mesh     ok")

    ticks = pre_ticks + post_ticks
    runs = healthy + faulty
    minutes = runs * ticks * interval / 60.0
    print("\n--- Plan ---")
    print("  runs           %d healthy + %d faulty" % (healthy, faulty))
    print("  ticks per run  %d (%d pre-fault + %d post)" % (ticks, pre_ticks, post_ticks))
    print("  interval       %ds" % interval)
    print("  targets        %s" % ", ".join(targets))
    print("  injector       %s" % injector_name)
    print("  output         %s" % out_dir)
    print("  wall time      ~%.0f minutes" % minutes)

    if problems:
        print("\n--- Blocked ---")
        for p in problems:
            print("  - %s" % p)
        return False

    print("\nPreflight passed.")
    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Record runs for the Goal 6 evaluation")
    ap.add_argument("--mode", choices=["live", "synthetic"], default="live")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--healthy-runs", type=int, default=5)
    ap.add_argument("--faulty-runs", type=int, default=5)
    ap.add_argument("--pre-fault-ticks", type=int, default=10)
    ap.add_argument("--post-fault-ticks", type=int, default=20)
    ap.add_argument("--interval", type=int, default=60,
                    help="seconds between ticks (default 60, matching the paper)")
    ap.add_argument("--rate-window", default="2m",
                    help="PromQL rate() window (default 2m). Keep it at or below "
                         "the tick interval so consecutive ticks are not mostly "
                         "the same data.")
    ap.add_argument("--services", default=",".join(DEFAULT_SERVICES))
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--injector", choices=["chaos", "exec"], default="chaos")
    ap.add_argument("--fault-type", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true",
                    help="run preflight and print the plan, record nothing")
    args = ap.parse_args()

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]

    unknown = [t for t in targets if t not in services]
    if unknown:
        raise SystemExit("Target(s) not in --services: %s" % ", ".join(unknown))

    if args.mode == "live":
        ok = preflight_live(
            services, targets, args.injector, args.out, args.interval,
            args.pre_fault_ticks, args.post_fault_ticks,
            args.healthy_runs, args.faulty_runs,
        )
        if args.dry_run:
            return
        if not ok:
            raise SystemExit(
                "\nPreflight failed. Fix the items above, or use --mode synthetic "
                "to exercise the pipeline without a cluster.\n"
            )
    elif args.dry_run:
        print("Dry run: synthetic mode needs no cluster; %d runs would be written to %s"
              % (args.healthy_runs + args.faulty_runs, args.out))
        return

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    source_obj = None
    injector = None
    if args.mode == "live":
        source_obj = LiveSignalSource(services, rate_window=args.rate_window)
        injector = ChaosMeshInjector() if args.injector == "chaos" else ExecInjector()

    plan = ([(False, None)] * args.healthy_runs +
            [(True, targets[i % len(targets)]) for i in range(args.faulty_runs)])

    print("\nRecording %d runs into %s\n" % (len(plan), args.out))
    written = []
    for i, (is_positive, target) in enumerate(plan):
        kind = "faulty/%s" % target if is_positive else "healthy"
        run_id = "%s_%s_%03d" % (args.mode, "fault" if is_positive else "healthy", i)
        print("  [%d/%d] %s (%s)" % (i + 1, len(plan), run_id, kind))

        if args.mode == "live":
            run = record_live_run(
                run_id, source_obj, services, is_positive, target,
                args.pre_fault_ticks, args.post_fault_ticks,
                args.interval, injector, args.fault_type,
            )
        else:
            run = record_synthetic_run(
                run_id, services, is_positive, target,
                args.pre_fault_ticks, args.post_fault_ticks, rng,
            )

        annotate_disruption(run, args.pre_fault_ticks, args.interval)
        written.append(save_run(run, args.out))

    print("\nWrote %d runs to %s" % (len(written), args.out))
    print("\nEvaluate them with:")
    print("    python scripts/36_evaluate_goal6.py --source recorded --dir %s" % args.out)


if __name__ == "__main__":
    main()
