"""
Analyse a live recording: disruption detection, calibration, and the
PREFACE vs PREFACE-DBN head-to-head, with the DBN evaluated over several seeds.

Why several seeds: the DBN's particle filter is stochastic. On the first
September recording a single seed reported 100% recall; across eight seeds
recall actually ranged 33-67%. Any figure quoted from one seed is luck.

Usage
-----
    python scripts/41_analyze_rerun.py
    python scripts/41_analyze_rerun.py --run-dir data/experiments/runs_v2 \\
        --baseline-dir data/experiments/runs --seeds 8

Steps
-----
    1. Disruption detection, baseline recording vs this recording, run by run
    2. Calibrate the DBN on this recording
    3. PREFACE-DBN across --seeds seeds
    4. PREFACE baseline (deterministic)
    5. Head-to-head
"""

import argparse
import importlib.util
import os
import random
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import src.dll_compat  # noqa: F401  - torch before pandas (see module docs)

import numpy as np

from src.disruption import earliness
from src.goal6_evaluator import Goal6Evaluator
from src.run_dataset import load_runs, summarize

# PREFACE's m_e + 3*s_e, in the standardised units of the anomaly signal.
PREFACE_THRESHOLD = 3.0
DEGRADING_BOUNDARY = 2.5


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt(value, pct=False):
    if value is None:
        return "n/a"
    return ("%.0f%%" % (100 * value)) if pct else ("%.2f" % value)


def score(runs, replay_fn):
    """Run one reasoner over every run and return the shared metric set."""
    evaluator = Goal6Evaluator()
    detected = correct = 0
    leads = []
    for run in runs:
        t_detect, predicted, false_alarm = replay_fn(run)
        evaluator.record_experiment(run.run_id, run.is_positive, run.t_fault, t_detect,
                                    run.injected_service, predicted, false_alarm)
        if run.is_positive and t_detect is not None:
            detected += 1
            correct += int(predicted == run.injected_service)
        if run.is_positive:
            timing = earliness(t_detect, run.t_disruption, run.t_fault, run.interval_seconds)
            if timing["earliness_ticks"] is not None:
                leads.append(timing["earliness_ticks"])
    metrics = evaluator.compute_metrics()
    return {
        "recall": metrics["recall"],
        "fpr": metrics["false_positive_rate"],
        # compute_metrics reports 0.0 when nothing was detected, which would read
        # as "always wrong"; keep it undefined instead.
        "rca": (correct / detected) if detected else None,
        "latency": metrics["detection_latency"] if detected else None,
        "earl_median": float(np.median(leads)) if leads else None,
        "earl_positive": (sum(1 for x in leads if x > 0) / len(leads)) if leads else None,
        "earl_n": len(leads),
    }


def aggregate(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return "n/a"
    a = np.array(values, dtype=float)
    return "%.2f +/- %.2f (%.2f..%.2f, n=%d)" % (a.mean(), a.std(), a.min(), a.max(), len(a))


def main():
    ap = argparse.ArgumentParser(description="Analyse a live recording")
    ap.add_argument("--run-dir", default="data/experiments/runs_v2")
    ap.add_argument("--baseline-dir", default="data/experiments/runs",
                    help="earlier recording to compare disruption detection against")
    ap.add_argument("--params-out", default="data/experiments/params/live_v2.json")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    os.chdir(REPO)
    runs = load_runs(args.run_dir)
    baseline = load_runs(args.baseline_dir)
    if not runs:
        raise SystemExit("No runs in %s" % args.run_dir)
    sources = set(r.source for r in runs)
    print("recording: %s" % summarize(runs))
    print("baseline : %s" % (summarize(baseline) if baseline else "none"))
    if sources != {"live"}:
        print("WARNING: recording is %s, not live telemetry; not evidence." % "+".join(sorted(sources)))

    # ------------------------------------------------------------------
    section("1. Disruption detection: baseline recording vs this recording")
    by_id = {r.run_id: r for r in baseline}
    print("%-18s %-22s %-12s %-12s %s" % ("run", "injected", "baseline", "this run", "lead after fault"))
    for run in runs:
        def show(r):
            if r is None:
                return "-"
            return "none" if r.t_disruption is None else "tick %d" % r.t_disruption
        lead = ""
        if run.is_positive and run.t_disruption is not None:
            ticks = run.t_disruption - run.t_fault
            lead = "%d ticks (%.0f s)" % (ticks, ticks * run.interval_seconds)
        print("%-18s %-22s %-12s %-12s %s" % (run.run_id, run.injected_service or "-",
                                              show(by_id.get(run.run_id)), show(run), lead))

    print()
    print("Peak anomaly signal on healthy runs (Degrading boundary %.1f, PREFACE threshold %.1f):"
          % (DEGRADING_BOUNDARY, PREFACE_THRESHOLD))
    for run in runs:
        if not run.is_positive:
            peaks = [max(t.anomaly_signals.values()) for t in run.ticks]
            print("  %-18s peak %.3f, %d of %d ticks at or above %.1f"
                  % (run.run_id, max(peaks), sum(1 for p in peaks if p >= DEGRADING_BOUNDARY),
                     len(peaks), DEGRADING_BOUNDARY))

    # ------------------------------------------------------------------
    section("2. Calibrate on this recording")
    result = subprocess.run([sys.executable, "scripts/38_calibrate_from_runs.py",
                             "--dir", args.run_dir, "--out", args.params_out],
                            capture_output=True, text=True)
    for line in (result.stdout + result.stderr).strip().splitlines()[-16:]:
        print(line)
    if result.returncode != 0:
        raise SystemExit("calibration failed")

    # ------------------------------------------------------------------
    g6 = load_module("g6", "scripts/36_evaluate_goal6.py")
    cmp = load_module("cmp", "scripts/39_compare_baseline.py")
    graph = g6.graph_for_services(sorted(set(s for r in runs for s in r.services)))
    params = g6.load_params_file(args.params_out)

    section("3. PREFACE-DBN across %d seeds" % args.seeds)
    dbn_rows = []
    for seed in range(args.seeds):
        np.random.seed(seed)
        random.seed(seed)
        row = score(runs, lambda r: g6.replay(r, graph, params))
        dbn_rows.append(row)
        print("  seed %d: recall %s  FPR %s  RCA %s  latency %s  earliness median %s (n=%d)"
              % (seed, fmt(row["recall"], True), fmt(row["fpr"], True), fmt(row["rca"], True),
                 fmt(row["latency"]), fmt(row["earl_median"]), row["earl_n"]))

    # ------------------------------------------------------------------
    section("4. PREFACE baseline (deterministic, threshold %.1f)" % PREFACE_THRESHOLD)
    pre = score(runs, lambda r: cmp.replay_preface(r, PREFACE_THRESHOLD))
    print("  recall %s  FPR %s  RCA %s  latency %s  earliness median %s (n=%d)"
          % (fmt(pre["recall"], True), fmt(pre["fpr"], True), fmt(pre["rca"], True),
             fmt(pre["latency"]), fmt(pre["earl_median"]), pre["earl_n"]))

    # ------------------------------------------------------------------
    section("5. Head-to-head (DBN as mean +/- std over seeds)")
    print("%-26s %-10s %s" % ("metric", "PREFACE", "PREFACE-DBN"))
    for label, key in [("recall", "recall"), ("false positive rate", "fpr"),
                       ("root cause accuracy", "rca"), ("detection latency (ticks)", "latency"),
                       ("earliness median (ticks)", "earl_median"),
                       ("positive lead fraction", "earl_positive")]:
        print("%-26s %-10s %s" % (label, fmt(pre[key]), aggregate(dbn_rows, key)))


if __name__ == "__main__":
    main()
