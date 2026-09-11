"""
Head-to-head: original PREFACE reasoning vs PREFACE-DBN, on identical data.

Both reasoners consume the *same* recorded anomaly signals, so the RECTIFIER and
the autoencoder are held constant and only the decision layer differs. Any gap in
the results is therefore attributable to the reasoner, which is the comparison
the project exists to make.

The two reasoners
-----------------
PREFACE (baseline)
    Memoryless. At each tick, alarm if the largest per-service anomaly signal
    exceeds a fixed threshold; localize by ranking the signals and taking the
    top one. This mirrors the decision logic in src/preface_baseline.py, but
    driven from recorded signals rather than raw vectors so both sides see
    identical input. Because `a_t^s` is already standardized against the healthy
    training distribution, PREFACE's `m_e + 3*s_e` rule corresponds to a
    threshold of 3.0 in these units.

PREFACE-DBN
    The particle filter over hidden health states plus the directional causal
    analyzer, exactly as the evaluator runs it.

Usage
-----
    python scripts/39_compare_baseline.py --dir data/experiments/runs
    python scripts/39_compare_baseline.py --dir data/experiments/runs \\
        --params data/experiments/params/live.json
"""

import argparse
import importlib.util
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.goal6_evaluator import Goal6Evaluator
from src.run_dataset import load_runs, summarize, SOURCE_LIVE
from src.disruption import earliness

RESULTS_DIR = "data/experiments/comparison"

# PREFACE's m_e + 3*s_e, expressed in the standardized units of a_t^s.
DEFAULT_THRESHOLD = 3.0


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location(
        "g6", os.path.join(os.path.dirname(__file__), "36_evaluate_goal6.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Reasoner A: original PREFACE
# ---------------------------------------------------------------------------
def replay_preface(run, threshold):
    """
    Memoryless threshold + rank localization.

    No state is carried between ticks - that absence is the point of the
    comparison, since it is exactly what the DBN adds.
    """
    t_detect = None
    predicted_rc = "None"
    fp_occurred = False

    for rec in run.ticks:
        if not rec.anomaly_signals:
            continue
        top_service, top_value = max(rec.anomaly_signals.items(), key=lambda kv: kv[1])
        if top_value <= threshold:
            continue
        if run.is_positive:
            if rec.tick >= run.t_fault and t_detect is None:
                t_detect = rec.tick
                predicted_rc = top_service
        else:
            fp_occurred = True

    return t_detect, predicted_rc, fp_occurred


# ---------------------------------------------------------------------------
def score(runs, replay_fn, label):
    """Run one reasoner over every run and collect the shared metric set."""
    evaluator = Goal6Evaluator()
    breakdown = []
    timing = []

    for run in runs:
        t_detect, predicted_rc, fp = replay_fn(run)
        evaluator.record_experiment(
            experiment_id=run.run_id,
            is_positive=run.is_positive,
            t_fault=run.t_fault,
            t_detect=t_detect,
            actual_rc=run.injected_service,
            predicted_rc=predicted_rc,
            fp_occurred=fp,
        )
        if run.is_positive and t_detect is not None:
            breakdown.append((run.injected_service, predicted_rc))
        if run.is_positive:
            timing.append(earliness(t_detect, run.t_disruption, run.t_fault,
                                    run.interval_seconds))

    metrics = evaluator.compute_metrics()

    lead = [t["earliness_ticks"] for t in timing if t.get("earliness_ticks") is not None]
    metrics["earliness_measurable"] = len(lead)
    metrics["earliness_median_ticks"] = float(np.median(lead)) if lead else None
    metrics["earliness_positive_frac"] = (
        float(sum(1 for x in lead if x > 0)) / len(lead) if lead else None
    )
    metrics["reasoner"] = label
    return metrics, breakdown


def fmt(value, pct=False, signed=False):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if pct:
            return "%.1f%%" % (100.0 * value)
        return ("%+.2f" if signed else "%.2f") % value
    return str(value)


def main():
    ap = argparse.ArgumentParser(description="PREFACE vs PREFACE-DBN on identical runs")
    ap.add_argument("--dir", default="data/experiments/runs")
    ap.add_argument("--params", default=None, help="calibrated parameters JSON")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="PREFACE alarm threshold in standardized units (default 3.0)")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit("\nNo recorded runs in '%s'.\n" % args.dir)

    sources = set(r.source for r in runs)
    if len(sources) > 1:
        raise SystemExit("\n'%s' mixes %s runs; separate them.\n" % (args.dir, sorted(sources)))
    source = sources.pop()

    print("Data: %s" % summarize(runs))
    if source != SOURCE_LIVE:
        print("\n  !! These runs are %s. This comparison exercises both reasoners" % source)
        print("     but is NOT evidence about real failure prediction.\n")

    g6 = _load_evaluator_module()
    services = sorted(set(s for r in runs for s in r.services))
    G = g6.graph_for_services(services)

    if args.params:
        params = g6.load_params_file(args.params)
    else:
        print("[params] no --params given; using Goal 5 simulated priors")
        params = g6.learned_parameters(G)

    np.random.seed(1337)
    random.seed(1337)

    preface, preface_bd = score(
        runs, lambda r: replay_preface(r, args.threshold), "PREFACE"
    )
    dbn, dbn_bd = score(
        runs, lambda r: g6.replay(r, G, params), "PREFACE-DBN"
    )

    rows = [
        ("Precision",              "precision",              True,  False),
        ("Recall",                 "recall",                 True,  False),
        ("F1",                     "f1_score",               True,  False),
        ("False positive rate",    "false_positive_rate",    True,  False),
        ("Root cause accuracy",    "root_cause_accuracy",    True,  False),
        ("Detection latency (tk)", "detection_latency",      False, False),
        ("Earliness median (tk)",  "earliness_median_ticks", False, True),
        ("Positive lead time",     "earliness_positive_frac", True, False),
    ]

    width = 24
    print("\n" + "=" * 64)
    print("%-*s %14s %14s   %s" % (width, "metric", "PREFACE", "PREFACE-DBN", "winner"))
    print("-" * 64)
    for label, key, pct, signed in rows:
        a, b = preface.get(key), dbn.get(key)
        winner = ""
        if isinstance(a, float) and isinstance(b, float):
            lower_better = key in ("false_positive_rate", "detection_latency")
            if abs(a - b) < 1e-9:
                winner = "tie"
            elif (a < b) == lower_better:
                winner = "PREFACE"
            else:
                winner = "PREFACE-DBN"
        print("%-*s %14s %14s   %s"
              % (width, label, fmt(a, pct, signed), fmt(b, pct, signed), winner))
    print("=" * 64)

    print("\nRoot-cause errors")
    for name, bd in (("PREFACE", preface_bd), ("PREFACE-DBN", dbn_bd)):
        wrong = [(a, p) for a, p in bd if a != p]
        if not wrong:
            print("  %-12s none" % name)
            continue
        counts = {}
        for pair in wrong:
            counts[pair] = counts.get(pair, 0) + 1
        print("  %-12s %d/%d wrong" % (name, len(wrong), len(bd)))
        for (actual, predicted), c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print("      %s -> %s (x%d)" % (actual, predicted, c))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "preface_vs_dbn_%s.csv" % source)
    pd.DataFrame([preface, dbn]).to_csv(out, index=False)
    print("\nSaved to %s" % out)

    if source != SOURCE_LIVE:
        print("\n  Reminder: %s data. Not a result." % source)


if __name__ == "__main__":
    main()
