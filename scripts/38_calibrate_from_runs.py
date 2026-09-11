"""
Calibrate the DBN's parameters from recorded runs.

Replaces the Goal 5 arrangement, where `generate_synthetic_data()` sampled from a
hand-written ground truth (`true_mu = [0.1, 3.0, 5.5]`) and EM then recovered
those same values. That is a valid unit test of the estimator - and worth
keeping as one - but it is not calibration, because no telemetry is involved.

Here the labels come from the fault-injection schedule (see src.weak_labels) and
the anomaly signals come from recorded runs, so the fitted emission means,
transition matrix and topological modifiers describe the system's actual
behaviour.

Usage
-----
    python scripts/38_calibrate_from_runs.py --dir data/experiments/runs
    python scripts/38_calibrate_from_runs.py --dir data/experiments/runs \\
        --out data/experiments/params/live.json

Then evaluate with those parameters:
    python scripts/36_evaluate_goal6.py --source recorded \\
        --dir data/experiments/runs --params data/experiments/params/live.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dbn_learner import DBNParameterLearner
from src.run_dataset import load_runs, summarize, SOURCE_LIVE
from src.weak_labels import build_training_set, label_summary, check_balance

DEFAULT_OUT = "data/experiments/params/calibrated.json"


def graph_for(services):
    """Reuse the evaluator's topology logic so both see the same graph."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "g6", os.path.join(os.path.dirname(__file__), "36_evaluate_goal6.py")
    )
    g6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g6)
    return g6.graph_for_services(services)


def main():
    ap = argparse.ArgumentParser(description="Calibrate DBN parameters from recorded runs")
    ap.add_argument("--dir", default="data/experiments/runs")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--propagate-downstream", action="store_true",
                    help="also label descendants of the injected service as Degrading "
                         "(an assumption about propagation, off by default)")
    ap.add_argument("--min-per-state", type=int, default=30)
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(
            "\nNo recorded runs in '%s'. Record some with scripts/37_record_runs.py\n"
            % args.dir
        )

    sources = set(r.source for r in runs)
    print("Loaded: %s" % summarize(runs))
    if sources != {SOURCE_LIVE}:
        print("\n  WARNING: these runs are %s, not live telemetry." % "+".join(sorted(sources)))
        print("  Parameters fitted to simulated signals describe the simulator,")
        print("  not the system. Useful for testing this script; not for results.\n")

    services = sorted(set(s for r in runs for s in r.services))
    G = graph_for(services)

    states, scores, sequences = build_training_set(
        runs, graph=G, propagate_downstream=args.propagate_downstream
    )
    print("\n%s" % label_summary(states))

    warnings = check_balance(states, args.min_per_state)
    for w in warnings:
        print("  WARNING: %s" % w)

    faulty = sum(1 for r in runs if r.is_positive)
    with_disruption = sum(1 for r in runs if r.is_positive and r.t_disruption is not None)
    print("  %d/%d faulty runs reached a user-visible disruption" % (with_disruption, faulty))
    if faulty and not with_disruption:
        print("  WARNING: with no disruption timestamps, the Degrading/Critical split")
        print("  falls back to a fixed window rather than the real error interval.")

    learner = DBNParameterLearner()
    mu, sigma = learner.calibrate_emissions(states, scores)
    T = learner.calibrate_transitions(sequences)
    topological = learner.calibrate_topological_influences(sequences, G)

    print("\n--- Calibrated parameters ---")
    print("  mu    (Normal, Degrading, Critical) = %s" % np.round(mu, 4))
    print("  sigma                               = %s" % np.round(sigma, 4))
    print("  transition matrix P(H_t | H_t-1):")
    for i, row in enumerate(np.round(T, 4)):
        print("    from %-9s -> %s" % (("Normal", "Degrading", "Critical")[i], row))

    if mu[0] >= mu[1] or mu[1] >= mu[2]:
        print("\n  WARNING: emission means are not increasing across states")
        print("  (expected Normal < Degrading < Critical). The anomaly signal may not")
        print("  separate these states, which would make P(Critical) untrustworthy.")

    payload = {
        "source": "+".join(sorted(sources)),
        "num_runs": len(runs),
        "num_labels": int(len(states)),
        "propagate_downstream": bool(args.propagate_downstream),
        "services": services,
        "mu": [float(x) for x in mu],
        "sigma": [float(x) for x in sigma],
        "transition": [[float(x) for x in row] for row in T],
        "topological": {
            str(k): [float(x) for x in v] for k, v in (topological or {}).items()
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\nSaved to %s" % args.out)
    print("\nEvaluate with these parameters:")
    print("    python scripts/36_evaluate_goal6.py --source recorded --dir %s --params %s"
          % (args.dir, args.out))


if __name__ == "__main__":
    main()
