"""
Goal 6: system evaluation.

Two sources, and the difference between them is the whole point:

  --source recorded   Replay real runs recorded by 37_record_runs.py, whose
                      anomaly signals came from Prometheus through the Rectifier
                      and the autoencoder. This is the only mode whose numbers
                      are evidence about the system.

  --source synthetic  Generate signals inline from np.random.normal and score
                      the model on them. The classes are separated by roughly
                      ten standard deviations, so any threshold between 0.5 and
                      4.0 scores 100%. That makes this a useful smoke test that
                      the filter and the metric code run end to end, and NOT a
                      system evaluation. Results print with a banner saying so.

Usage
-----
    python scripts/36_evaluate_goal6.py --source recorded
    python scripts/36_evaluate_goal6.py --source recorded --dir data/experiments/runs
    python scripts/36_evaluate_goal6.py --source synthetic --trials 100
"""

import argparse
import importlib.util
import json
import os
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.dbn_learner import DBNParameterLearner
from src.goal6_evaluator import Goal6Evaluator
from src.run_dataset import RunRecord, load_runs, summarize, SOURCE_LIVE, SOURCE_SYNTHETIC
from src.disruption import earliness

DEFAULT_RUN_DIR = "data/experiments/runs"
DISCOVERED_GRAPH = "data/experiments/discovered_service_graph.json"
RESULTS_DIR = "data/experiments/goal6"

# A service is alarming when its Critical posterior exceeds this. Shared by both
# sources so their numbers stay comparable.
ALARM_THRESHOLD = 0.5

NUM_PARTICLES = 500


# ---------------------------------------------------------------------------
# Service graph
# ---------------------------------------------------------------------------
def mock_graph():
    """Fallback topology used by the synthetic source."""
    G = nx.DiGraph()
    services = ["ts-ui-dashboard", "ts-train-service", "ts-route-service", "ts-order-service"]
    G.add_nodes_from(services)
    G.add_edge("ts-ui-dashboard", "ts-train-service")
    G.add_edge("ts-ui-dashboard", "ts-route-service")
    G.add_edge("ts-ui-dashboard", "ts-order-service")
    return G


def graph_for_services(services):
    """
    Build the topology for a recorded run set.

    Prefers the graph discovered from live Istio telemetry, narrowed to the
    services the runs actually contain. Falls back to a star topology only if no
    discovered graph exists, and says so, because a wrong graph quietly degrades
    root-cause localization.
    """
    services = list(services)
    if os.path.exists(DISCOVERED_GRAPH):
        with open(DISCOVERED_GRAPH, encoding="utf-8") as fh:
            data = json.load(fh)
        G = nx.DiGraph()
        G.add_nodes_from(services)
        kept = 0
        for edge in data.get("edges", []):
            src, dst = edge.get("source"), edge.get("destination")
            if src in G and dst in G:
                G.add_edge(src, dst)
                kept += 1
        print("[graph] discovered topology: %d nodes, %d edges" % (G.number_of_nodes(), kept))
        if not nx.is_directed_acyclic_graph(G):
            # The localizer walks predecessors; a cycle makes "upstream-most"
            # undefined. Condense strongly-connected components to restore a DAG.
            G = nx.condensation(G)
            print("[graph] cycles found - condensed to DAG")
        return G

    print("[graph] WARNING: %s not found - falling back to star topology." % DISCOVERED_GRAPH)
    print("[graph] Run scripts/26_discover_service_graph.py for the real graph.")
    G = nx.DiGraph()
    G.add_nodes_from(services)
    root = services[0]
    for s in services[1:]:
        G.add_edge(root, s)
    return G


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def load_params_file(path):
    """Load parameters produced by 38_calibrate_from_runs.py."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    mu = np.array(d["mu"], dtype=np.float32)
    sigma = np.array(d["sigma"], dtype=np.float32)
    T = np.array(d["transition"], dtype=np.float32)
    topological = {int(k): np.array(v, dtype=np.float32)
                   for k, v in d.get("topological", {}).items()} or None

    print("[params] loaded from %s" % path)
    print("[params] fitted on %d runs (%s), %d labels"
          % (d.get("num_runs", 0), d.get("source", "?"), d.get("num_labels", 0)))
    if d.get("source") != SOURCE_LIVE:
        print("[params] WARNING: these were fitted on %s data, not live telemetry."
              % d.get("source"))
    return mu, sigma, T, topological


def learned_parameters(G):
    """
    Fall back to the Goal 5 parameters when no calibrated file is supplied.

    Goal 5 fits on its own simulated sequences, so these are priors from a
    simulator rather than from cluster telemetry. Prefer --params with output
    from 38_calibrate_from_runs.py.
    """
    spec = importlib.util.spec_from_file_location(
        "goal5", os.path.join(os.path.dirname(__file__), "31_run_goal5_experiment.py")
    )
    goal5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(goal5)

    np.random.seed(42)
    random.seed(42)

    state_seqs, anom_scores = goal5.generate_synthetic_data(G, num_ticks=2000)
    all_states = np.concatenate([np.array(v) for v in state_seqs.values()])
    all_scores = np.concatenate([np.array(v) for v in anom_scores.values()])

    learner = DBNParameterLearner()
    mu, sigma = learner.calibrate_emissions(all_states, all_scores)
    T = learner.calibrate_transitions(state_seqs)
    topological = learner.calibrate_topological_influences(state_seqs, G)
    return mu, sigma, T, topological


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay(run, G, params):
    """
    Feed one run's recorded anomaly signals through a fresh DDN.

    Returns (t_detect, predicted_rc, fp_occurred). Detection is the first tick at
    or after the fault where any service's Critical posterior crosses the alarm
    threshold; on a healthy run any crossing at all is a false positive.
    """
    mu, sigma, T, topological = params
    ddn = DynamicDecisionNetworkPhase3(
        service_graph=G,
        num_particles=NUM_PARTICLES,
        learned_T=T,
        learned_mu=mu,
        learned_sigma=sigma,
        learned_topological=topological,
    )

    t_detect = None
    predicted_rc = "None"
    fp_occurred = False

    for rec in run.ticks:
        out = ddn.step(rec.anomaly_signals)
        alarm = any(p["Critical"] > ALARM_THRESHOLD for p in out["posteriors"].values())
        if not alarm:
            continue
        if run.is_positive:
            if rec.tick >= run.t_fault and t_detect is None:
                t_detect = rec.tick
                predicted_rc = out["root_cause"]
        else:
            fp_occurred = True

    return t_detect, predicted_rc, fp_occurred


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def recorded_runs(directory):
    """Load recorded runs, refusing to mix live and synthetic in one result."""
    runs = load_runs(directory)
    if not runs:
        raise SystemExit(
            "\nNo recorded runs in '%s'.\n"
            "Record some first:\n"
            "    python scripts/37_record_runs.py --help\n"
            "Or run the smoke test instead:\n"
            "    python scripts/36_evaluate_goal6.py --source synthetic\n" % directory
        )
    sources = set(r.source for r in runs)
    if len(sources) > 1:
        raise SystemExit(
            "\n'%s' mixes %s runs. A reported result must come from one source. "
            "Separate them into different directories.\n" % (directory, sorted(sources))
        )
    return runs, sources.pop()


def synthetic_runs(G, trials):
    """
    Build the original inline generator's trials as RunRecords.

    Kept identical to the previous implementation so historical numbers still
    reproduce, but routed through the same replay path as recorded runs so the
    two modes cannot silently diverge.
    """
    services = list(G.nodes())
    candidates = [s for s in services if G.in_degree(s) > 0] or services[1:]
    upstream = services[0]

    np.random.seed(1337)
    random.seed(1337)

    runs = []
    for i in range(trials):
        is_positive = i >= trials // 2
        t_fault = 15 if is_positive else None
        injected = random.choice(candidates) if is_positive else None

        run = RunRecord(
            run_id="synthetic_%03d" % i,
            source=SOURCE_SYNTHETIC,
            is_positive=is_positive,
            services=services,
            t_fault=t_fault,
            injected_service=injected,
            fault_type="cpu" if is_positive else None,
        )
        for tick in range(30):
            signals = {}
            for s in services:
                if is_positive and tick >= t_fault:
                    if s == injected:
                        signals[s] = float(np.random.normal(5.0, 0.5))
                    elif s == upstream:
                        signals[s] = float(np.random.normal(1.5, 0.5))
                    else:
                        signals[s] = float(max(0.0, np.random.normal(0.1, 0.1)))
                else:
                    signals[s] = float(max(0.0, np.random.normal(0.1, 0.1)))
            run.add_tick(tick, signals)
        runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def banner(source):
    line = "=" * 72
    if source == SOURCE_LIVE:
        print(
            "\n%s\nSYSTEM EVALUATION - signals recorded from Prometheus via the\n"
            "Rectifier and autoencoder.\n%s" % (line, line)
        )
    else:
        print(
            "\n%s\n!! SMOKE TEST, NOT A SYSTEM EVALUATION !!\n\n"
            "Signals are drawn from np.random.normal. Healthy is 0.1 +/- 0.1 and\n"
            "faulty is 5.0 +/- 0.5, about ten standard deviations apart, so any\n"
            "threshold in between scores 100%%. These numbers show the filter and\n"
            "the metric code run; they say nothing about real failure prediction.\n"
            "Do not report them as results.\n%s" % (line, line)
        )


def report_earliness(timing):
    """
    Report the metric the project exists to demonstrate: how much warning the
    model gives before users are affected.

    Earliness is only defined for runs where the model fired *and* the run
    actually reached a user-visible disruption. Runs missing either are counted
    and excluded rather than silently treated as zero.
    """
    print("\n--- Earliness (lead time before user-visible disruption) ---")

    if not timing:
        print("  no faulty runs")
        return {}

    measurable = [t for t in timing if t.get("earliness_ticks") is not None]
    no_disruption = sum(1 for t in timing if t.get("earliness_ticks") is None)

    if not measurable:
        print("  NOT MEASURABLE for any of the %d faulty runs." % len(timing))
        print("  Either no disruption was reached, or the runs carry no workload")
        print("  data. Record with Istio telemetry available so p95 latency and")
        print("  error rate are captured.")
        return {"earliness_measurable": 0, "earliness_runs": len(timing)}

    ticks = np.array([t["earliness_ticks"] for t in measurable], dtype=float)
    secs = np.array([t["earliness_seconds"] for t in measurable], dtype=float)
    pct = np.array([t["earliness_percentage"] for t in measurable
                    if t.get("earliness_percentage") is not None], dtype=float)
    react = np.array([t["reaction_ticks"] for t in measurable
                      if t.get("reaction_ticks") is not None], dtype=float)

    positive = int((ticks > 0).sum())

    print("  measurable in %d of %d faulty runs%s"
          % (len(measurable), len(timing),
             "" if not no_disruption else " (%d never disrupted)" % no_disruption))
    print("  positive lead time   %d/%d runs (%.0f%%)"
          % (positive, len(measurable), 100.0 * positive / len(measurable)))
    print("  earliness  median    %+.1f ticks  (%+.1f min)"
          % (np.median(ticks), np.median(secs) / 60.0))
    print("             mean      %+.1f ticks  (%+.1f min)"
          % (ticks.mean(), secs.mean() / 60.0))
    print("             range     %+.0f to %+.0f ticks" % (ticks.min(), ticks.max()))
    if len(react):
        print("  reaction   median    %+.1f ticks" % np.median(react))
    if len(pct):
        print("  earliness as %% of error interval: median %.0f%%" % np.median(pct))

    if positive < len(measurable):
        print("  NOTE: %d run(s) had NEGATIVE lead time - the model fired only after"
              % (len(measurable) - positive))
        print("        users were already affected.")

    return {
        "earliness_runs": len(timing),
        "earliness_measurable": len(measurable),
        "earliness_positive_frac": float(positive) / len(measurable),
        "earliness_median_ticks": float(np.median(ticks)),
        "earliness_median_minutes": float(np.median(secs) / 60.0),
        "earliness_mean_minutes": float(secs.mean() / 60.0),
        "reaction_median_ticks": float(np.median(react)) if len(react) else None,
    }


def report(metrics, breakdown, source, runs, timing=None):
    print("\nProvenance: %s" % summarize(runs))
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print("%s: %.4f" % (label, v))
        else:
            print("%s: %s" % (label, v))

    print("\n--- Root cause breakdown ---")
    print("Detected faults (RCA denominator): %d" % len(breakdown))
    pairs = {}
    for item in breakdown:
        key = (item["actual"], item["predicted"])
        pairs[key] = pairs.get(key, 0) + 1
    for (actual, predicted), count in sorted(pairs.items()):
        mark = "OK " if actual == predicted else "   "
        print("  %s %s -> %s  (x%d)" % (mark, actual, predicted, count))

    n_services = len(set(item["actual"] for item in breakdown))
    if n_services > 1:
        print("\nChance baseline for %d candidate services: %.2f%%"
              % (n_services, 100.0 / n_services))

    earl = report_earliness(timing or [])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = dict(metrics)
    out.update(earl)
    out["source"] = source
    out["num_runs"] = len(runs)
    suffix = "recorded" if source == SOURCE_LIVE else "smoketest"
    path = os.path.join(RESULTS_DIR, "evaluation_summary_%s.csv" % suffix)
    pd.DataFrame([out]).to_csv(path, index=False)
    print("\nSaved to %s" % path)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Goal 6 evaluation")
    ap.add_argument("--source", choices=["recorded", "synthetic"], default="recorded")
    ap.add_argument("--dir", default=DEFAULT_RUN_DIR, help="recorded run directory")
    ap.add_argument("--trials", type=int, default=100, help="synthetic trial count")
    ap.add_argument("--params", default=None,
                    help="parameters JSON from 38_calibrate_from_runs.py; "
                         "defaults to the Goal 5 simulated priors")
    args = ap.parse_args()

    if args.source == "recorded":
        runs, source = recorded_runs(args.dir)
        services = sorted(set(s for r in runs for s in r.services))
        G = graph_for_services(services)
    else:
        source = SOURCE_SYNTHETIC
        G = mock_graph()
        runs = synthetic_runs(G, args.trials)

    banner(source)

    print("\nLoading DBN parameters...")
    params = learned_parameters(G)
    print("  mu    = %s" % np.round(params[0], 3))
    print("  sigma = %s" % np.round(params[1], 3))

    np.random.seed(1337)
    random.seed(1337)

    evaluator = Goal6Evaluator()
    breakdown = []
    timing = []

    print("\nReplaying %d runs..." % len(runs))
    for run in runs:
        t_detect, predicted_rc, fp = replay(run, G, params)
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
            breakdown.append({"actual": run.injected_service, "predicted": predicted_rc})
        if run.is_positive:
            timing.append(dict(
                earliness(t_detect, run.t_disruption, run.t_fault, run.interval_seconds),
                run_id=run.run_id,
                localized=(predicted_rc == run.injected_service),
            ))

    report(evaluator.compute_metrics(), breakdown, source, runs, timing)
    banner(source)


if __name__ == "__main__":
    main()
