"""
Weak supervision: turn recorded-run ground truth into DBN state labels.

Why weak supervision
--------------------
The DBN's hidden health state (Normal / Degrading / Critical) is by definition
never observed. Fitting its emission and transition tables needs labelled
sequences, and nobody hand-labels telemetry per service per minute.

The fault-injection schedule supplies them for free. We know exactly which
service was injected and when, and (from src.disruption) when the failure became
user-visible. That pins down three regions per faulty run:

    tick < t_fault                  Normal      - nothing wrong yet
    t_fault <= tick < t_disruption  the error interval, where the fault is
                                    corrupting state but users are not yet hurt
    tick >= t_disruption            Critical    - users are affected

The error interval is where Degrading lives, which is precisely the state the
original PREFACE threshold could not express. We split it: the earlier part is
Degrading, the later part Critical, because degradation worsens toward
disruption rather than flipping at one instant.

What is NOT labelled
--------------------
Only the injected service is labelled unhealthy. Downstream services are left
Normal by default even though some are probably suffering, because "the fault
propagated to this neighbour" is an assumption, not ground truth, and baking it
into the labels would teach the model the very propagation behaviour we then
claim to have discovered. `propagate_downstream=True` opts into that assumption
explicitly for anyone who wants it; the honest default is off, which makes the
learned topological influence conservative rather than self-confirming.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

NORMAL, DEGRADING, CRITICAL = 0, 1, 2
STATE_NAMES = ("Normal", "Degrading", "Critical")

# Fraction of the error interval spent in Degrading before switching to
# Critical. 0.5 splits it evenly; the exact value matters less than having a
# graded transition rather than an instantaneous flip.
DEGRADING_FRACTION = 0.5

# When a faulty run never reached disruption, we cannot know how far it got.
# Label this many ticks after injection as Degrading and stop there, rather than
# inventing a Critical phase that was never observed.
UNRESOLVED_DEGRADING_TICKS = 5


def label_run(
    run,
    graph: Optional[nx.DiGraph] = None,
    propagate_downstream: bool = False,
) -> Dict[str, List[int]]:
    """
    Produce a per-service state sequence for one recorded run.

    Returns {service: [state per tick]}, aligned with run.ticks.
    """
    n = len(run.ticks)
    labels = {s: [NORMAL] * n for s in run.services}

    if not run.is_positive or run.t_fault is None:
        return labels

    target = run.injected_service
    if target not in labels:
        return labels

    t_fault = run.t_fault
    t_disruption = run.t_disruption

    if t_disruption is not None and t_disruption > t_fault:
        error_interval = t_disruption - t_fault
        switch = t_fault + max(1, int(round(error_interval * DEGRADING_FRACTION)))
        for t in range(n):
            if t < t_fault:
                continue
            elif t < switch:
                labels[target][t] = DEGRADING
            else:
                labels[target][t] = CRITICAL
    else:
        # No observed disruption: mark a bounded Degrading window only.
        for t in range(t_fault, min(n, t_fault + UNRESOLVED_DEGRADING_TICKS)):
            labels[target][t] = DEGRADING

    if propagate_downstream and graph is not None and target in graph:
        # Explicit opt-in to the propagation assumption. Descendants are marked
        # one level less severe than the origin, never Critical, so they cannot
        # out-rank the true root cause during calibration.
        for descendant in nx.descendants(graph, target):
            if descendant not in labels:
                continue
            for t in range(n):
                if labels[target][t] != NORMAL:
                    labels[descendant][t] = DEGRADING

    return labels


def build_training_set(
    runs: Sequence,
    graph: Optional[nx.DiGraph] = None,
    propagate_downstream: bool = False,
):
    """
    Flatten a set of recorded runs into the arrays the DBN learner expects.

    Returns
    -------
    states : np.ndarray
        Flat array of state labels, one entry per (service, tick).
    scores : np.ndarray
        Matching anomaly signals, same order. Used to fit the emission model.
    sequences : dict
        {key: [state per tick]} preserving per-run, per-service ordering, which
        the transition and topological calibrators need intact - flattening
        would create bogus transitions across run boundaries.
    """
    states: List[int] = []
    scores: List[float] = []
    sequences: Dict[str, List[int]] = {}

    for run in runs:
        labels = label_run(run, graph=graph, propagate_downstream=propagate_downstream)
        for service, seq in labels.items():
            # Key per run so the transition calibrator never joins the end of
            # one run to the start of the next.
            sequences["%s::%s" % (run.run_id, service)] = seq
            for t, tick in enumerate(run.ticks):
                if t >= len(seq):
                    break
                states.append(seq[t])
                scores.append(float(tick.anomaly_signals.get(service, 0.0)))

    return np.array(states, dtype=int), np.array(scores, dtype=float), sequences


def label_summary(states: np.ndarray) -> str:
    """Human-readable class balance, for printing before calibration."""
    if len(states) == 0:
        return "no labels"
    counts = [int((states == k).sum()) for k in (NORMAL, DEGRADING, CRITICAL)]
    total = len(states)
    parts = [
        "%s %d (%.1f%%)" % (STATE_NAMES[k], counts[k], 100.0 * counts[k] / total)
        for k in range(3)
    ]
    return "%d labels: " % total + ", ".join(parts)


def check_balance(states: np.ndarray, min_per_state: int = 30) -> List[str]:
    """
    Report states with too few examples to fit an emission distribution.

    A Gaussian fitted to a handful of points is not calibration, it is noise, so
    the caller should surface this rather than quietly producing parameters.
    """
    warnings = []
    for k in (NORMAL, DEGRADING, CRITICAL):
        count = int((states == k).sum())
        if count < min_per_state:
            warnings.append(
                "only %d %s labels (want >= %d); its emission parameters will be "
                "unreliable" % (count, STATE_NAMES[k], min_per_state)
            )
    return warnings
