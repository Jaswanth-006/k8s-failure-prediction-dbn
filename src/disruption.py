"""
Disruption detection: when does a fault become visible to users?

Why this exists
---------------
Earliness is the whole point of the project - "we predict the failure N minutes
before users are affected". Measuring it needs two timestamps: when the model
raised the alarm, and when the failure actually became user-visible. The second
one was never captured, so `earliness_interval` has been null in every result.

This module supplies it, using the criterion from Denaro et al. (FSE 2024) §4.1.2
so our numbers are directly comparable to the paper's:

    A disruptive failure occurs at the first tick where the user-facing metrics
    (request response time and HTTP failure rate) differ from the healthy
    baseline to a degree that is both statistically significant (Mann-Whitney U)
    and practically large (Vargha-Delaney A12 effect size).

Using both tests matters. With enough samples a trivial shift becomes
"significant", so the effect size is what keeps the detected disruption
meaningful rather than merely detectable.

Definitions
-----------
    error interval     fault injection -> disruptive failure
    reaction interval  fault injection -> first correct prediction
    earliness interval first correct prediction -> disruptive failure

Earliness is positive when the model predicts before users are hurt. That is the
number the project lives or dies by.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import mannwhitneyu

# Standard Vargha-Delaney thresholds. 0.71 is "large"; below it the shift is
# real but too small to call a user-visible disruption.
A12_NEGLIGIBLE = 0.56
A12_SMALL = 0.56
A12_MEDIUM = 0.64
A12_LARGE = 0.71

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_EFFECT = A12_LARGE

# User-facing signals. Latency and errors are what a user actually perceives;
# request rate is collected for context but is not a disruption signal on its
# own, since load legitimately varies.
DISRUPTION_SIGNALS = ("p95_latency_ms", "error_rate")


def vargha_delaney_a12(treatment: Sequence[float], control: Sequence[float]) -> float:
    """
    Vargha-Delaney A12: P(treatment > control) + 0.5 * P(treatment == control).

    0.5 means the two samples are indistinguishable; 1.0 means every treatment
    value exceeds every control value. Computed from ranks so it is robust to
    the heavy-tailed latency distributions typical of microservice telemetry.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    n_t, n_c = len(t), len(c)
    if n_t == 0 or n_c == 0:
        return 0.5

    combined = np.concatenate([t, c])
    # average ranks so ties are handled correctly
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    _assign_tied_ranks(combined, ranks)

    rank_sum_t = ranks[:n_t].sum()
    return (rank_sum_t / n_t - (n_t + 1) / 2.0) / n_c


def _assign_tied_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    """Replace ranks of tied values with their average, in place."""
    order = values.argsort(kind="mergesort")
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1


@dataclass
class SignalVerdict:
    """Outcome of comparing one signal's window against the baseline."""
    signal: str
    p_value: float
    a12: float
    baseline_median: float
    window_median: float
    disrupted: bool


@dataclass
class DisruptionResult:
    """Where (and whether) a run became user-visibly disrupted."""
    t_disruption: Optional[int]
    verdicts: List[SignalVerdict]
    reason: str

    @property
    def disrupted(self) -> bool:
        return self.t_disruption is not None


def compare_window(
    window: Sequence[float],
    baseline: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = DEFAULT_MIN_EFFECT,
    signal: str = "",
) -> SignalVerdict:
    """
    Test one window of a signal against the healthy baseline.

    Disruption requires the window to be *worse*, not merely different, so the
    Mann-Whitney test is one-sided (greater). Both latency and error rate are
    signals where higher means worse.
    """
    w = np.asarray(window, dtype=float)
    b = np.asarray(baseline, dtype=float)

    if len(w) < 2 or len(b) < 2:
        return SignalVerdict(signal, 1.0, 0.5, float("nan"), float("nan"), False)

    # Both samples flat at the same level - common when error_rate is 0.0 across
    # a healthy run - makes the rank test undefined. Only bail out when the
    # window is not worse; a flat window sitting above a flat baseline is a real
    # disruption and must still go through the test below.
    w_flat = float(np.ptp(w)) == 0.0
    b_flat = float(np.ptp(b)) == 0.0
    if w_flat and b_flat and w[0] <= b[0]:
        return SignalVerdict(signal, 1.0, 0.5, float(np.median(b)), float(np.median(w)), False)

    try:
        _, p = mannwhitneyu(w, b, alternative="greater")
    except ValueError:
        p = 1.0

    a12 = vargha_delaney_a12(w, b)
    return SignalVerdict(
        signal=signal,
        p_value=float(p),
        a12=float(a12),
        baseline_median=float(np.median(b)),
        window_median=float(np.median(w)),
        disrupted=bool(p < alpha and a12 >= min_effect),
    )


def detect_disruption(
    workload: List[Dict[str, float]],
    baseline_ticks: int,
    window: int = 5,
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = DEFAULT_MIN_EFFECT,
    signals: Sequence[str] = DISRUPTION_SIGNALS,
    persistence: int = 3,
    correct_multiple_testing: bool = True,
) -> DisruptionResult:
    """
    Find the first tick at which the run is user-visibly disrupted.

    Parameters
    ----------
    workload
        Per-tick dicts of user-facing metrics, one entry per tick of the run.
    baseline_ticks
        How many leading ticks are known-healthy. These form the control sample,
        so this should be the pre-fault portion of the run.
    window
        Trailing window compared against the baseline at each candidate tick. A
        window smooths over single-tick spikes, which is what keeps a transient
        blip from being called a disruption.

    persistence
        How many consecutive ticks must meet the criterion before a disruption
        is declared. The reported timestamp is the *first* tick of that streak.

        This is not optional decoration. With a short window against a short
        baseline, a one-sided rank test on pure noise clears p < 0.05 with a
        large A12 often enough to matter, and the criterion is re-tested at
        every tick, so chance flags accumulate across a run. Empirically a
        single-tick rule fires on healthy runs; requiring 3 consecutive ticks
        removes that while barely delaying real disruptions, which persist by
        definition.
    correct_multiple_testing
        Divide alpha by the number of candidate ticks (Bonferroni). The
        criterion is evaluated once per post-baseline tick, so an uncorrected
        alpha of 0.05 means roughly one false flag every twenty ticks by
        construction. On a 40-tick run that alone produced a 38% false-positive
        rate on healthy data in testing; with correction and persistence
        together it drops to near zero while real disruptions are still caught
        within a few ticks of onset.

    A run is disrupted at the first tick where *any* user-facing signal is both
    significantly and substantially worse. Latency and errors are alternative
    symptoms of the same disruption, not conditions that must co-occur.
    """
    n = len(workload)
    if n == 0:
        return DisruptionResult(None, [], "no workload data recorded")
    if baseline_ticks < 2:
        return DisruptionResult(None, [], "baseline too short (need >= 2 ticks)")
    if baseline_ticks >= n:
        return DisruptionResult(None, [], "no post-baseline ticks to test")

    available = [s for s in signals if any(s in w for w in workload)]
    if not available:
        return DisruptionResult(
            None, [],
            "workload data has none of %s" % ", ".join(signals),
        )

    series = {
        s: np.array([float(w.get(s, 0.0)) for w in workload], dtype=float)
        for s in available
    }

    n_tests = max(1, n - baseline_ticks)
    effective_alpha = alpha / n_tests if correct_multiple_testing else alpha

    best: List[SignalVerdict] = []
    streak_start: Optional[int] = None
    streak_verdicts: List[SignalVerdict] = []

    for t in range(baseline_ticks, n):
        start = max(baseline_ticks, t - window + 1)
        verdicts = []
        for s in available:
            baseline = series[s][:baseline_ticks]
            win = series[s][start:t + 1]
            verdicts.append(compare_window(win, baseline, effective_alpha, min_effect, s))

        if any(v.disrupted for v in verdicts):
            if streak_start is None:
                streak_start = t
                streak_verdicts = verdicts
            if t - streak_start + 1 >= persistence:
                names = ", ".join(v.signal for v in streak_verdicts if v.disrupted)
                return DisruptionResult(
                    streak_start, streak_verdicts,
                    "disrupted on %s, sustained %d ticks" % (names, persistence),
                )
        else:
            # Streak broken - a real disruption does not recover on its own.
            streak_start = None
        best = verdicts

    if streak_start is not None:
        return DisruptionResult(
            None, best,
            "criterion met at tick %d but not sustained for %d ticks"
            % (streak_start, persistence),
        )
    return DisruptionResult(None, best, "no signal reached significance with a large effect")


def earliness(
    t_detect: Optional[int],
    t_disruption: Optional[int],
    t_fault: Optional[int],
    interval_seconds: float = 60.0,
) -> Dict[str, Optional[float]]:
    """
    Compute the paper's timing metrics for one run.

    Returns reaction and earliness in both ticks and seconds, plus earliness as a
    percentage position inside the error interval (fault -> disruption), which is
    how the paper's Table 4 reports it. A negative earliness means the model
    predicted only after users were already affected.
    """
    out: Dict[str, Optional[float]] = {
        "reaction_ticks": None,
        "reaction_seconds": None,
        "earliness_ticks": None,
        "earliness_seconds": None,
        "earliness_percentage": None,
        "error_interval_ticks": None,
    }

    if t_fault is not None and t_detect is not None:
        out["reaction_ticks"] = t_detect - t_fault
        out["reaction_seconds"] = (t_detect - t_fault) * interval_seconds

    if t_disruption is None or t_detect is None:
        return out

    out["earliness_ticks"] = t_disruption - t_detect
    out["earliness_seconds"] = (t_disruption - t_detect) * interval_seconds

    if t_fault is not None:
        error_interval = t_disruption - t_fault
        out["error_interval_ticks"] = error_interval
        if error_interval > 0:
            # Position of the prediction inside the error interval. The paper
            # reports high percentages as good: predicting at 10% of the way in
            # leaves 90% of the window to act.
            out["earliness_percentage"] = 100.0 * (t_disruption - t_detect) / error_interval

    return out
