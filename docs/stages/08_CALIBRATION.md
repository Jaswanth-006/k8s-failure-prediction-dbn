# Stage 08: Calibration (Where the DBN's Numbers Come From)

> The DBN needs numbers: what signal each state produces (μ, σ), how states change (transition matrix), and how a sick parent affects a child. We **learn them from the recorded runs**, using the fault schedule as labels. This is called *weak supervision*.

---

## 1. In simple words

To learn "what a Critical service's signal looks like", we need examples labelled *this minute was Critical*. Nobody labels telemetry by hand. But **we caused the faults ourselves**, so we know:

- *before* we injected the fault → the service was **Normal**;
- *after* users were measurably affected (Stage 06) → it was **Critical**;
- *in between* → it was getting worse: **Degrading**.

These labels are "weak" because they come from the schedule, not from inspecting each minute. Once every minute has a label, calibration is just counting and averaging:

- **average signal** of all Critical minutes → μ_Critical;
- **how often** Degrading was followed by Critical → T[Degrading → Critical].

## 2. Why this stage exists

- The original parameters were **guesses** (μ = 0, 2.5, 5.0).
- "Goal 5" fitted parameters with EM on data **generated from a hand-written ground truth** (`true_mu = [0.1, 3.0, 5.5]`) and recovered the generator's own values. That's a valid test of the estimator, but not calibration, because no telemetry was involved.
- Parameters fitted to **our** cluster's signals are what let the DBN treat healthy noise (up to about 6) as Normal. That's the reason it raised no false alarms.

---

## 3. Step by step

```
Step 1  Load recorded runs             data/experiments/runs_v2/*.json (3 healthy + 3 faulty, 28 ticks each)
Step 2  Label every (run, service, tick) with a state   src/weak_labels.py::label_run
Step 3  Flatten: states[], scores[] and per-run sequences   build_training_set
Step 4  Check class balance            warn if any state has < 30 labels
Step 5  Fit emissions                  mean and std of signals per state
Step 6  Fit transitions                count state→state pairs (+1 smoothing), normalise rows
Step 7  Fit topological modifiers      compare transitions by worst-parent state
Step 8  Save JSON                      data/experiments/params/live_v2.json
Step 9  Used by                        evaluator (--params) and operator (PREFACE_PARAMS, default live_v2.json)
```

Command: `python scripts/38_calibrate_from_runs.py --dir data/experiments/runs_v2 --out data/experiments/params/live_v2.json`
(`scripts/41_analyze_rerun.py` runs it for you.)

---

## 4. Technical depth

### 4.1 The labelling rule (`label_run`)

For a faulty run with injection tick t_f and disruption tick t_d, **only the injected service** gets non-Normal labels:

```
error_interval = t_d − t_f
switch = t_f + max(1, round(error_interval × 0.5))        # DEGRADING_FRACTION = 0.5

tick <  t_f            → Normal
t_f ≤ tick < switch    → Degrading
tick ≥ switch          → Critical
```

- If the fault never caused a disruption, label 5 ticks after injection as Degrading and stop. Never invent a Critical phase that wasn't observed.
- Healthy runs: every service, every tick → Normal.
- **Downstream services stay Normal**, even though they may suffer. Labelling them sick would *assume* the propagation the model is supposed to discover. (`--propagate-downstream` opts in; it's off by default.)

### 4.2 The v2 labels, derived

Every fault was injected at tick 8 and runs have 28 ticks (0–27). Python's `round` rounds halves to even, so round(2.5) = 2 and round(5.5) = 6.

| Fault | t_d | interval | switch | Degrading ticks | Critical ticks |
|---|---|---|---|---|---|
| train | 13 | 5 | 8 + 2 = 10 | 8, 9 → **2** | 10–27 → **18** |
| route | 13 | 5 | 8 + 2 = 10 | 8, 9 → **2** | 10–27 → **18** |
| order | 19 | 11 | 8 + 6 = 14 | 8–13 → **6** | 14–27 → **14** |

- **Total labels:** 6 runs × 8 services × 28 ticks = **1,344**.
- **Degrading:** 2 + 2 + 6 = **10**. **Critical:** 18 + 18 + 14 = **50**. **Normal:** 1,344 − 60 = **1,284**.

These match the calibration output exactly (1284 / 10 / 50).

### 4.3 Emission fit (`calibrate_emissions`)

For each state k, take every signal labelled k:

```
μ_k = mean(signals with label k)
σ_k = sqrt( max( sample variance (ddof=1), 0.01 ) )      # floor avoids zero spread
if fewer than 5 samples: fall back to defaults (0/2.5/5.0, 1.0/1.2/1.5)
```

This is the **maximum-likelihood estimate** of a Gaussian (with the unbiased variance).

| | v1 (Sep 11) | **v2 (Sep 15, used)** |
|---|---|---|
| Faulty runs with a disruption | 1/3 | 3/3 |
| Labels N / D / C | 844 / 12 / 8 | **1284 / 10 / 50** |
| μ (N, D, C) | 0.78, 7.71, 10.40 | **0.66, 7.00, 10.54** |
| σ (N, D, C) | 1.61, 4.73, **0.18** | **0.80, 4.84, 0.80** |

**Why is σ_Degrading so wide (4.84)?** Degrading labels start at the injection tick. But at tick 8 the 2-minute `rate()` window hasn't caught the stress yet, so the signal is about 0. At tick 9 it jumps to about 10. The 10 Degrading signals, sorted:

```
0.00  0.00  0.00  9.73  9.79  9.79  9.79  9.79  10.48  10.66
└── the three injection ticks ──┘└──────── fault visible ────────┘
```

Their mean is 7.00, but no Degrading tick actually had a signal near 7. It's a two-humped set, which one Gaussian can only describe with a huge σ.

**Why v1 was unstable:** with only 8 Critical labels, σ_Critical = 0.18. A Critical tick at 9.9 instead of 10.4 was then judged very unlikely, so the DBN's recall varied from 33% to 67% across seeds. v2's 50 labels gave σ = 0.80.

### 4.4 Transition fit (`calibrate_transitions`)

Count consecutive label pairs within each run and service, starting every cell at **1** (Laplace smoothing, so no probability is exactly 0), then divide each row by its sum.

Each run × service sequence has 27 transitions, and there are 48 sequences, so **1,296 transitions** in total:

| From \ To | Normal | Degrading | Critical | Row sum | Row probabilities |
|---|---|---|---|---|---|
| **Normal** | 1236 + 1 | 3 + 1 | 0 + 1 | 1242 | 0.9960, 0.0032, 0.0008 |
| **Degrading** | 0 + 1 | 7 + 1 | 3 + 1 | 13 | 0.0769, 0.6154, 0.3077 |
| **Critical** | 0 + 1 | 0 + 1 | 47 + 1 | 50 | 0.0200, 0.0200, 0.9600 |

Where the counts come from:
- **N→D = 3:** one per fault, at tick 7→8.
- **D→D = 7:** 1 + 1 + 5. **D→C = 3:** one per fault.
- **C→C = 47:** 17 + 17 + 13. Critical runs to the end, so there is never C→N.

Keys are `run_id::service`, so the last tick of one run is **never** joined to the first tick of the next.

### 4.5 Topological modifiers (`calibrate_topological_influences`)

The intent:
1. For each service and tick, find the worst parent state (0, 1 or 2).
2. Count transitions separately for worst parent = 0, 1, 2, again with +1 smoothing.
3. Modifier for worst parent w = the average difference `P(next | parent = w) − P(next | parent = 0)`, re-centred to sum to 0.

Stored as `"topological": {"1": [...], "2": [...]}` and added to transition rows in the DBN.

---

## 5. Real example: route fault labels against real signals

```
tick:     5     6     7  |  8     9  |  10     11     12     13  ...  27
label:    N     N     N  |  D     D  |  C      C      C      C   ...  C
signal:  0.00  0.00  0.00| 0.00 10.66| 11.55  11.24  11.11  11.54 ...
                           ^ labelled Degrading, but the signal hasn't moved yet (rate-window lag)
```

The other 7 services in this run are labelled Normal on every tick, including the dashboard's 3.16 at tick 9. That's part of why μ_Normal is 0.66 rather than 0: the Normal state learned to expect some noise.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Weak labelling | `src/weak_labels.py` |
| Estimators | `src/dbn_learner.py` (`DBNParameterLearner`) |
| Calibration script | `scripts/38_calibrate_from_runs.py` |
| Output | `data/experiments/params/live_v2.json` (v1: `live_v1.json`) |
| Loading | `scripts/36_evaluate_goal6.py::load_params_file`, `src/inference_adapter.py::load_params` |

## 7. Limits and known issues

- **⚠ Bug: the topological modifiers are silently zero.** `build_training_set` stores sequences under keys like `"live_fault_003::ts-route-service"`, but `calibrate_topological_influences` looks them up by plain service name (`"ts-route-service"`). **None of the 8 graph nodes is found**, every service is skipped, all counts stay at the smoothing value 1, and every modifier comes out `[0, 0, 0]`. Because a (zero) modifier dict exists, the DBN also skips its hand-written default parent influence. **Net effect: in the calibrated DBN, a sick parent does not change a child's transition probabilities.** The fix is to group sequences by run before looking up parents. Note too that, even fixed, the default labels keep victims Normal, so the learned influence would be small.
- **Calibrated and evaluated on the same recording.** `scripts/41_analyze_rerun.py` fits parameters on `runs_v2` and then scores `runs_v2`. There is no held-out test set, so the v2 results are **in-sample**. This is the most important caveat for the results.
- **Degrading has only 10 labels** (the calibrator asks for 30). Its μ and σ are unreliable.
- **The labels are schedule-based.** The rate-window lag makes tick 8 "Degrading" while the signal is still 0.
- **The same μ and σ for every service.** A single emission model is shared by all 8 services, even though their noise levels differ (Stage 05, error levels 0.036–0.141).

## 8. Questions a reviewer may ask

**Q: How did you get labels for hidden states that can't be observed?**
A: From the fault-injection schedule and the measured disruption time: Normal before injection, Degrading for the first half of the fault-to-disruption interval, Critical afterwards. Only the injected service is labelled unhealthy.

**Q: How are the emission parameters estimated?**
A: Maximum likelihood for a Gaussian: the mean and the sample standard deviation of the signals under each label, with a small variance floor.

**Q: What is Laplace smoothing and why use it?**
A: Adding 1 to every transition count, so an unseen transition (e.g. Critical → Normal) gets a small non-zero probability instead of being impossible forever.

**Q: Did you evaluate on data you trained on?**
A: Yes. The v2 DBN parameters were calibrated on the same six runs it was evaluated on. The autoencoder was trained on separate healthy data, but the DBN comparison is in-sample. A held-out recording is the most important next step.

**Q: Does the graph influence get learned?**
A: The code intends to learn it, but a key-mismatch bug made the learned modifiers zero. We found this while preparing these notes, and it's documented, not hidden.

**Q: Why did v1 perform worse?**
A: Only one of its three faults reached a disruption, leaving 8 Critical labels and σ_Critical = 0.18, far too narrow. v2 fixed the recording method, so all three faults produced disruptions.
