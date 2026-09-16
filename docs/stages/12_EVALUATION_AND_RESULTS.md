# Stage 12: Evaluation and Results (Does It Work, and How Do We Know?)

> We replay the **same recorded signals** through two decision layers, PREFACE's memoryless threshold and PREFACE-DBN, so only the reasoning differs. The DBN is run with 8 random seeds. We also test the live operator end to end. This page gives the method, every headline number, and the caveats you must state.

---

## 1. In simple words

Two referees watch the same recorded matches:

- **Referee A (PREFACE)** blows the whistle whenever any single reading crosses a fixed line.
- **Referee B (PREFACE-DBN)** keeps a running belief and blows the whistle when a service is probably Critical.

We count how often each one:
- **catches** the real fouls (faults detected),
- **blows the whistle when nothing happened** (false alarms on healthy runs),
- **points at the right player** (root cause),
- **blows the whistle before the crowd reacts** (warning time).

Referee B uses a little randomness, so we replay its matches 8 times with different seeds and report the average and spread.

## 2. Why this stage exists

- To test the project's claim: *does adding a DBN improve on PREFACE?*
- To make that test **fair**: both methods get identical inputs (the Rectifier and autoencoder are held fixed), so any difference comes from the decision layer.
- To keep the project **honest**. Earlier evaluations produced impressive numbers that turned out to be invalid (4.6).

---

## 3. Step by step

```
Step 1  Recordings (Stage 06)       runs_v2: 3 healthy + 3 CPU-fault runs, 28 ticks each (v1 also kept)
Step 2  Calibrate DBN (Stage 08)    on the same recording
Step 3  PREFACE-DBN replay          for seed in 0..7: fresh DBN per run, step through all ticks
Step 4  PREFACE replay              deterministic, threshold log1p(3) = 1.386
Step 5  Score both                  recall, false-positive rate, root-cause accuracy, latency, warning time
Step 6  Report                      DBN as mean ± std over seeds; PREFACE as a single value
Step 7  Threshold sweep             PREFACE at z > 3, 5, 10, 19
Step 8  Operator test (Stage 11)    live cluster, shadow mode, one fault
```

Command (steps 2–6): `python scripts/41_analyze_rerun.py --run-dir data/experiments/runs_v2 --baseline-dir data/experiments/runs --seeds 8`

---

## 4. Technical depth

### 4.1 The two reasoners, precisely

**PREFACE** (`scripts/39_compare_baseline.py::replay_preface`), memoryless:
```
for each tick:
    top_service, top_signal = max(anomaly_signals)
    alarm if top_signal > log1p(3) = 1.386          # z > 3, PREFACE's m + 3σ rule
    root cause = top_service
```

**PREFACE-DBN** (`scripts/36_evaluate_goal6.py::replay`):
```
dbn = DynamicDecisionNetworkPhase3(graph, 500 particles, calibrated params)
for each tick:
    out = dbn.step(anomaly_signals)
    alarm if any service has P(Critical) > 0.5
    root cause = out["root_cause"]                  # causal analyzer, Stage 09
```

### 4.2 How each run is scored

| Run type | Rule |
|---|---|
| Faulty | **detection tick** = first alarm at or after `t_fault`; **predicted root cause** = root cause at that tick. Alarms *before* the fault are ignored here. |
| Healthy | **false alarm** = an alarm on *any* tick |

| Metric | Formula |
|---|---|
| Recall (faults detected) | detected faulty runs / faulty runs |
| False-positive rate | healthy runs with a false alarm / healthy runs |
| Root-cause accuracy | correctly localised detections / detections |
| Detection latency | mean of (detection tick − t_fault), in ticks |
| Warning time (earliness) | t_disruption − detection tick; we report the median across runs |

**Seeds.** For the DBN, each metric is computed per seed, then reported as mean ± std (with min–max) over 8 seeds. On v1, a single seed once showed 100% recall where 8 seeds gave 33–67%.

### 4.3 Results: second recording (v2, the main result)

| Metric | PREFACE (z > 3) | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected | 3/3 | 3/3 on every seed |
| **Healthy runs with a false alarm** | **3/3** (alarming on 16, 20 and 17 of 28 ticks) | **0/3 on every seed** |
| Root cause correct | 2/3 (blamed the dashboard for the train fault) | 3/3 on every seed |
| Detection latency | 0.67 tick \* | 1.71 ± 0.42 ticks (1.00–2.33) |
| Warning time before disruption (median) | 5.0 min \* | 3.5 ± 0.5 min (3–4) |

\* PREFACE was **already alarming before every fault** (on 3–5 of the 8 pre-fault ticks). Its "fast detection" is just an alarm that's almost always on.

### 4.4 Results: first recording (v1)

| Metric | PREFACE (z > 3) | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected | 3/3 | 42% ± 14% (33–67%) |
| Healthy runs with a false alarm | 3/3 (4, 4 and 11 of 18 ticks) | 0/3 on every seed |
| Root cause correct | 1/3 | 100% of the faults it detected |
| Detection latency | 0.33 tick \* | 1.88 ± 0.22 ticks |

v1's weak DBN recall is explained by its thin calibration (8 Critical labels, σ = 0.18; Stage 08).

### 4.5 Per-run detail (v2)

| Run | Disruption | PREFACE | PREFACE-DBN (8 seeds) |
|---|---|---|---|
| train fault | tick 13 | alarming on 5/8 pre-fault ticks; at the fault tick blamed **ts-ui-dashboard** | detected ticks 9–12 (median 10), **ts-train-service** on every seed |
| route fault | tick 13 | alarming on 3/8 pre-fault ticks; blamed ts-route-service | ticks 9–10 (median 10), ts-route-service on every seed |
| order fault | tick 19 | alarming on 4/8 pre-fault ticks; blamed ts-order-service | ticks 9–10 (median 10), ts-order-service on every seed |
| healthy ×3 | none | false alarms from tick 0 | none on any seed |

### 4.6 Threshold sensitivity

PREFACE-DBN has no threshold on the signal, so its row doesn't change.

| PREFACE threshold (signal / z) | Healthy runs with a false alarm (v1 + v2) | Root cause correct, v2 |
|---|---|---|
| **1.386 / z > 3 (the paper's rule)** | **6 of 6** | 2 of 3 |
| 1.792 / z > 5 | 6 of 6 | 3 of 3 |
| 2.398 / z > 10 | 5 of 6 | 3 of 3 |
| 3.0 / z > 19 (our earlier mistake) | 2 of 6 | 3 of 3 |
| **PREFACE-DBN** | **0 of 6 on every seed** | **3 of 3 on every seed** |

**The correction story (be ready to tell it).** The first version of this comparison used a PREFACE threshold of **3.0**, described as "3σ". But the signal is log1p(z), so 3.0 means z > e³ − 1 ≈ **19σ**, a PREFACE far stricter than the paper's, which rarely false-alarmed. That version concluded "the DBN has no advantage". Using the correct value log1p(3) = 1.386 reverses the conclusion. The fix is in `scripts/39` and `scripts/41`, and the results were reproduced exactly.

### 4.7 Why PREFACE false-alarms here

The autoencoder is **under-dispersed** on live healthy data. Held-out validation peaked at 1.65, but live healthy runs peaked at 2.23–6.47 (Stage 05). Healthy noise routinely crosses 3σ of the training error, and a fixed threshold passes it straight through. The DBN's Normal state was calibrated on the live runs (μ = 0.66, σ = 0.80), and its Critical state sits at 10.5 ± 0.8, so healthy spikes can't build a Critical belief.

### 4.8 The operator, end to end

In shadow mode on a live route-service fault (Stage 11), **all 9 checks passed**:
- root cause named at 0.8 minutes;
- P(Critical) ≥ 0.95 at 3.9 minutes;
- Reschedule_Pod decided at 10.9 minutes, logged as WOULD_EXECUTE;
- the deployment was unchanged;
- P(Critical) was 0 within 2 minutes of the fault's removal;
- ticks took 0.06–0.10 s.

**It acts too late:** the debounce, not the model, set the response time.

### 4.9 Earlier results that were withdrawn

| Earlier claim | Why it was invalid |
|---|---|
| "100% precision / recall / F1, 0% FPR" (Goal 6) | Signals were generated inside the evaluator: healthy 0.1 ± 0.1, faulty 5.0 ± 0.5, about 10σ apart. Any threshold scores 100%. It's now labelled a smoke test. |
| "Parameters learned from telemetry" (Goal 5) | EM was fitted to data sampled from hand-written parameters and recovered them. It's a valid estimator test, but not calibration. |
| "The DBN was tricked into blaming a proxy" (pilot) | One run, plus a localizer bug that always favoured the graph root. |
| "44% root-cause accuracy" (Goal 6) | Simulated signals on a flat star graph; all 28 wrong answers blamed the hub. |
| "PREFACE-DBN has no advantage" | Wrong threshold (19σ instead of 3σ). |

---

## 5. Real example: one fault, both referees

**Train-service fault (v2)**, fault at tick 8, disruption at tick 13:

```
tick   train signal  dashboard  PREFACE                        PREFACE-DBN (seed 0)
  7       0.00         1.76     ALARM (blames dashboard)       quiet
  8       0.00         1.65     ALARM (blames dashboard)  ←    quiet             (fault injected; not visible yet)
                                 counted as detection, wrong root cause
  9       9.73         0.74     ALARM (train)                  P(C) train 0.69 → ALARM, root = train  ←
 10      10.54         0.88     ALARM                          P(C) 0.97
 13                                                            (disruption confirmed)
```

PREFACE's "detection" at tick 8 happened **before the fault was even visible**, from noise, and blamed the wrong service. The DBN alarmed one tick later, when the evidence appeared, and named the right service.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Full analysis over seeds, head-to-head | `scripts/41_analyze_rerun.py` |
| PREFACE replay, single-seed comparison | `scripts/39_compare_baseline.py` |
| DBN replay, graph and parameter loading | `scripts/36_evaluate_goal6.py` |
| Metric computation | `src/goal6_evaluator.py`, `src/disruption.py::earliness` |
| Global-error PREFACE (not replayable on recordings) | `src/preface_baseline.py` |
| Operator test | `scripts/42_operator_fault_test.py` |
| Full numbers | `docs/RESULTS_LIVE.md`, `docs/final_report.md` |

## 7. Limits and caveats (state these yourself before you are asked)

1. **Small n.** 3 faults and 3 healthy runs per recording (6 + 6 in total). This supports "it works end to end and shows a clear pattern", not a statistical claim.
2. **In-sample.** The v2 DBN was calibrated on the same six runs it was evaluated on. There's no held-out recording.
3. **Warning time is measured against a criterion with a floor.** The disruption test can't confirm a sudden fault earlier than 5 ticks after injection (Stage 06, 4.7). For train and route, user p95 had already risen at tick 9, and the DBN alarmed at tick 9–10. So **the DBN detected at about the same time latency first rose, not 3.5 minutes before users could feel anything**. The 3.5 minutes is relative to the paper's statistical disruption criterion.
4. **The PREFACE baseline is an analogue.** It thresholds the highest per-service signal; the paper thresholds the global reconstruction error and then ranks. The recordings store only per-service signals.
5. **Sudden CPU faults only**, on 3 services, on one node, with mock services.
6. **The graph influence in the DBN was zero** (calibration bug, Stage 08). Results reflect temporal belief plus causal root-cause analysis, not learned propagation.
7. **The operator was tested once**, in shadow mode. No live mitigation has been measured.
8. **The source paper reports 13–102 minutes of warning;** ours is minutes, because these faults reach users quickly.

## 8. Questions a reviewer may ask

**Q: What is your main result in one sentence?**
A: On live recordings, PREFACE's own 3σ rule false-alarmed on all 6 healthy runs while PREFACE-DBN raised none on any seed, with equal detection and better root-cause accuracy (3/3 against 2/3). It's a small sample.

**Q: How is the comparison fair?**
A: Both read identical recorded anomaly signals from the same Rectifier and autoencoder; only the decision layer differs.

**Q: Why report 8 seeds?**
A: The particle filter is random. A single seed once showed 100% recall where the true range was 33–67%.

**Q: Your PREFACE detects faster. Isn't that better?**
A: No. It was alarming before every fault and on most healthy ticks, so its "detection" is an always-on alarm. On the train fault it alarmed before the fault was visible and blamed the wrong service.

**Q: Did you tune the threshold to make PREFACE look bad?**
A: We used the paper's rule (z > 3) and also show the sweep. PREFACE still false-alarms on 2 of 6 healthy runs even at 19σ. Our earlier mistake, 3.0, was the version that flattered PREFACE.

**Q: Is the evaluation in-sample?**
A: Yes, for the DBN's parameters. That's the main thing to fix next, with a separate recording for testing.

**Q: Did you really predict before users were affected?**
A: Relative to the paper's statistical disruption criterion, by 3–4 minutes. Relative to the first visible latency rise, detection was roughly simultaneous, because these CPU faults hit latency within a minute. Gradual faults are needed to show true early warning.
