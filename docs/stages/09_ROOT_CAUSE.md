# Stage 09: Root-Cause Analysis (Which Service Started It?)

> When several services look sick, which one **caused** it? The directional causal analyzer scores every service with three kinds of evidence (its own sickness, whether it explains its dependents' sickness, and whether it is explained by a sick parent) and names the top service as root cause.

---

## 1. In simple words

A detective at a crime scene with several injured people asks:

1. **How badly hurt is each person?** (their own evidence)
2. **Did this person's trouble start before the people who depend on them?** Then they may be the origin.
3. **Is this person hurt only because someone they depend on was hurt first?** Then they're a victim.

Score = *own evidence + "I explain others" − "I'm explained by someone else"*. The highest score, if it is clearly positive, is the root cause.

## 2. Why this stage exists

- **Symptoms spread.** A slow service makes its callers wait, so several services' signals rise at once.
- **"Blame the highest score"** (PREFACE's rule) follows the loudest symptom, which can be a victim, or pure noise at the moment of alarm.
- **The operator acts on one service.** Restarting the wrong one wastes the action and adds disruption.

---

## 3. Step by step (`DirectionalCausalAnalyzer.step`, called inside the DBN each tick)

```
Input: anomaly signals a_s, DBN posteriors P_s(N/D/C), service graph

1. Track when each service started degrading
     unhealthy_s = P_s(Critical) + P_s(Degrading)
     if unhealthy_s > 0.4: remember the first tick (degradation_start) if not set
     else:                 forget it (reset to −1)
2. For every service s compute
     intrinsic  I_s  (own evidence)
     upstream   U_s  (children who are unhealthy and started after s)
     victim     V_s  (parents who are unhealthy and started before s)
     score_s = I_s + U_s − V_s
3. Rank by score
4. Declare the top service root cause only if score > 0 AND I > 0.5; otherwise "None"
5. Label each service ROOT_CAUSE / PROPAGATED_VICTIM / NORMAL (for diagnostics)
```

---

## 4. Technical depth

The live pipeline passes **no extra telemetry** to this module, so it runs its "Goal 3" path. The formulas below are exactly what runs live and in evaluation.

### 4.1 Intrinsic evidence (range 0 to 5)

```
I_s = 2 · P_s(Critical) + 1 · P_s(Degrading) + min(a_s, 10) / 5
```

- Probability of Critical counts double.
- The raw anomaly signal also counts, so a very unusual service gets evidence even before the DBN's belief has built up. This is why the root cause can be named on the first fault tick.

### 4.2 Upstream causal evidence ("I explain my dependents")

Graph convention: an edge s → c means **s calls c** (c is s's child).

```
U_s = Σ over children c with unhealthy_c > 0.3:
        unhealthy_c × (1.0 + pressure)     if s and c both degrading and start_s ≤ start_c
        unhealthy_c × 0.5                  if s degrading but c has no start tick
        0                                  otherwise
pressure = 0.1 · log1p(request_count on edge s→c)
```

(In the live path the graph is loaded without request counts, so `pressure` = 0.)

### 4.3 Victim evidence ("I'm explained by a sick caller")

```
V_s = Σ over parents p with unhealthy_p > 0.3 and start_p set and (start_s unset or start_p ≤ start_s):
        unhealthy_p × 2.5
```

### 4.4 Decision and classification

```
score_s = I_s + U_s − V_s
root_cause = argmax score  if  max score > 0  and  its I > 0.5     else "None"

PROPAGATED_VICTIM  if V > 1.0 and V > 0.5·U
ROOT_CAUSE         if score > 0 and I > 0.5 and score ≥ 0.5·I
NORMAL             otherwise
```

### 4.5 The multi-signal extension (built, not used live)

`step()` also accepts `service_telemetry` (CPU, memory, error rate) and `edge_telemetry` (edge error rate, p95). With them, it blends the intrinsic score with physical evidence (weight 0.7), requires some real stress before naming a root cause, and gates upstream and victim evidence by actual edge stress. This is "Goal 4" in `docs/GOAL4_MULTI_SIGNAL_TELEMETRY.md`. **The operator and the evaluator don't pass this telemetry**, so it isn't part of the reported results.

---

## 5. Real examples from our data (seed 0)

### 5.1 Train-service fault, tick 9 (first fault tick)

| Service | P(C) | P(D) | signal | I = 2P(C) + P(D) + min(a,10)/5 | U | V | score | label |
|---|---|---|---|---|---|---|---|---|
| **ts-train-service** | 0.69 | 0.31 | 9.73 | 1.38 + 0.31 + 1.95 = **3.64** | 0 | 0 | **3.64** | ROOT_CAUSE |
| ts-route-service (train calls it) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | **2.50** | −2.50 | PROPAGATED_VICTIM |
| ts-ui-dashboard (calls train) | 0.00 | 0.00 | 0.74 | 0.15 | 0 | 0 | 0.15 | NORMAL |

Root cause: **ts-train-service** (score 3.64, I > 0.5).
- Route got **victim evidence 2.50** = unhealthy(train) 1.00 × 2.5, because its caller train became unhealthy first.
- PREFACE, at its alarm (tick 8, the injection tick), blamed the dashboard, whose noise value 1.65 was the highest. The DBN didn't, because the dashboard's belief stayed Normal.

### 5.2 Route-service fault, tick 10

`ts-route-service`: P(C) 0.64, P(D) 0.36, signal 11.55 → I = 1.28 + 0.36 + 2.00 = **3.64** → root cause. Train 0.25, dashboard 0.19.

### 5.3 How often a root cause is named

| Run | ticks with a named root cause (of 28) | longest streak |
|---|---|---|
| train fault | 19 (every tick from 9 to 27) | 19 |
| route fault | 19 | 19 |
| order fault | 19 | 19 |
| healthy 000 | 0 | 0 |
| healthy 001 | 1 | 1 |
| healthy 002 | 2 | 1 |

On healthy runs a root cause was occasionally named **from the anomaly signal alone**: a signal above 2.5 gives I > 0.5 even with P(Critical) = 0. It was never named twice in a row, and never with high risk. The decision policy (Stage 10) resets its debounce whenever the name changes or is "None", and also requires P(Critical) ≥ 0.95, so these blips can't cause an action.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Analyzer | `src/causal_rca.py` (`DirectionalCausalAnalyzer`) |
| Called from | `src/ddn_core_phase3.py::step` (step 4) |
| Old simple localizer (fallback only) | `DynamicDecisionNetworkPhase3._localize_root_cause` |
| Design notes | `docs/GOAL3_DIRECTIONAL_CAUSALITY.md`, `docs/GOAL4_MULTI_SIGNAL_TELEMETRY.md` |

## 7. Limits and known issues

- **⚠ Direction of propagation.** The scoring treats a **callee** of a sick caller as the victim (5.1: route blamed on train). In synchronous request chains, slowness usually travels the other way: a slow callee makes its **caller** wait. If a caller and its callee become unhealthy **in the same tick** (likely with 60-second ticks), `start_caller ≤ start_callee` holds, so the caller earns upstream evidence and the callee earns victim evidence. **That would push blame toward the caller.** In our runs the callers' beliefs stayed Normal, so this never happened, but it's a real design risk. A backpressure-aware variant would reverse the victim direction for latency faults.
- **The hand-set weights** (2, 1, /5, 0.3, 0.4 thresholds, 2.5 victim weight) were not tuned or learned.
- **The anomaly term can name a root cause without any belief** (5.3). It's harmless because of the policy's gates, but it's why the debounce counts "named", not "critical".
- **The multi-signal version is untested live.**
- **A small test:** 3 faults, where the right answer was the most anomalous service each time after the fault became visible. The analyzer hasn't been tested on a case where a victim is louder than the cause.
- **An earlier bug, now fixed:** the original localizer returned the first critical service without a critical parent in topological order, so the graph root (the dashboard) won whenever it went critical. The early "the DBN blamed a proxy" pilot result came from that.

## 8. Questions a reviewer may ask

**Q: How is root cause different from "highest anomaly score"?**
A: It combines the DBN's belief (not a single reading), the anomaly, and timing along the graph: who degraded first, and who depends on whom. Victims get negative evidence.

**Q: Show me the score for the train fault.**
A: At the first fault tick, train had I = 2×0.69 + 0.31 + 9.73/5 = 3.64 and no victim evidence, so it was named. Route, which train calls, got −2.5 as a victim. The dashboard scored 0.15.

**Q: Why did PREFACE get the train fault wrong?**
A: It alarmed at the injection tick, before the fault was visible, because the dashboard's noise (1.65) crossed its line, and it blamed the highest score, which was the dashboard.

**Q: What would break this method?**
A: Simultaneous degradation of a caller and its callee (it would favour the caller), a noisy victim louder than the cause, or a wrong graph.

**Q: Why the 0.5 intrinsic threshold?**
A: So that a service needs some real evidence (probability mass or a clearly unusual signal) before being named. It's a hand-set constant.
