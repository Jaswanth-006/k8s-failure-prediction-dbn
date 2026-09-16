# Stage 13: Final Review Preparation

> A pitch you can say from memory, the numbers to know, likely panel questions with short answers, and the honest weaknesses to raise yourself. Each answer points to the stage page with the full explanation.

---

## 1. The 30-second pitch

> "Kubernetes fixes failures after users see them. The PREFACE paper predicts failures early with an autoencoder, but decides with a fixed 3-sigma threshold, with no memory and no idea which service depends on which. We kept PREFACE's Rectifier and autoencoder and replaced its threshold with a Dynamic Bayesian Network. It tracks each service's probability of being Normal, Degrading or Critical over time, finds the root cause using the service graph discovered from Istio, and picks an action by maximum expected utility inside a safe Kubernetes operator. On live recordings from our cluster, PREFACE's rule false-alarmed on all 6 healthy runs; our DBN raised none, detected every fault and named the right service every time."

## 2. The 2-minute walkthrough (follow the data)

1. **Cluster** (01): 8 mock TrainTicket services on kind, 3 autoscalers, a load generator with a sine wave from 2 to 10 requests per second, and Istio sidecars.
2. **Telemetry** (02): every 60 s, Prometheus gives per-pod CPU and node CPU (model input), plus user p95 latency and error rate at the entry point (ground truth only).
3. **Graph** (03): Istio request counters give 7 edges, e.g. ui → train → route.
4. **Rectifier** (04): 1–3 pods per service → 7 statistics each → a fixed 63-number vector.
5. **Autoencoder** (05): trained on 100 healthy ticks; reconstruction error per service → z-score → signal = log1p(z).
6. **Faults** (06): Chaos Mesh CPU stress at tick 8; runs recorded to JSON; disruption found with Mann-Whitney U and A12.
7. **Calibration** (08): labels from the schedule (Normal / Degrading / Critical) → Gaussian emissions and a transition matrix.
8. **DBN** (07): 500-particle filter: predict with transitions, weight by the Gaussian likelihood of the signals, resample → P(Critical).
9. **Root cause** (09): own evidence + explains-dependents − explained-by-parent.
10. **Decision** (10): expected utility over 5 actions, 11-tick debounce, P ≥ 0.95, cooldown, rate limit.
11. **Operator** (11): Kopf, a `FailurePredictor` resource, shadow mode, two independent live gates.
12. **Results** (12): PREFACE 6/6 false alarms against DBN 0/6; root cause 2/3 against 3/3; operator 9/9 checks, but late.

---

## 3. Numbers to remember

| Item | Value |
|---|---|
| Services / graph edges | 8 / 7 |
| Rectifier width | 8 × 7 + 7 = **63** |
| Autoencoder layers | 63 → 31 → 15 → **7** → 15 → 31 → 63 |
| Healthy training data | 100 ticks (80 train / 20 validation), 50 epochs |
| Tick / rate window | 60 s / 2 min |
| Anomaly signal | log1p(z); **z > 3 ⇔ signal > 1.386** |
| DBN | 3 states, **500 particles**, signals clipped to [0, 15] |
| Emission μ (N, D, C) | 0.66, 7.00, 10.54 |
| Emission σ (N, D, C) | 0.80, 4.84, 0.80 |
| Transition "stay" probabilities | Normal 0.996, Degrading 0.615, Critical 0.960 |
| Labels (N / D / C) | 1284 / 10 / 50 |
| Disruption test | one-sided Mann-Whitney, α = 0.05/20 = 0.0025, A12 ≥ 0.71, window 5, persistence 3 |
| Disruptions v2 | train 13, route 13, order 19 (fault at 8) |
| Policy | debounce 11 ticks, P(Critical) ≥ 0.95, cooldown 300 s, 3 actions/hour |
| Utilities when Critical | Do_Nothing −50, Scale 20, Restart 30, Traffic 25, **Reschedule 45** |
| **False alarms on healthy runs** | **PREFACE 6/6, DBN 0/6** |
| Root cause v2 | PREFACE 2/3, DBN 3/3 |
| DBN detection latency / warning | 1.71 ± 0.42 ticks / 3.5 ± 0.5 min (against the disruption criterion) |
| Operator test | root cause named 0.8 min, P ≥ 0.95 at 3.9 min, action at 10.9 min, 9/9 checks, 0.06–0.10 s per tick |

---

## 4. Likely questions and short answers

### Motivation and novelty
**Q: What problem are you solving?**
Predicting microservice failures on an autoscaling cluster before users are affected, finding the responsible service, and acting safely. → 00

**Q: What exactly is new compared with PREFACE?**
(1) A DBN with hidden health states instead of a memoryless threshold; (2) root-cause analysis over the discovered service graph; (3) DBN parameters calibrated from recorded faults with weak labels; (4) disruption and earliness measured with the paper's statistics; (5) an expected-utility decision policy and a safe Kubernetes operator. → 00 §6

**Q: Why is autoscaling the hard part?**
The number of pods, and so the number of metrics, changes every minute, while a neural network needs a fixed input size. → 04

### Kubernetes and data
**Q: What is an HPA, and how does it behave here?**
It adds replicas when CPU exceeds 50% of the request. With a 10m request, about 5m of use triggers it, so it scales within about 2 ticks during a fault. → 01

**Q: Why mock services, not the real TrainTicket?**
Laptop memory. We kept TrainTicket's names and call structure. → 01

**Q: What does the model see, and what does it not see?**
It sees per-pod CPU and node CPU. It never sees latency or errors; those define the failure. → 02

**Q: How did you get the dependency graph?**
From Istio's `istio_requests_total` counters: one edge per pair of our services that exchanged requests. → 03

### Autoencoder
**Q: Why is it trained on healthy data only?**
To learn what normal looks like, without failure labels. High reconstruction error means unlike normal. → 05

**Q: Why log1p?**
Fault z-scores reach about 42,000; the log compresses them to about 10, in a range the Gaussians can model. → 05

### Probability and the DBN (expect the most questions here)
**Q: Explain your DBN in one minute.**
Each service has a hidden state: Normal, Degrading or Critical. Each minute the state evolves by a transition matrix (possibly shifted by a sick parent) and emits an anomaly signal from a Gaussian for that state. We track P(state | all signals so far) with a particle filter: predict by sampling transitions, weight each particle by the likelihood of the observed signals, resample, and read probabilities as particle fractions. → 07

**Q: Write Bayes' rule as you use it.**
belief_t(k) ∝ N(a_t; μ_k, σ_k) × Σ_j belief_{t−1}(j) · T[j, k]. The prior comes from the previous belief and the transitions; the likelihood comes from the emission. → 07 §4.5

**Q: Give a numeric example.**
Route fault, tick 9, signal 10.66. The prior is 0.1% Critical, but the likelihood under Normal is about 10⁻³⁴ against 0.495 under Critical. The exact posterior is 0.68 Critical, then 0.94 and 0.99 on the next ticks. → 07 §4.5

**Q: What is the Markov assumption, and is it reasonable?**
The next state depends only on the current state (and the parents). The belief summarises the history. It's a standard simplification, reasonable over 1-minute steps.

**Q: Why a particle filter instead of exact inference?**
The joint state is 3⁸ = 6,561 here, which is small, but parent coupling ties services together, and at 30 services it's 3³⁰. Particles scale linearly. We saw the cost: randomness (hence 8 seeds) and degeneracy (tick 9 showed 100% Degrading where exact inference gives 68% Critical). → 07 §4.6, §5.1

**Q: How does the graph enter the probabilities?**
Through the transition step: a child's next-state row is shifted by its worst parent's state. The calibrated shift came out zero because of a bug, so in our results the graph acted through root-cause analysis. → 07, 08 §7

**Q: Why three states?**
Degrading represents the error interval, fault present but users not yet hurt. That in-between state is what makes early warning possible. → 07

**Q: Why did the DBN not false-alarm?**
Healthy spikes (2–6) are about 10σ from Critical's 10.5 ± 0.8, and a single spike can't build a persistent belief. → 07 §5.2

### Calibration
**Q: How do you learn parameters for states you can't observe?**
Weak supervision from the fault schedule: Normal before injection, Degrading for the first half of the time to disruption, Critical after. Then maximum-likelihood Gaussians and Laplace-smoothed transition counts. → 08

**Q: Why is σ_Degrading so large?**
Degrading labels start at the injection tick, when the rate window hasn't seen the stress yet (signal about 0), and continue into ticks at about 10. That mix has a huge spread. → 08 §4.3

### Root cause and decision
**Q: How is root cause decided?**
score = 2·P(C) + P(D) + min(a, 10)/5 + (children who degraded after me) − 2.5 × (parents who degraded before me); name the top service if its score is positive and its own evidence exceeds 0.5. → 09

**Q: What is MEU? Compute one.**
EU(a) = Σ P(state)·U(state, a). For P = (0, 0.01, 0.99), EU(Reschedule) = 0.05 + 44.55 = 44.6. → 10

**Q: Why did it choose Reschedule?**
Above P(Critical) ≈ 0.40, Reschedule has the highest expected utility, and the policy only acts at 0.95 or more. → 10 §4.3

### Operator and safety
**Q: How do you prevent harmful automation?**
Shadow mode by default. Live action needs `shadowMode: false` on the resource **and** `PREFACE_ALLOW_LIVE=true` in the environment. Also a debounce, the 0.95 threshold, a cooldown, a rate limit, per-action flags and an audit log. The test verified the deployment was unchanged. → 11

**Q: What does the operator publish?**
`kubectl get fp` status: risk, root cause, decision, debounce progress, cooldown, health, recent actions, checkpoint. → 11

### Evaluation
**Q: How do you define a disruption?**
User p95 or error rate worse than baseline by a one-sided Mann-Whitney U test (Bonferroni) with A12 ≥ 0.71, for 3 ticks in a row. → 06

**Q: How is the comparison fair?**
Both reasoners read identical recorded signals; only the decision layer differs. → 12

---

## 5. Weaknesses to raise yourself (and how to frame them)

A panel trusts you more if you name these first.

| Weakness | How to say it |
|---|---|
| **Small sample** (6 faults, 6 healthy) | "Enough to show the system works end to end and a clear false-alarm gap; not enough for statistical significance." |
| **In-sample DBN calibration** | "Parameters were fitted on the recording we evaluated. A held-out recording is the first next step." |
| **Warning time vs. first visible latency** | "The 3.5 minutes is against the paper's disruption test, which with our 8-tick baseline can't fire before 5 ticks. Latency first rose one tick after injection, so detection was roughly simultaneous. Gradual faults are needed to show real lead time." |
| **Operator acts late** | "The model was 95% sure at 3.9 minutes, but the 11-tick debounce delayed action to 10.9. The fix is a debounce on high-risk ticks; its false-action cost needs measuring." |
| **Graph influence not learned** | "A key-mismatch bug made the learned parent modifiers zero, so the graph contributed through root-cause analysis only. We found and documented it." |
| **Chosen action not implemented live** | "At high risk the utilities always pick Reschedule, which has no live implementation; Restart and Scale-out do. Live mitigation is future work." |
| **PREFACE baseline is an analogue** | "We threshold per-service signals; the paper thresholds global error. The recordings hold only per-service signals." |
| **Sudden CPU faults, mock services, one node** | "The testbed was sized for a laptop. Gradual, memory and network faults are future work." |
| **Earlier inflated results** | "Early simulated evaluations showed 100%. We identified why they were invalid, rebuilt the evaluation on live data, and withdrew them. One correction (the 19σ threshold) reversed a conclusion, and we report both." |

## 6. Future work (if asked "what next?")

1. A held-out recording, and more runs for confidence intervals.
2. Gradual faults (ramped CPU, memory growth) and network-delay faults.
3. Fix the topological calibration and measure whether parent influence helps.
4. A debounce counting ticks above the risk threshold; measure false actions.
5. Live Reschedule and Traffic_Shift; measure whether early action prevents disruption.
6. Replay PREFACE's global-error rule; learn or tune the utilities.
7. Package the operator as an in-cluster Deployment with RBAC.

## 7. Demo commands (if you're asked to show it)

```bash
docker start preface-dbn-control-plane                       # restart the stopped cluster
kubectl get pods                                             # 8 services + loadgen
kubectl port-forward -n monitoring svc/prometheus-server 9090:80 --address 127.0.0.1
kopf run src/operator.py --verbose --standalone              # shadow mode
kubectl get fp train-ticket-predictor -o yaml                # live status
python scripts/41_analyze_rerun.py --run-dir data/experiments/runs_v2 --baseline-dir data/experiments/runs --seeds 8
```
