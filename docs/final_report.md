# PREFACE-DBN: Final Report

## Summary

PREFACE-DBN predicts microservice failures on an autoscaling Kubernetes cluster before users are affected, names the service responsible, and decides on a mitigation. It extends PREFACE (Denaro et al., FSE 2024) in two ways: it replaces PREFACE's memoryless alarm threshold with a Dynamic Bayesian Network (DBN) that reasons over time and over the service call graph, and it adds a Kubernetes operator that turns the DBN's output into a safe, gated action.

The system was built and run end to end on a live cluster with real telemetry. The main findings:

- **It detected all 3 injected CPU faults, raised no false alarms on healthy runs, named the correct service every time, and gave 3–4 minutes of warning before users were affected.**
- **Against PREFACE's own alarm rule, PREFACE-DBN removes a severe false-alarm problem.** Across both live recordings PREFACE alarmed on all 6 healthy runs, for most of each run; PREFACE-DBN alarmed on none, on every seed. PREFACE-DBN also named the right service on 3 of 3 faults where PREFACE managed 2. This rests on 6 healthy runs and 6 faults.
- **The operator, running the same model live, named the faulty service on the first tick and passed all 9 end-to-end checks — but its 11-tick debounce delayed the action to 10.9 minutes, later than the 5 minutes the same fault took to reach users in the recorded runs.** It changed nothing on the cluster in shadow mode and cleared within 2 minutes of the fault's removal.
- **Autoscaling delays a CPU fault's impact on users without preventing it.**

Earlier versions of this report and of the evaluation published results that do not hold, including an earlier version of this very comparison that concluded the DBN had no advantage. They are listed under [Corrections to earlier results](#corrections-to-earlier-results).

## Problem

Kubernetes heals reactively: it restarts a container after it crashes and replaces a pod after it fails a health check. By then users have already seen errors. There is usually a window of minutes between a fault starting to corrupt a service and the failure becoming visible, and that window is where prediction is useful.

Autoscaling makes prediction hard. The number of pods, and therefore the number of metrics collected each minute, changes continuously; the PREFACE paper measured TrainTicket producing anywhere from 3,444 to 3,636 metrics per minute in steady operation, and up to 59,512 at full scale-out. A neural network needs a fixed number of inputs. PREFACE solved this with its Rectifier, but then decided with a memoryless rule: alarm when reconstruction error exceeds its training mean plus three standard deviations, and blame the service with the highest score. That rule ignores both how a service's health evolves over time and which services depend on which.

## System

| Stage | What it does |
|---|---|
| **Rectifier** | Groups each tick's pod metrics by service and reduces every metric to seven statistics — mean, min, Q1, median, Q3, max and pod count — so the vector has a fixed length whatever the replica count. Services with no running pods, and whole telemetry gaps, are filled from an exponential moving average (α = 0.2). |
| **Autoencoder** | Symmetric bottleneck network with a linear output, trained only on healthy telemetry (median/IQR normalisation, 50 epochs). For each service, the reconstruction error is converted to a z-score against the errors seen on healthy training data, then log-scaled: the anomaly signal is log1p(z). |
| **DBN** | Each service has a hidden state: Normal, Degrading or Critical. A hand-written NumPy particle filter (500 particles) tracks those states using Gaussian emissions per state, a transition matrix, and the influence of parent services along the call graph. Observations are clipped at 15. Parameters are calibrated from recorded fault runs. |
| **Root-cause analysis** | A directional causal analyzer over the service graph discovered from Istio, which separates a service's own evidence from pressure inherited from its dependencies. |
| **Decision policy** | Picks the action with the highest expected utility among Do_Nothing, Scale_Out, Restart_Pod, Reschedule_Pod and Traffic_Shift. An action is only eligible after an 11-tick debounce, with P(Critical) ≥ 0.95, outside a 300 s per-service cooldown, and within a limit of 3 actions per hour. |
| **Operator** | A Kopf controller that runs the pipeline every 60 s and writes risk, root cause, decision and health to a `FailurePredictor` resource. Changing the cluster requires two independent switches: `PREFACE_ALLOW_LIVE=true` in the operator's environment and `shadowMode: false` on the resource. With either off it only records what it would do. `Restart_Pod` and `Scale_Out` are implemented; the other actions return `NOT_IMPLEMENTED`. |

## Testbed

| | |
|---|---|
| Cluster | kind (Kubernetes 1.34), one node, 11 GB |
| Services | 8 mock HTTP services modelled on TrainTicket, calling each other synchronously |
| Call graph (discovered from Istio) | ui-dashboard → user, train, order, station; train → route; order → payment → inventory |
| Autoscaling | HPA on train, order and ui-dashboard, 1–3 replicas |
| Traffic | In-cluster load generator, 2–10 requests/s on a 6-minute sine |
| Telemetry | Prometheus with node-exporter and kube-state-metrics; Istio request metrics |
| Faults | Chaos Mesh `StressChaos`, 2 CPU workers at 80%, capped in practice by each pod's 150m CPU limit |
| Cadence | 60 s ticks, 2-minute `rate()` window |

## Method

**Collection and scoring are separate processes.** A recorder writes each run to a file stamped `source: live` or `source: synthetic`; the evaluator replays those files, refuses to mix the two kinds, and labels anything synthetic as a smoke test regardless of how it was invoked.

**Disruption** is the first tick where user-facing p95 latency (requests from the load generator into ui-dashboard) or error rate is worse than the pre-fault baseline both significantly (one-sided Mann-Whitney U) and substantially (Vargha-Delaney A12 ≥ 0.71), for 3 consecutive ticks, with a Bonferroni correction for the number of ticks tested. The persistence and correction are necessary: on simulated data, the bare rule flagged 38.5% of healthy runs; with both, 0.5%, while still catching every simulated fault.

**Calibration** uses weak supervision from the injection schedule. Before the fault is Normal; the interval from fault to disruption is split into Degrading then Critical; after the disruption is Critical.

**The PREFACE baseline** alarms when any service's anomaly signal exceeds log1p(3) = 1.386 — PREFACE's rule that error must be more than 3σ above healthy training error — and blames the highest-scoring service. Both reasoners read the same recorded signals, so only the decision layer differs.

**The DBN is evaluated over 8 random seeds**, because its particle filter is stochastic. A single seed once reported 100% recall where eight seeds gave 33–67%.

**Metrics.** Detection latency is the time from fault to first alarm. Warning time is the time from alarm to disruption. Root-cause accuracy is the share of detected faults blamed on the injected service. A false alarm is any alarm on a healthy run.

## Results

Two live recordings were made, each with 3 healthy runs and 3 CPU-fault runs (train, route and order services). Between them, three flaws in the recording method were found and fixed: latency was averaged across the whole mesh, which hid faults in low-traffic services; runs were recorded back to back, so the previous fault contaminated the next baseline; and too few ticks were recorded after each fault to confirm a late disruption.

### Disruption detection

| Fault | First recording | Second recording | Reached users after |
|---|---|---|---|
| ts-train-service | tick 11 | tick 13 | 5 min |
| ts-route-service | missed | tick 13 | 5 min |
| ts-order-service | missed | tick 19 | 11 min |
| Healthy runs | 0/3 false | 0/3 false | — |

### PREFACE vs PREFACE-DBN, second recording

| Metric | PREFACE (z > 3) | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected | 3/3 | 3/3 on every seed |
| **Healthy runs with a false alarm** | **3/3**, alarming on 16–20 of 28 ticks | **0/3 on every seed** |
| Root cause correct | 2/3 | 3/3 on every seed |
| Detection latency | 0.67 tick \* | 1.71 ± 0.42 ticks |
| Warning time before disruption (median) | 5.0 min \* | 3.5 ± 0.5 min |

\* PREFACE was already alarming before every fault (3–5 of 8 pre-fault ticks), so its detection and warning times reflect an alarm that is almost always on, not early detection.

On the first recording the pattern is the same: PREFACE false-alarmed on all 3 healthy runs and named the right service on 1 of 3 faults; PREFACE-DBN raised no false alarm on any seed, though with v1's thinner calibration it detected only 33–67% of faults.

### Why a 3σ rule fails here, and what the threshold does

Healthy anomaly scores on this cluster routinely exceed 3σ of the training error. The likely reason is that the autoencoder's healthy training data under-represents live variation: held-out healthy validation peaked at 1.65, but live healthy runs peaked between 2.23 and 6.47. A fixed threshold passes that through. PREFACE-DBN's emissions are calibrated on recorded runs, so that level of healthy noise falls inside its Normal state, and a one-off spike cannot move a persistent belief on its own.

| PREFACE threshold | Healthy runs with a false alarm (both recordings) |
|---|---|
| **z > 3, the paper's rule** | **6 of 6** |
| z > 10 | 5 of 6 |
| z > 19 | 2 of 6 |
| **PREFACE-DBN** | **0 of 6 on every seed** |

PREFACE only approaches PREFACE-DBN when its threshold is raised far above its published rule, and even at 19σ it still false-alarms on 2 of 6 healthy runs.

### Autoscaling

During the order-service fault the autoscaler added two replicas within 2 ticks. Chaos Mesh had stressed only the pod that existed at injection, so the new replicas were healthy and user latency briefly dropped, but the stressed pod stayed in the service's rotation and latency climbed back. The disruption arrived after 11 minutes, against 5 for the two services without that absorption.

Per-run tables, the evidence for each recording fix, calibration details and the full head-to-head are in [`RESULTS_LIVE.md`](RESULTS_LIVE.md).

## Operator under a live fault

The operator was tested end to end in shadow mode: CPU stress was injected into ts-route-service and the operator's published status was read every tick (`scripts/42_operator_fault_test.py`).

| Minutes after fault | What the operator reported |
|---|---|
| 0.8 | ts-route-service named as root cause, P(Critical) 0.01 |
| 1.9 – 3.9 | P(Critical) 0.67, 0.94, 0.98 |
| 10.9 | Debounce reached 11 ticks; `Reschedule_Pod` decided and logged as `WOULD_EXECUTE` |
| 11.9 – 14.0 | Cooldown, no further action |
| 16.0 | Two minutes after the fault was removed: P(Critical) 0, no root cause |

All 9 checks passed, including that the deployment's generation, replicas and restart stamp were unchanged, and every tick took 0.06–0.10 s against a 5 s budget.

**Two findings.** First, the debounce, not the model, sets the response time. The model was confident within 4 minutes, but the action waited for the 11th tick, while the same fault reached users after 5 minutes in the recorded runs. In this configuration the operator would act after users were already affected. Second, recovery is abrupt: P(Critical) fell from 1.00 to 0.00 in one tick despite a 0.96 Critical-to-Critical transition, because the calibrated emissions are narrow enough that a normal observation overwhelms the transition prior.

## Corrections to earlier results

- **"PREFACE-DBN shows no advantage over PREFACE" (first version of the live comparison)** used a PREFACE threshold of 3.0, described as its m + 3σ rule. The anomaly signal is log1p of a z-score, so 3.0 is about 19σ — a baseline far stricter than the paper's, and one that almost never false-alarmed. With the paper's rule the conclusion reverses, as reported above.
- **"100% precision, recall and F1, 0% false positives" (Goal 6)** came from an evaluation that generated its own signals with `np.random.normal`: healthy at 0.1 ± 0.1, faulty at 5.0 ± 0.5. The classes are roughly ten standard deviations apart, so any threshold between 0.5 and 4.0 scores 100%. It measured the generator. The simulation is kept, labelled as a smoke test.
- **"Parameters learned from historical telemetry" (Goal 5)** fitted EM to sequences sampled from a hand-written ground truth (`true_mu = [0.1, 3.0, 5.5]`), recovering the generator's own values. That is a sound test of the estimator, not calibration. Calibration now uses recorded fault runs.
- **"The baseline won; the DBN was tricked into blaming a proxy" (`pilot_cpu_01`)** rested on one run, and on a bug. The original root-cause localizer returned the first critical service without a critical parent in topological order, and ts-ui-dashboard — the root of the graph, with no parents — therefore won whenever it went critical. At the time, anomaly signals were also clipped at 10, which erased the difference in magnitude between a cause and its victims; signals are now log-scaled and clipped at 15. The localizer has since been replaced by the causal analyzer.
- **44% root-cause accuracy (Goal 6)** was measured on simulated signals over a flat four-service star, where the dashboard is the parent of every candidate; all 28 wrong answers blamed the dashboard. On the same simulated signals, a flat star scored 60% and the discovered graph 100% (5 runs).
- **The operator** was an empty file, and its mitigation actions logged kubectl commands instead of calling the Kubernetes API. Its inference path also used a hardcoded service graph that did not match the cluster, default rather than calibrated parameters, and a constant `node_cpu` of 0.1 against the real value the model was trained on (about 0.02). It also reported risk from a field that stays empty until the debounce is satisfied, so risk read 0 for the first 11 minutes of a fault. All of these are fixed.
- **The testbed** was described as "the TrainTicket benchmark". It is 8 mock services modelled on TrainTicket, and `yes > /dev/null` was a manual injector, not Chaos Mesh.

## Limitations

- **Few runs.** 3 faults and 6 healthy runs per recording support no statistical claim.
- **The PREFACE baseline is an analogue.** It thresholds the highest per-service signal; the paper thresholds global reconstruction error and then ranks services. Recorded runs store only per-service signals, so the global rule could not be replayed, and it may behave differently.
- **Only sudden CPU faults.** Gradual degradation, memory faults and network delay were not tested. Gradual faults are where temporal reasoning should matter most.
- **CPU features only.** Network delay is largely invisible to CPU metrics, so the class where PREFACE was weakest remains untested.
- **The autoencoder is under-dispersed on live healthy data.** Its healthy scores exceed 3σ of training error far more often than they should, which is what breaks a fixed threshold. More, and more varied, healthy training data would help both reasoners.
- **Small testbed.** One node and mock services without business logic.
- **Short warning time.** 3–4 minutes against 13–102 minutes in the PREFACE paper, because these faults reach users within 5–11 minutes.
- **Degrading is under-sampled.** Only 10–12 labels per recording, so its emission parameters are unreliable.
- **Debounce semantics.** The counter measures consecutive ticks a service is *named* root cause, not ticks it is *critical*. The expected-utility choice and the P(Critical) threshold still gate every action, so this is not a safety gap, but it is not what "11 critical ticks" suggests.
- **The debounce acts after the disruption.** With 60 s ticks, 11 ticks is 11 minutes; these faults reach users in 5–11. The debounce is what suppresses one-off spikes, so shortening it has a cost that has not been measured.
- **One operator test.** The end-to-end operator result is a single run on a single service.
- **Mitigation only in shadow mode.** `Reschedule_Pod` and `Traffic_Shift` have no live implementation, and no action has been applied to a live cluster to measure recovery.

## Future work

1. Gradual faults (ramped CPU load, memory growth) to test where temporal reasoning should outperform a threshold.
2. Network-delay faults with the multi-signal telemetry already collected (latency and error rate per edge).
3. Enough runs for confidence intervals on every metric, and a replay of PREFACE's global-error rule.
4. Live mitigation on a disposable cluster, measuring whether acting early shortens or prevents the disruption.
5. A shorter or adaptive debounce that counts ticks above the risk threshold, so the operator can act before a disruption, measured against the false actions it lets through; and live implementations of reschedule and traffic shift.

## Reproduce

```bash
python scripts/40_collect_healthy.py --minutes 25 --interval 15
python scripts/20_audit_and_train_cpu_only.py
python scripts/37_record_runs.py --mode live --healthy-runs 3 --faulty-runs 3 \
    --pre-fault-ticks 8 --post-fault-ticks 20 --interval 60 --rate-window 2m \
    --out data/experiments/runs_v2
python scripts/41_analyze_rerun.py --run-dir data/experiments/runs_v2 \
    --baseline-dir data/experiments/runs --seeds 8
```
