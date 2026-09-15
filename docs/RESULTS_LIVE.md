# PREFACE-DBN on live telemetry

Results from two recordings on a live cluster. Every number here comes from real Prometheus and Istio telemetry scored by both reasoners, not from simulated signals.

> **Correction.** An earlier version of this document compared against PREFACE with an alarm threshold of 3.0 and described it as PREFACE's m + 3σ rule. The anomaly signal is log1p of a z-score, so 3.0 is actually about 19σ — a far stricter PREFACE than the paper describes — and that version concluded PREFACE-DBN had no advantage. The comparison below uses the paper's rule, z > 3, which is log1p(3) = 1.386 in these units, and the conclusion is reversed.

## Setup

| | |
|---|---|
| Cluster | kind (Kubernetes 1.34), single node, 11 GB |
| Workload | 8 mock microservices modelled on TrainTicket, Istio sidecars, HPA on train, order and ui-dashboard (1–3 replicas) |
| Traffic | In-cluster load generator, 2–10 req/s on a 6-minute sine |
| Telemetry | Prometheus with node-exporter and kube-state-metrics, Istio request metrics |
| Faults | Chaos Mesh `StressChaos`, 2 CPU workers at 80%, injected at tick 8 |
| Targets | ts-train-service, ts-route-service, ts-order-service |
| Tick | 60 s, `rate()` window 2 min |
| Anomaly model | Rectifier + autoencoder trained on 99 ticks of real healthy telemetry. Each service's signal is log1p of the z-score of its reconstruction error against healthy training error |
| Disruption | First tick where user-facing p95 latency or error rate is worse than the pre-fault baseline, significantly (Mann-Whitney U) and substantially (Vargha-Delaney A12 ≥ 0.71), for 3 consecutive ticks, with Bonferroni correction |
| PREFACE | Alarm when any service's signal exceeds log1p(3) = 1.386, i.e. its error is more than 3σ above healthy training error; blame the highest-scoring service |
| PREFACE-DBN | Particle filter over hidden health states plus directional causal root-cause analysis, evaluated over 8 seeds because the filter is stochastic |

## Two recordings

| | September 11 (v1) | September 15 (v2) |
|---|---|---|
| Runs | 3 healthy + 3 faulty | 3 healthy + 3 faulty |
| Ticks per run | 18 (8 before the fault, 10 after) | 28 (8 before, 20 after) |
| Latency measured on | every edge in the mesh | user requests only (loadgen → ts-ui-dashboard) |
| Pause between runs | none (3.1 min gap) | waits until latency recovers |

v1 exposed three methodological problems. v2 fixes all three.

### Disruption detection

| Fault | v1 | v2 | Reached users after |
|---|---|---|---|
| ts-train-service | tick 11 | tick 13 | 5 min |
| ts-route-service | **missed** | tick 13 | 5 min |
| ts-order-service | **missed** | tick 19 | 11 min |
| healthy runs | 0/3 false | 0/3 false | — |

## What was wrong in v1, and the evidence

### 1. Averaging latency over the whole mesh hid the order-service fault

ts-order-service receives about 20% of user requests. When it slows down, a fifth of user-facing requests are slow, well above the 5% tail that p95 reads. Pooled with every fast internal hop (payment→inventory, train→route and so on), the slow fraction fell below 5% and the global p95 did not move.

Replaying Prometheus history over the v1 order-service fault, p95 in ms after injection:

```
global (all edges)   14   9  18  27  23  24  21  10  10  23   flat: fault invisible
entry (user-facing)  21  10  42  59  51  58  40  19  16  48   rises 2-3x
```

The same replay over the three v1 healthy runs raised no false disruptions with the entry-point query.

### 2. Back-to-back runs contaminated the next baseline

A CPU fault outlives its removal. Measured from Prometheus, entry p95 took **3.0, 3.5 and 4.0 minutes** to settle after the three v1 faults, but each run started **3.1 minutes** after the previous one ended. User-facing p95 in the 6 minutes before each v1 fault run started:

```
before train fault (after a healthy run)   21  12  10  20  22  22  21
before route fault (after train fault)     76  99 127 104  94  91  79
before order fault (after route fault)     87  86  69  66  72  79  92
```

A baseline that opens at 72–127 ms is higher than the fault that follows it, so the rank test finds nothing worse. Dropping the two contaminated baseline ticks recovers the missed route-service disruption with either query.

In v2 the recorder waits after each fault until p95 stays under a threshold relative to that run's own pre-fault latency (1.5× the median, at least 1.1× the maximum) for three consecutive 30-second polls, after a 3-minute minimum. Both v2 cooldowns released at 180 s, with p95 back to 22 ms and 21 ms. The route-service run then opened on a clean baseline:

```
v1 route-service pre-fault p95   72  22  23  21  14  13  19  22   replayed from Prometheus
v2 route-service pre-fault p95   23  23  22  11  21  23  23  22   captured live
```

The v1 row is a replay: v1 run files stored only the old whole-mesh latency, so there is no live entry-point series for them. Replayed ticks can shift by one depending on alignment (an alignment one minute earlier gives `79 72 22 23 21 14 13 19`), but the contamination at the start is present either way.

### 3. Ten ticks after the fault were too few

The v1 route-service disruption first met the criterion at tick 16. The run ended at tick 17, so it could never hold for the required 3 ticks. v2 records 20 ticks after the fault.

## Autoscaling delays the order-service disruption but does not prevent it

ts-order-service has an HPA; ts-route-service does not. During the v2 order-service fault:

```
tick   8      1 replica    18 ms   fault injected
tick  10      3 replicas   52 ms   autoscaler reacts within 2 ticks
ticks 11-12                64 67
ticks 13-16                36 11 30 34          extra replicas take some of the load
ticks 17-22                53 58 42 41 52 68    fault re-emerges
tick  19                                        disruption confirmed
```

Per-pod CPU shows why. Chaos Mesh stressed only the pod that existed at injection time; the replicas the autoscaler added were never stressed:

```
original pod   9m5kr   1-2m before the fault, then 22, 95, and flat at 150m for the rest of the run
new pod        4hsm6   first seen at tick 11, 0-1m throughout
new pod        bmlbv   first seen at tick 11, 0-1m throughout
```

The stressed pod sits at exactly 150m because that is its CPU limit (the three fault targets all run with `requests: 10m, limits: 150m`), so the cgroup caps the stress well below the 2 workers at 80% requested. The small request also explains how quickly the autoscaler reacts: the HPA targets 50% of a 10m request, so about 5m of use is enough to scale out. The new replicas absorbed traffic and temporarily hid the fault, but the throttled pod stayed in the Service's rotation, so a share of requests kept reaching it and latency climbed back. The disruption arrived 11 minutes after injection, against 5 minutes for the two other services.

This corrects an earlier reading. From v1 it looked as though autoscaling had absorbed the order-service fault entirely, so that "no disruption" was the right answer. The longer v2 window shows the absorption is only temporary. v1 stopped recording at tick 17, and in v2 the fault re-emerged from tick 17 onwards, so v1 most likely ended just before the same thing happened. That last step is an inference from v2: v1 has no data past tick 17.

## Calibration

DBN parameters are fitted by weak supervision from the fault schedule: before the fault is Normal, the fault-to-disruption interval is split Degrading then Critical, and after the disruption is Critical.

| | v1 | v2 |
|---|---|---|
| Faulty runs with a disruption | 1/3 | 3/3 |
| Labels (Normal / Degrading / Critical) | 844 / 12 / 8 | 1284 / 10 / 50 |
| Emission μ | 0.78, 7.71, 10.40 | 0.66, 7.00, 10.54 |
| Emission σ | 1.61, 4.73, **0.18** | 0.80, 4.84, 0.80 |

v2 transition matrix P(H_t | H_{t-1}):

```
from Normal     0.9960  0.0032  0.0008
from Degrading  0.0769  0.6154  0.3077
from Critical   0.0200  0.0200  0.9600
```

With only 8 Critical labels, v1 fitted a Critical spread of σ = 0.18, so narrow that Critical ticks landing a little off μ = 10.4 were judged unlikely. That is the probable cause of v1's unstable DBN recall (below). v2 has 50 Critical labels and σ = 0.80.

**Degrading stays under-sampled in both** (12 and 10 labels, fewer than the 30 the calibrator asks for), so its emission parameters are unreliable. These faults move from normal to disruption in 5–11 ticks, which leaves little error interval to label.

## Head-to-head

### v2 (September 15)

| Metric | PREFACE (z > 3) | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected | 3/3 | 3/3 on every seed |
| **Healthy runs with a false alarm** | **3/3**, alarming on 16, 20 and 17 of 28 ticks | **0/3 on every seed** |
| Root cause correct | 2/3 (blamed ts-ui-dashboard for the train fault) | 3/3 on every seed |
| Detection latency | 0.67 tick \* | 1.71 ± 0.42 ticks (1.00–2.33) |
| Warning time before disruption, median | 5.0 min \* | 3.5 ± 0.5 min (3–4) |

### v1 (September 11)

| Metric | PREFACE (z > 3) | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected | 3/3 | 42% ± 14% (33–67%) |
| **Healthy runs with a false alarm** | **3/3**, alarming on 4, 4 and 11 of 18 ticks | **0/3 on every seed** |
| Root cause correct | 1/3 (blamed station and inventory) | 100% of the faults it detected |
| Detection latency | 0.33 tick \* | 1.88 ± 0.22 ticks |

\* **PREFACE's detection and warning times are not meaningful here.** It was already alarming before every fault — on 3–5 of the 8 pre-fault ticks in each run — so it "detects" each fault the moment the fault starts simply because its alarm is almost always on.

A single v1 seed had once reported 100% DBN recall; across 8 seeds it was 33–67%, which is why every DBN figure here is given over seeds.

### How much the comparison depends on PREFACE's threshold

PREFACE-DBN has no threshold, so its results do not change. PREFACE's do:

| PREFACE threshold | Healthy runs with a false alarm (v1 + v2) | Root cause correct, v2 |
|---|---|---|
| **z > 3 — the paper's rule** | **6 of 6** | 2 of 3 |
| z > 5 | 6 of 6 | 3 of 3 |
| z > 10 | 5 of 6 | 3 of 3 |
| z > 19 (threshold 3.0, used in error by the earlier version) | 2 of 6 | 3 of 3 |
| **PREFACE-DBN, for comparison** | **0 of 6 on every seed** | **3 of 3 on every seed** |

### Reading the comparison

**With its own rule, PREFACE false-alarms on every healthy run in both recordings, and for most of each run.** Healthy anomaly scores on this cluster routinely exceed 3σ of the training error. The likely reason is that the autoencoder's healthy training data (80 ticks, close together in time) under-represents live variation: held-out healthy validation peaked at 1.65, but the live healthy runs peaked between 2.23 and 6.47. A fixed threshold passes that straight through.

**PREFACE-DBN raised no false alarm on any of the 6 healthy runs, on any seed.** Its emission model is calibrated on recorded runs, so that level of healthy noise falls inside its Normal state, and a one-off spike cannot move a persistent belief on its own.

**It also names the root cause more reliably** — 3 of 3 on the second recording on every seed, against PREFACE's 2 of 3 there and 1 of 3 on the first.

**PREFACE only approaches it when its threshold is raised far above its published rule**, and even at 19σ it still false-alarms on 2 of 6 healthy runs.

So on this data PREFACE-DBN keeps PREFACE's detection while removing the false alarms a 3σ rule produces on real telemetry, and localizes better. Two limits apply:

- **The PREFACE here is an analogue, not the paper's exact rule.** It thresholds the highest per-service signal; the paper thresholds the global reconstruction error and then ranks services. Recorded runs store only per-service signals, so the global rule could not be replayed, and it may behave differently.
- **Six healthy runs and six faults** support no statistical claim.

## Per-run detail (v2)

Every fault was injected at tick 8. Warning time is the number of ticks (minutes) between detection and disruption.

| Run | Disruption | PREFACE (z > 3) | PREFACE-DBN (8 seeds) |
|---|---|---|---|
| train fault | tick 13 | alarming on 5 of 8 ticks before the fault; at the fault, blamed **ts-ui-dashboard** | detected at ticks 9–12 (median 10), blamed **ts-train-service** on every seed, 3 min warning |
| route fault | tick 13 | alarming on 3 of 8 ticks before the fault; blamed ts-route-service | ticks 9–10 (median 10), ts-route-service on every seed, 3 min warning |
| order fault | tick 19 | alarming on 4 of 8 ticks before the fault; blamed ts-order-service | ticks 9–10 (median 10), ts-order-service on every seed, 9 min warning |
| healthy runs | none | false alarm on 16, 20 and 17 of 28 ticks, from tick 0 | no alarm on any seed |

The order-service fault has the longest warning time because autoscaling delayed the *disruption*, not its detection.

The per-run warning times use each run's median detection tick across the 8 seeds (3, 3 and 9 minutes). The head-to-head's 3.5 ± 0.5 minutes takes the median across the three runs within each seed (3 or 4 minutes), then averages over seeds, so the two need not match exactly.

## Operator under a live fault

The operator runs the same model live and publishes what it sees to the `FailurePredictor` status. With the operator in shadow mode (`shadowMode: true`, `PREFACE_ALLOW_LIVE` unset), CPU stress was injected into ts-route-service, which has no autoscaler, and the status was read after every tick (`scripts/42_operator_fault_test.py`).

```
min after fault   P(Critical) on root   root cause          debounce   decision
  0.8             0.01                  ts-route-service     1/11
  1.9             0.67                  ts-route-service     2/11
  2.9             0.94                  ts-route-service     3/11
  3.9             0.98                  ts-route-service     4/11
 10.9             0.99                  ts-route-service    11/11      Reschedule_Pod -> WOULD_EXECUTE
 11.9 - 14.0      0.98 - 0.99           ts-route-service    12-14/11   Do_Nothing (cooldown)
 14.0             fault removed
 15.0             1.00                  ts-route-service
 16.0 - 18.0      0.00                  none                 0/11
```

All 9 checks passed: no inference errors; every tick within the 5 s budget (0.06–0.10 s during the fault and recovery); healthy ticks quiet; ts-route-service named as root cause; P(Critical) on it reached 0.95; the intervention became eligible; `Reschedule_Pod` was decided on ts-route-service and logged only as `WOULD_EXECUTE`; the deployment's generation, replica count and restart stamp were unchanged afterwards; and risk fell once the fault was removed. The decision was written to the audit log with the expected utilities behind it (Reschedule_Pod 44.6, Restart_Pod 29.8, Scale_Out 19.9, Do_Nothing −49.5).

**The debounce makes the operator act too late for these faults.** The model was above 0.95 within 4 minutes, but the 11-tick debounce held the decision until the 11th tick, 10.9 minutes after injection. In the recorded runs the same fault on the same service reached users after 5 minutes. So in this configuration the operator predicts in time and would act about 6 minutes after users were already affected. The debounce is what suppresses one-off spikes, so shortening it trades against false actions; a debounce that counts ticks above the risk threshold, rather than ticks a service is named, would be the natural next step.

**Recovery is abrupt.** P(Critical) went from 1.00 to 0.00 between two ticks once the fault's effect left the 2-minute `rate()` window, despite a Critical-to-Critical transition probability of 0.96. The calibrated emissions are narrow (Critical σ = 0.80 around μ = 10.54), so a normal-looking observation is so unlikely under Critical that the few particles that move to Normal take nearly all the weight. Clearing quickly is correct here, but it means the transition matrix does little to smooth strong evidence in the other direction.

This is a single test run.

## Caveats

- **n is small.** Three faults and six healthy runs per recording support no statistical claim.
- **The PREFACE baseline is an analogue** built from per-service signals; the paper's global-error rule could not be replayed from the recorded data.
- **Abrupt faults only.** Gradual degradation is untested.
- **Warning time is short.** 3.5 minutes for PREFACE-DBN, against 13–102 minutes in the source paper, because these faults reach users within 5–11 minutes.
- **Replays are approximate.** Prometheus range queries evaluate on step boundaries, so replayed disruption ticks can differ from live capture by a few ticks (the v1 train-service fault was tick 11 live and tick 14 replayed). The replays are used as evidence of direction, never as headline numbers; the headline numbers come from live capture.
- **CPU features only**, a single node, and mock services without business logic.

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
