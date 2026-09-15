# PREFACE-DBN on live telemetry

Results from two recordings on a live cluster. Every number here comes from real Prometheus and Istio telemetry scored by both reasoners, not from simulated signals.

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
| Anomaly model | Rectifier + autoencoder trained on 99 ticks of real healthy telemetry |
| Disruption | First tick where user-facing p95 latency or error rate is worse than the pre-fault baseline, significantly (Mann-Whitney U) and substantially (Vargha-Delaney A12 ≥ 0.71), for 3 consecutive ticks, with Bonferroni correction |
| Reasoners | PREFACE: memoryless threshold 3.0 plus rank localization. PREFACE-DBN: particle filter plus directional causal RCA |
| Seeds | PREFACE-DBN evaluated over 8 seeds. The particle filter is stochastic, and one seed is not trustworthy |

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

With only 8 Critical labels, v1 fitted a Critical spread of σ = 0.18, so narrow that Critical ticks landing a little off μ = 10.4 were judged unlikely. That is the probable cause of v1's unstable recall (below). v2 has 50 Critical labels and σ = 0.80.

**Degrading stays under-sampled in both** (12 and 10 labels, fewer than the 30 the calibrator asks for), so its emission parameters are unreliable. These faults move from normal to disruption in 5–11 ticks, which leaves little error interval to label.

## Head-to-head

### v2 (September 15)

| Metric | PREFACE | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected (recall) | 100% | 100% ± 0 |
| False alarms on healthy runs | 0% | 0% ± 0 |
| Root cause correct | 100% | 100% ± 0 |
| Detection latency | 1.00 tick | 1.71 ± 0.42 ticks (1.00–2.33) |
| Warning time before disruption, median | 4.0 min | 3.5 ± 0.5 min (3–4) |
| Faults detected before users were affected | 3/3 | 3/3 on every seed |

### v1 (September 11), same reasoners

| Metric | PREFACE | PREFACE-DBN (8 seeds) |
|---|---|---|
| Faults detected (recall) | 100% | 42% ± 14% (33–67%) |
| False alarms on healthy runs | **67%** (2 of 3) | 0% ± 0 |
| Root cause correct | 100% | 100% ± 0 |
| Detection latency | 1.00 tick | 1.88 ± 0.22 ticks |

A single v1 seed had reported 100% DBN recall; across 8 seeds it was 33–67%. That is why every DBN figure here is given over seeds.

### Reading the comparison

**On v2 the DBN shows no advantage.** Both reasoners catch every fault, raise no false alarms and name the right service, and PREFACE is about 0.7 ticks faster. That fits the fault type: an abrupt CPU spike is exactly what a memoryless threshold handles best, and the DBN's temporal filtering only costs time.

**The v1 false-alarm gap did not replicate.** PREFACE false-alarmed on two v1 healthy runs but on no v2 healthy run. The anomaly pipeline and threshold are identical across both recordings, so this is run-to-run variation in how high healthy signals happen to peak, not an effect of the v2 fixes.

**PREFACE-DBN's only remaining edge is false alarms, pooled across both recordings: PREFACE 2 of 6 healthy runs, PREFACE-DBN 0 of 6.** That is consistent with the DBN's purpose, suppressing transient spikes, but six runs cannot establish it.

**Healthy signals are noisy, and occasionally spike hard.** The v2 healthy runs peaked at 2.23, 2.51 and 2.55, right around the Degrading boundary of 2.5 and about 0.45 below PREFACE's threshold of 3.0. The v1 healthy runs peaked at 2.48, **3.73 and 6.47**. A 6.47 spike on a run with no fault is well inside the range the calibration assigns to faulty states, and it is exactly the kind of one-off excursion a memoryless threshold fires on and temporal filtering is meant to ignore. PREFACE alarmed on both v1 spikes; PREFACE-DBN, with v1's calibration, stayed quiet on all 8 seeds. That is the strongest single piece of support for the DBN in either recording, and it is still two runs.

## Per-run detail (v2)

Every fault was injected at tick 8. Warning time is the number of ticks (minutes) between detection and disruption; the percentage is how much of the fault-to-disruption window was still ahead at detection, the source paper's convention.

| Run | Disruption | PREFACE detects | PREFACE-DBN detects (8 seeds) | Warning, PREFACE | Warning, DBN (median) | Service blamed |
|---|---|---|---|---|---|---|
| train fault | tick 13 | tick 9 | ticks 9–12, median 10 | 4 min (80%) | 3 min (60%) | train-service, by both, on every seed |
| route fault | tick 13 | tick 9 | ticks 9–10, median 10 | 4 min (80%) | 3 min (60%) | route-service, by both, on every seed |
| order fault | tick 19 | tick 9 | ticks 9–10, median 10 | 10 min (91%) | 9 min (82%) | order-service, by both, on every seed |
| 3 healthy runs | none | no alarm | no alarm on any seed | — | — | — |

PREFACE flags each fault one tick after injection; PREFACE-DBN takes one to four ticks as its belief accumulates. The order-service fault has the longest warning time because autoscaling delayed the *disruption*, not its detection.

The two warning-time figures in this document are aggregated in opposite orders, so they need not match. The per-run column takes each run's median detection tick across the 8 seeds, giving 3, 3 and 9 minutes. The head-to-head's 3.5 ± 0.5 minutes takes the median across the three runs within each seed (3 or 4 minutes), then averages over seeds.

## Healthy false alarms, both recordings

Both recordings share the anomaly pipeline and threshold, so their healthy runs can be pooled for false alarms.

| Recording | Run | Peak anomaly signal | PREFACE | PREFACE-DBN (8 seeds) |
|---|---|---|---|---|
| v1 | healthy_000 | 2.48 | quiet | quiet |
| v1 | healthy_001 | 3.73 | **alarm** | quiet |
| v1 | healthy_002 | 6.47 | **alarm** | quiet |
| v2 | healthy_000 | 2.23 | quiet | quiet |
| v2 | healthy_001 | 2.51 | quiet | quiet |
| v2 | healthy_002 | 2.55 | quiet | quiet |
| **pooled** | | | **2 of 6** | **0 of 6** |

## Caveats

- **n is small.** Three faults and six healthy runs support no statistical claim. Scoring 3/3 on root cause by chance, with three candidate services, has probability 1/27 (3.7%).
- **Abrupt faults only.** Gradual degradation, where temporal reasoning should matter most, is untested.
- **Warning time is short.** 3.5–4 minutes, against 13–102 minutes in the source paper, because these faults reach users within 5–11 minutes.
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
