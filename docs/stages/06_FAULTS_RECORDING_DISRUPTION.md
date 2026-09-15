# Stage 06: Fault Injection, Recording Runs, and Detecting Disruption

> To test a failure predictor we need failures whose **time and place we know**. We inject CPU stress with Chaos Mesh, record every minute of each run to a file, and use the paper's statistical test to find **the moment users were affected**. That moment is the ground truth for calibration (Stage 08) and evaluation (Stage 12).

---

## 1. In simple words

Like a fire drill:

1. We **start a controlled fire** in one room (CPU stress on one service) at a known minute.
2. We **film everything**: the model's view (anomaly signals) and the audience's experience (user latency and errors), minute by minute, saved to a file.
3. A **referee** looks at the audience and decides the minute people clearly noticed. That's a statistics test, not an opinion.
4. Between drills we **wait for the smoke to clear**, so the next drill starts clean.

Healthy runs are the same recording **without** a fire. They tell us whether the model cries wolf.

## 2. Why this stage exists

- **Ground truth.** Calibration needs labelled Normal/Degrading/Critical periods; evaluation needs to know whether and where a fault really happened.
- **Earliness needs two timestamps**: when the model alarmed, and when users were affected. The second one is the disruption.
- **Honesty by construction.** Recording and scoring are separate programs. Each file is stamped `source: live` or `source: synthetic`, and the evaluator refuses to mix them. An earlier evaluation generated its own random signals and scored them in the same process ("100% precision"), which measured the random generator, not the system.

---

## 3. Step by step

### 3.1 One-time: install Chaos Mesh

```bash
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh --create-namespace \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
  --set dashboard.create=false --wait --timeout 10m
```

kind uses **containerd**, so Chaos Mesh's node daemon must be told where containerd's socket is. (In Git Bash on Windows, prefix the command with `MSYS_NO_PATHCONV=1`, or the socket path gets mangled.)

### 3.2 Record a session

```bash
python scripts/37_record_runs.py --mode live --dry-run          # preflight only
python scripts/37_record_runs.py --mode live --healthy-runs 3 --faulty-runs 3 \
    --pre-fault-ticks 8 --post-fault-ticks 20 --interval 60 --rate-window 2m \
    --out data/experiments/runs_v2
```

### 3.3 What the recorder does

```
Preflight: kubectl reachable? autoencoder file exists? Prometheus healthy? StressChaos CRD installed?
Plan:      3 healthy runs, then 3 faulty runs cycling targets train → route → order
           run ids: live_healthy_000..002, live_fault_003 (train), 004 (route), 005 (order)

For each run (28 ticks, one per 60 s):
   if faulty and tick == 8: apply StressChaos to the target
   signals  = Prometheus → Rectifier → autoencoder        (what the model sees)
   workload = p95 latency, error rate, request rate at the entry point (what users feel)
   append {tick, anomaly_signals, workload}
   (finally) if a fault was injected: delete the StressChaos

After each run: detect disruption from the workload series → store t_disruption
Save:  <out>/<run_id>.json
After a faulty run (except the last): wait until user latency has recovered
```

The whole v2 session took about 3 hours: 6 × 28 minutes plus recovery waits.

---

## 4. Technical depth

### 4.1 The fault: Chaos Mesh StressChaos

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata: { name: preface-cpu-<service>, namespace: default }
spec:
  mode: all                                   # every pod matching the selector at injection time
  selector: { namespaces: [default], labelSelectors: { app: <service> } }
  stressors: { cpu: { workers: 2, load: 80 } } # 2 CPU burners at 80%
  duration: '30m'                             # upper bound; the recorder deletes it at run end
```

Chaos Mesh's daemon runs CPU-burning stress workers **inside the target container's cgroup**, so the burn is counted as that container's CPU, and it's capped by that container's **150m limit**. That's why the stressed pod reads exactly 150m.

**Targets and why:**
| Target | Why chosen |
|---|---|
| `ts-train-service` | has an autoscaler and a dependency (route) |
| `ts-route-service` | leaf service, **no autoscaler**, so the fault persists |
| `ts-order-service` | has an autoscaler and a chain (payment → inventory) |
| (`ts-ui-dashboard` excluded) | it is the graph root, which gives root-cause analysis nothing to separate |

### 4.2 A tick's timing, and why tick 8 looks normal

The fault is applied **at the start** of tick 8, and that tick's metrics are queried seconds later. A `rate(...[2m])` over the last 2 minutes barely contains the stress yet, so **tick 8 looks healthy**. The fault becomes visible at tick 9. This one-tick lag shows up in the labels (Stage 08) and in the disruption floor (4.7).

### 4.3 The run file

```json
{
  "run_id": "live_fault_004", "source": "live", "is_positive": true,
  "t_fault": 8, "injected_service": "ts-route-service", "fault_type": "cpu",
  "t_disruption": 13, "interval_seconds": 60.0, "schema_version": 2,
  "services": ["ts-ui-dashboard", "ts-user-service", "..."],
  "ticks": [
    {"tick": 9,
     "anomaly_signals": {"ts-route-service": 10.66, "ts-ui-dashboard": 3.16, "ts-train-service": 1.79, "...": 0.0},
     "workload": {"p95_latency_ms": 76.0, "error_rate": 0.0, "request_rate": 3.4}},
    "..."
  ]
}
```

`RunRecord` validates on load: a faulty run must have `t_fault` and a valid `injected_service`, and a healthy run must have neither. A mislabelled run silently corrupts every metric, so it fails loudly.

### 4.4 Waiting for recovery between runs

A CPU fault outlives its removal: latency took **3.0–4.0 minutes** to settle. After each faulty run, `wait_for_recovery`:

```
threshold = max(1.5 × median(pre-fault p95), 1.1 × max(pre-fault p95))
poll every 30 s; count "calm" polls where p95 < threshold
release when ≥ 180 s have passed AND 3 consecutive calm polls; give up (with a warning) at 600 s
```

The threshold is relative to **that run's own healthy latency**, so ordinary jitter can't stall the session. In v2, both waits released at 180 s, with p95 back to 22 ms and 21 ms.

### 4.5 Disruption detection (`src/disruption.py`)

**Definition (from the PREFACE paper, §4.1.2):** a disruption is the first tick at which a user-facing metric is worse than the healthy baseline **both significantly and substantially**.

For each tick t after the baseline:

```
baseline = p95 (and error rate) at ticks 0..7            (8 pre-fault ticks)
window   = ticks max(8, t−4) .. t                         (up to 5 ticks)
test 1   Mann-Whitney U, one-sided "window > baseline":  p < α / (number of ticks tested)
test 2   Vargha-Delaney A12(window, baseline) ≥ 0.71     ("large" effect)
tick t "meets the criterion" if EITHER latency OR error rate passes BOTH tests
disruption = first tick of a run of 3 consecutive ticks meeting the criterion
```

**Mann-Whitney U, in plain words:** put all baseline and window values in one sorted list. If the window values sit near the top far more than chance allows, the window is significantly higher. It uses ranks, not raw values, so it's robust to latency's heavy tails.

**A12, in plain words:** pick one window value and one baseline value at random. A12 is the probability the window value is larger (ties count half). 0.5 means no difference and 1.0 means every window value is larger. The thresholds are 0.56 (small), 0.64 (medium) and **0.71 (large)**.

```
A12 = ( R_window / n_window − (n_window + 1)/2 ) / n_baseline      (R = rank sum)
```

**Why both, plus a correction and persistence:**
- A p-value alone flags tiny shifts once there is enough data. A12 keeps only substantial ones.
- The test is repeated at every tick (20 times per run). At α = 0.05, about 1 in 20 would be flagged by chance. **Bonferroni**: α = 0.05 / 20 = **0.0025**.
- **Persistence 3** removes one-off flukes.
- On simulated data, the bare rule flagged **38.5%** of healthy runs; with the correction and persistence together, **0.5%**, while still catching every simulated fault.

The detector also runs on **healthy** runs: a "disruption" there means it's firing on noise. v1 and v2 each had **0/3** false disruptions.

### 4.6 Worked example: route fault (v2)

Baseline p95 (ticks 0–7): `23 23 22 11 21 23 23 22` ms. α = 0.0025.

| t | window (ms) | p-value | A12 | meets? |
|---|---|---|---|---|
| 8 | 10 | — (fewer than 2 values) | — | no |
| 9 | 10, 76 | 0.556 | 0.50 | no |
| 10 | 10, 76, 93 | 0.249 | 0.67 | no |
| 11 | 10, 76, 93, 95 | 0.107 | 0.75 | no |
| 12 | 10, 76, 93, 95, 76 | 0.047 | 0.80 | no (p too big) |
| **13** | 76, 93, 95, 76, 66 | **0.0008** | **1.00** | **yes** |
| 14 | 93, 95, 76, 66, 66 | 0.0008 | 1.00 | yes |
| 15 | 95, 76, 66, 66, 84 | 0.0008 | 1.00 | yes → 3 in a row |

**Disruption = tick 13**, 5 ticks (5 minutes) after injection.

### 4.7 ⚠ The measurement floor

Look at the table again. **User latency tripled at tick 9** (76 ms against a 23 ms baseline), but the test couldn't confirm it until tick 13. That's partly built into the test:

- The window always starts at tick 8, the injection tick, which still looks normal (4.2), until t ≥ 13.
- While the window contains that normal value, and has at most 5 values against 8 baseline values, the smallest possible one-sided p-value doesn't drop below 0.0025 in time. At t = 12 it was 0.047.
- The smallest p for 5 window values all above 8 baseline values is 1 / C(13, 5) = 1/1287 = 0.00078. That's reachable only once the window is ticks 9–13.

So with this setup (8-tick baseline, 5-tick window, Bonferroni over 20 ticks), **a sudden fault cannot be confirmed earlier than 5 ticks after injection**. Train and route both landed exactly on that floor (tick 13); in both, p95 had already risen at tick 9. Order is different: latency rose at tick 10 and the autoscaler pushed it back down, so its tick 19 reflects real behaviour.

**What this means for the results:** the "3.5 minutes of warning" (Stage 12) is measured against this statistical criterion. The DBN alarmed at tick 9–10, about when user latency **first visibly rose**, not minutes before it.

### 4.8 Timing metrics

```
error interval      = t_disruption − t_fault           (how long users were protected by nothing)
reaction interval   = t_detect − t_fault                (how fast the model noticed)
earliness (warning) = t_disruption − t_detect           (positive = warned before disruption)
earliness %         = 100 × earliness / error interval  (how the paper reports it)
```

### 4.9 v1 → v2: three recording flaws fixed

| Flaw in v1 (Sep 11) | Evidence | Fix in v2 (Sep 15) |
|---|---|---|
| Latency averaged over **every** edge in the mesh | order fault (20% of traffic) invisible: global p95 flat, entry p95 up 2–3× | measure only loadgen → dashboard |
| Runs recorded **back to back** | next run's baseline opened at 72–127 ms, higher than the fault | `wait_for_recovery` |
| Only **10 ticks** after the fault | route's criterion first met at tick 16; run ended at 17 | 20 ticks after the fault |

---

## 5. Results

| Fault | v1 disruption | v2 disruption | v2 time after injection |
|---|---|---|---|
| ts-train-service | tick 11 | tick 13 | 5 min (at the floor) |
| ts-route-service | missed | tick 13 | 5 min (at the floor) |
| ts-order-service | missed | tick 19 | 11 min |
| healthy runs | 0/3 false | 0/3 false | — |

Every v2 disruption coincided with the latency test. Error rate stayed at zero in the ticks we inspected, so the faults showed up as slowness, not failed requests.

**Autoscaling during the order fault:** it scaled from 1 to 3 replicas within 2 ticks. Latency dipped (36, 11, 30, 34 ms) and then climbed again, because only the original pod was stressed and it stayed in rotation. Autoscaling delayed the disruption; it did not prevent it.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Recorder, injectors, recovery wait | `scripts/37_record_runs.py` |
| Run file format and validation | `src/run_dataset.py` |
| Mann-Whitney + A12 + persistence + Bonferroni; earliness | `src/disruption.py` |
| Example manifests | `manifests/chaos/cpu-stress-train.yaml`, `manifests/chaos/network-delay-station.yaml` (not used in results) |

## 7. Limits and known issues

- **The measurement floor** (4.7) makes early disruptions look later than they are.
- **Only sudden CPU faults.** No gradual ramps, memory leaks or network delay, even though gradual faults are where temporal reasoning should matter most.
- **The stress hits only the pods that exist at injection.** New replicas are healthy, which is realistic for some bugs and not for others.
- **Small n:** 3 faults and 3 healthy runs per recording.
- **Short baseline** (8 ticks) limits the statistical power.
- **One fault at a time**, always on one of three services.

## 8. Questions a reviewer may ask

**Q: How do you know when the failure became "disruptive"?**
A: We use the paper's criterion. User p95 latency or error rate must be worse than the pre-fault baseline, by a one-sided Mann-Whitney U test (Bonferroni-corrected) and with a Vargha-Delaney A12 of at least 0.71, for 3 consecutive ticks.

**Q: Why both a p-value and an effect size?**
A: Significance says the difference is unlikely to be chance; the effect size says it's big enough to matter. Either alone misleads.

**Q: Why the Bonferroni correction and persistence?**
A: The test is repeated every tick. Without them, the detector flagged 38.5% of simulated healthy runs; with them, 0.5%.

**Q: Isn't a 5-minute disruption time suspiciously equal for two faults?**
A: Yes, and it's explained. With an 8-tick baseline and a 5-tick window starting at the still-normal injection tick, the test cannot confirm before 5 ticks. Latency actually rose one tick after injection. We report warning time against the criterion and state this floor.

**Q: Why wait between runs?**
A: The fault's effect outlasts its removal by 3–4 minutes. Without waiting, the next run's "healthy" baseline was already slow, and two v1 disruptions went undetected.

**Q: Why is autoscaling not a fix?**
A: The new replicas absorbed some load, but the stressed pod stayed in the Service's rotation, so latency came back and the disruption arrived at 11 minutes instead of 5.
