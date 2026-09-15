# Stage 04: The Rectifier (Variable Pods → Fixed Vector)

> The autoscaler keeps changing how many pods a service has, so the number of measurements changes every minute. A neural network needs the same number of inputs every time. The Rectifier summarises each service's pods into 7 statistics, which gives a fixed-length vector of **63 numbers** per tick.

---

## 1. In simple words

Imagine taking the temperature of a class where the number of students changes every day: 20 on Monday, 35 on Tuesday. You can't keep a spreadsheet with one column per student. Instead, record **the average, the lowest, the highest, the middle value, the quarter-points, and how many students there were**. That's always 7 columns, whatever the class size.

The Rectifier does exactly this for each service's pods: 1 pod or 3 pods, you always get 7 numbers per service.

## 2. Why this stage exists

This is **PREFACE's core idea**, and the reason the paper exists:

- An autoencoder has a fixed input layer (63 neurons here).
- With autoscaling, `ts-train-service` might have 1 pod at 10:00 and 3 at 10:02. Raw per-pod CPU would be 1 number, then 3.
- The paper measured TrainTicket producing 3,444–3,636 metrics per minute in steady state, and up to 59,512 at full scale-out. No fixed network can take that raw.

Statistics over pods solve it, and they also keep useful information: `max` shows one hot pod, `count` shows scaling.

---

## 3. Step by step (what `process_tick` does)

```
Input : rows of (timestamp, pod_name, service_name, pod_phase, is_ready, kpi_name, value)
        e.g. 3 rows for ts-train-service pods, 1 row per other service, 1 node row

Step 1  If there are no rows at all -> return the moving-average vector, flag "imputed"
Step 2  Keep only pods that are Running AND Ready
Step 3  For each service (alphabetical order):
          if it has 0 pods   -> count = 0, other 6 stats = moving average from history
          else, for each KPI -> mean, min, Q1 (25th pct), median, Q3 (75th pct), max, count
Step 4  For each node KPI (node_cpu) -> the same 7 statistics over node rows
Step 5  Write the values into a fixed-order vector x_t (63 numbers)
Step 6  Update the moving average of every feature:  ema = 0.2·value + 0.8·ema
Output: (x_t, had_a_zero_pod_service)
```

---

## 4. Technical depth

### 4.1 The feature schema

`Rectifier(services, pod_kpis=["cpu_usage"], node_kpis=["node_cpu"])` builds names in a fixed order:

```
for service in sorted(services):            # 8 services
    for kpi in sorted(pod_kpis):            # 1 KPI: cpu_usage
        for stat in [mean, min, q1, median, q3, max, count]:   # 7
            "ts-inventory-service.cpu_usage.mean", ...
for kpi in node_kpis:                       # 1 KPI: node_cpu
    for stat in the same 7:
            "node_pool.node_cpu.mean", ...
```

**Width n = 8 × 1 × 7 + 1 × 7 = 63.** The first feature is `ts-inventory-service.cpu_usage.mean` (alphabetical order), and the last is `node_pool.node_cpu.count`.

> Memory was dropped: `scripts/20_audit_and_train_cpu_only.py` compares a 2-KPI Rectifier (cpu + memory = 119 features) with the CPU-only one (63). Memory drift in Kubernetes is non-stationary and was producing false anomalies, so the model became CPU-only.

### 4.2 The 7 statistics, and why these

For the values v₁…vₖ of k pods:

| Statistic | Formula | What it captures |
|---|---|---|
| mean | Σvᵢ / k | overall level |
| min | min vᵢ | a starved or idle pod |
| Q1 | 25th percentile | lower spread |
| median | 50th percentile | typical pod, robust to one outlier |
| Q3 | 75th percentile | upper spread |
| max | max vᵢ | **one hot pod**, which is what a single-pod fault looks like |
| count | k | scaling activity |

With 1 pod, the first 6 are all equal to that pod's value.

### 4.3 Zero-pod services and the moving average (EMA)

A service can briefly have no Ready pods (during a rollout or a crash). Writing 0 would look like "CPU dropped to zero", a fake anomaly. Instead:

- `count` is set to **0**, which is true and informative.
- The other 6 statistics take the service's **exponential moving average**:

```
ema_f  ←  α · x_f  +  (1 − α) · ema_f,       α = 0.2
```

With α = 0.2, the average remembers roughly the last 5 ticks. All features update it every tick, so it's always current. If a whole tick is missing (no rows at all), the entire EMA vector is returned with the "imputed" flag.

### 4.4 Implementation notes

- Uses **Polars** for group-and-filter, and NumPy for the percentiles.
- Pods are mapped to services by **substring**: `"ts-train-service" in "ts-train-service-745476bb9-flvkc"`.
- In our live queries, every row is written with `pod_phase="Running"` and `is_ready=True` (Prometheus CPU rates only exist for running containers), so the Ready filter never removes anything in practice. It matters only if richer pod-state data is supplied.
- The Rectifier keeps state (the EMA) between ticks. The operator therefore creates it **once** and reuses it (`InferenceAdapter.__init__`).

---

## 5. Real example from our data

A real healthy tick from the training data (`2026-09-11T19:53:34`), when the autoscaler had scaled `ts-train-service` and `ts-ui-dashboard` to **3 pods**:

**Input rows (CPU in cores):**
```
ts-train-service-745476bb9-flvkc   0.004627
ts-train-service-745476bb9-zpg7z   0.003994
ts-train-service-745476bb9-46qgz   0.003807
ts-ui-dashboard-...-n498f / 9vnmz / hg5vw   0.006598 / 0.006999 / 0.006405
ts-route-service-5b4f485ffb-h9ss9  0.004504     (one pod)
... one pod each for the other services ...
node                               0.021818     (node_cpu)
```

**Rectifier output for `ts-train-service`** (7 of the 63 numbers):
```
ts-train-service.cpu_usage.mean     0.00414
ts-train-service.cpu_usage.min      0.00381
ts-train-service.cpu_usage.q1       0.00390
ts-train-service.cpu_usage.median   0.00399
ts-train-service.cpu_usage.q3       0.00431
ts-train-service.cpu_usage.max      0.00463
ts-train-service.cpu_usage.count    3
```

**Node part:** all six value statistics are `0.02182` and count is `1` (one node).

Three pods became 7 numbers. One minute later, with 1 pod, it is still 7 numbers, so the vector length stays **63**.

In the healthy training set, `ts-train-service` and `ts-ui-dashboard` reached 3 pods, and all others stayed at 1. The model therefore saw real scaling during training.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Rectifier class | `src/rectifier.py` |
| Training use | `scripts/20_audit_and_train_cpu_only.py` |
| Recording use | `scripts/37_record_runs.py::LiveSignalSource.signals` |
| Live use | `src/inference_adapter.py::run_tick` |

## 7. Limits and known issues

- **Fixed service list.** Adding a service changes the width, so the autoencoder must be retrained.
- **Statistics lose identity.** You can see that *some* pod is hot (max) but not *which* one.
- **One KPI.** A 7-statistic summary of CPU cannot see network delay.
- **The Ready filter is effectively a no-op** with our current queries (see 4.4).

## 8. Questions a reviewer may ask

**Q: Why not just average the pods?**
A: An average hides a single sick pod among healthy replicas. That is exactly what happened when the autoscaler added fresh replicas during the order fault. `max`, `min` and the quartiles keep that signal.

**Q: What is the input size, and how is it computed?**
A: 8 services × 1 KPI × 7 statistics + 1 node KPI × 7 = 63.

**Q: What happens when a service has zero pods?**
A: `count` becomes 0, and the other statistics are filled from an exponential moving average (α = 0.2), so a missing pod doesn't look like CPU dropping to zero.

**Q: Is this your idea or the paper's?**
A: The paper's (PREFACE's Rectifier). We reimplemented it, made it CPU-only, and added EMA imputation for zero-pod services.
