# Stage 02: Telemetry (Measuring the Cluster)

> How do we see what the services are doing? Prometheus collects numbers from the cluster. Every 60 seconds we ask it three kinds of question: how much CPU each pod uses, how busy the machine is, and what users are experiencing.

---

## 1. In simple words

Kubernetes and Istio are always producing measurements: CPU used by each container, requests between services, how long each request took. **Prometheus** is a time-series database that collects those measurements every so often and stores them with labels such as `pod="ts-route-service-7d9f..."`.

Once a minute our pipeline asks Prometheus questions written in **PromQL**, for example *"what was each pod's average CPU use over the last 2 minutes?"*. The answers are the raw input to everything else.

We collect two different kinds of numbers, and keeping them separate is important:

| Kind | Examples | Who uses it |
|---|---|---|
| **What the model sees** (causes) | per-pod CPU, node CPU | Rectifier → autoencoder → DBN |
| **What users feel** (effects) | p95 latency and error rate of user requests | Disruption detection, which is the ground truth for evaluation |

The model never sees the user metrics. If it did, it would be "predicting" the failure by looking at the failure itself.

## 2. Why this stage exists

- No measurements means nothing to predict from.
- The measurements must be **identical at training time and at prediction time**. If the autoencoder learned from one query and was later shown another, every anomaly score would be biased by the mismatch rather than by real problems. We fixed exactly this bug (see section 7).

---

## 3. Step by step

```
Step 1  Install Prometheus with Helm (scripts/03_deploy_telemetry.sh)
          - server               : scrapes and stores metrics
          - node-exporter        : machine-level metrics (node CPU)
          - kube-state-metrics   : Kubernetes object state (replica counts)
          - alertmanager, pushgateway disabled (not needed)
Step 2  Istio sidecars (Stage 01) expose request metrics; Prometheus scrapes them
Step 3  Expose Prometheus to the laptop:
          kubectl port-forward -n monitoring svc/prometheus-server 9090:80 --address 127.0.0.1
Step 4  Every tick, our Python code sends HTTP GET /api/v1/query?query=<PromQL>
          to http://127.0.0.1:9090 and parses the JSON answer
```

---

## 4. Technical depth

### 4.1 Where each metric comes from

| Metric | Produced by | Meaning |
|---|---|---|
| `container_cpu_usage_seconds_total` | cAdvisor (inside the kubelet) | Total CPU-seconds the container has used since it started. It is a **counter** that only goes up. |
| `node_cpu_seconds_total{mode="idle"}` | node-exporter | CPU-seconds the machine spent idle. |
| `istio_requests_total` | Envoy sidecars | Count of requests, labelled with source, destination and response code. |
| `istio_request_duration_milliseconds_bucket` | Envoy sidecars | Latency histogram buckets. |

### 4.2 Counters and `rate()`

A counter such as `container_cpu_usage_seconds_total` is not useful on its own, because it just keeps growing. `rate(x[2m])` computes **how fast it grew per second, averaged over the last 2 minutes**. For CPU-seconds, "CPU-seconds per second" is simply **cores in use**. So `0.15` means 150 millicores.

Why a **2-minute window** with **60-second ticks**? The window must contain several samples to be stable, but it shouldn't be so long that consecutive ticks mostly share the same data. Early experiments in the repo used about 5 s ticks with a 2 m window, so consecutive ticks were nearly identical. We moved to 60 s ticks, as in the paper.

### 4.3 The exact queries

**Per-pod CPU** (model input). It is the same string in `scripts/40_collect_healthy.py`, `scripts/37_record_runs.py` and `src/inference_adapter.py`:

```promql
sum(rate(container_cpu_usage_seconds_total{container!="",container!="POD"}[2m])) by (pod)
```

- `container!=""` drops the pod-level total row (it would double count).
- `container!="POD"` drops the "pause" sandbox container Kubernetes adds to every pod.
- `by (pod)` gives one number per pod. The pod name contains the service name (`ts-route-service-5c7...`), which is how we map pods to services.

**Node CPU** (model input):

```promql
1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m]))
```

Idle fraction subtracted from 1 gives the busy fraction of the whole machine, typically about 0.02 on our cluster.

**User-facing metrics** (ground truth only), measured **only on requests entering the system**:

```promql
ENTRY = reporter="destination", destination_workload="ts-ui-dashboard", source_workload="loadgen"

p95_latency_ms = histogram_quantile(0.95,
                   sum(rate(istio_request_duration_milliseconds_bucket{ENTRY}[2m])) by (le))

error_rate     = sum(rate(istio_requests_total{ENTRY, response_code=~"5.."}[2m]))
                 / clamp_min(sum(rate(istio_requests_total{ENTRY}[2m])), 0.001)

request_rate   = sum(rate(istio_requests_total{ENTRY}[2m]))
```

- `histogram_quantile(0.95, ...)` estimates the 95th percentile from the histogram buckets (`le` = "less than or equal" bucket boundary).
- `response_code=~"5.."` matches server errors 500–599.
- `clamp_min(..., 0.001)` avoids dividing by zero when there is no traffic.
- `reporter="destination"` counts each request once, as seen by the receiving sidecar.

### 4.4 Why only the entry point for user metrics?

The dashboard waits for its downstream calls, so **its latency already includes every hop a user's request touches**. An earlier version averaged latency over *every* edge in the mesh. When `ts-order-service` (20% of user traffic) slowed down, the slow requests were mixed with many fast internal calls such as payment→inventory and train→route. The slow fraction fell under 5%, so p95 didn't move, and the fault looked harmless. Replayed from Prometheus:

```
p95 (ms) after the order-service fault was injected
global (all edges)   14   9  18  27  23  24  21  10  10  23   <- flat, fault invisible
entry (user-facing)  21  10  42  59  51  58  40  19  16  48   <- rises 2-3x
```

### 4.5 What one query answer looks like

```json
{"status": "success",
 "data": {"resultType": "vector",
          "result": [
            {"metric": {"pod": "ts-route-service-6b8c9d-x2k4p"}, "value": [1726400000.123, "0.1498"]},
            {"metric": {"pod": "ts-train-service-5f7d8c-q9z1m"}, "value": [1726400000.123, "0.0031"]}
          ]}}
```

`src/inference_adapter.py::_query_prometheus` turns this into rows of `timestamp, pod_name, service_name, pod_phase, is_ready, kpi_name, value`, the input format of the Rectifier (Stage 04). Pods whose names contain none of the 8 service names (Prometheus itself, Istio, loadgen) are dropped.

---

## 5. Real example from our data

Route-service fault, second recording. Model input and user effect side by side:

```
tick   route anomaly signal (from its CPU)   user p95 (entry point)
  7    0.00                                  22 ms
  8    0.00   <- fault injected this tick    10 ms
  9    10.66                                 76 ms
 10    11.55                                 93 ms
 11    11.24                                 95 ms
```

- **Healthy CPU:** the route pod's healthy median CPU in the training data was **0.00375 cores (3.75m)**.
- **Under stress:** a stressed pod is pinned at its **150m limit**. The per-pod trace recorded for the order-service fault shows exactly that: 1–2m before, then 22, 95, and flat at 150m.
- **One tick of lag:** tick 8 still looks healthy because the 2-minute `rate()` window hasn't caught the stress yet.
- **User effect:** at tick 9 the CPU signal jumps and user latency more than triples.

Stage 06 shows how we decide, statistically, when that counts as a disruption.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Prometheus install (node-exporter, kube-state-metrics on) | `scripts/03_deploy_telemetry.sh` |
| Healthy training collector | `scripts/40_collect_healthy.py` |
| Recorder (model + user metrics) | `scripts/37_record_runs.py` (`cpu_query`, `WORKLOAD_QUERIES`) |
| Live operator queries | `src/inference_adapter.py` (`cpu_query`, `node_cpu_query`) |

## 7. Limits and known issues

- **CPU only.** The model uses one KPI. Memory, network and error rate are collected elsewhere (Goal 4 code) but not used by the live pipeline. A network-delay fault would be largely invisible to it.
- **Past bugs we fixed** (worth mentioning, since they show why consistency matters):
  - The operator queried with `localhost`, which on Windows tries IPv6 `::1` first. The port-forward listens only on IPv4, so every query waited about 2 s. We switched to `127.0.0.1`, and a tick went from 4.2 s to 0.08 s.
  - `node_cpu` used to be a hardcoded `0.1` at inference while training used the real value (about 0.02). node-exporter is now installed and the real value is used everywhere. The operator raises an error instead of inventing a value if it's missing.
  - The operator's CPU query used to include the `POD` sandbox container, unlike the training query.
- **Replays are approximate.** Prometheus range queries land on step boundaries, so replayed history can be off by a tick. Headline numbers come from live capture.

## 8. Questions a reviewer may ask

**Q: Why use `rate()` instead of raw CPU values?**
A: The raw metric is a cumulative counter. `rate()` converts it into cores in use, averaged over the window.

**Q: Why doesn't the model get latency as an input?**
A: Latency is how we define the failure (the disruption). Feeding it to the model would let it detect the failure by looking at the failure, which isn't prediction.

**Q: Why the 95th percentile rather than the average?**
A: Users notice the slow tail. An average can stay flat while 1 in 20 requests becomes very slow.

**Q: What happens if Prometheus is down during a live tick?**
A: The operator catches the error, writes it to `status.health.lastError`, and leaves the DBN's belief untouched. Missing data isn't treated as "healthy" (Stage 11).
