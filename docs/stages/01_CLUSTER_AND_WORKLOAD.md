# Stage 01: The Cluster and the Workload

> What exactly are we monitoring? A local Kubernetes cluster running 8 small cooperating web services, a traffic generator that plays "users", and autoscalers that add pods under load.

---

## 1. In simple words

To predict failures we need something that can fail. We built a miniature online train-ticket shop on a laptop:

- **8 small programs** (microservices), each doing one job: a dashboard, users, trains, routes, orders, payments, inventory and stations.
- They **call each other**. When you ask the dashboard for an order, it asks the order service, which asks payment, which asks inventory.
- A **load generator** acts as a crowd of users, sending 2 to 10 requests every second, rising and falling like a wave.
- **Kubernetes** keeps them all running, and adds extra copies of a busy service automatically.

Everything runs inside Docker on one Windows laptop, using a tool called **kind**.

## 2. Why this stage exists

- The PREFACE paper's key difficulty is **autoscaling**: the number of pods, and therefore the number of measurements, keeps changing. We need a system that really autoscales, or we wouldn't be testing the hard part.
- We need **dependencies between services**, or root-cause analysis would have nothing to reason about.
- We need **steady user traffic**, or there would be no latency to measure and no way to tell when users are affected.

---

## 3. Step by step: how the testbed is built

```
Step 1  Docker Desktop (WSL2 backend, ~11 GB memory given to it)
Step 2  kind create cluster --name preface-dbn          -> 1-node Kubernetes 1.34
Step 3  install metrics-server (+ --kubelet-insecure-tls) -> HPA can read CPU
Step 4  istioctl install --set profile=minimal -y        -> service mesh control plane
        kubectl label namespace default istio-injection=enabled
Step 5  kubectl apply -f phase1-workload.yaml            -> 8 services + 3 HPAs
Step 6  kubectl apply -f manifests/loadgen.yaml          -> synthetic users
Step 7  Prometheus / node-exporter / Chaos Mesh          -> Stages 02 and 06
```

Scripts: `scripts/01_setup_local_cluster.ps1` does steps 2–3, and `scripts/03_deploy_telemetry.sh` does step 4 plus Prometheus.

> `scripts/02_deploy_trainticket_core.sh` is left over from the original plan to deploy the real TrainTicket benchmark with Helm. That chart is not in the repository and the script was **not used**. The real TrainTicket has dozens of Java services, far more memory than a laptop cluster can hold, so we used 8 mock services modelled on it.

---

## 4. Technical depth

### 4.1 The 8 services and who calls whom

All 8 services run **the same small Python program** (stored in a ConfigMap called `mock-app-code`). Each behaves differently because of two environment variables:

- `ROUTES`: "if the URL starts with this prefix, forward the request to that service" (used only by the dashboard).
- `DOWNSTREAM`: "on every request, also call these services".

| Service | Calls | How | HPA (autoscaler) |
|---|---|---|---|
| `ts-ui-dashboard` | train, user, order, station | `ROUTES` by URL prefix | yes, 1–3 replicas |
| `ts-train-service` | route | `DOWNSTREAM` | yes, 1–3 |
| `ts-order-service` | payment | `DOWNSTREAM` | yes, 1–3 |
| `ts-payment-service` | inventory | `DOWNSTREAM` | no |
| `ts-user-service` | nothing | leaf | no |
| `ts-route-service` | nothing | leaf | no |
| `ts-inventory-service` | nothing | leaf | no |
| `ts-station-service` | nothing | leaf | no |

Drawn as a graph (an arrow means "calls"):

```
                         loadgen (users)
                              │
                       ts-ui-dashboard
          ┌──────────┬────────┴─────────┬──────────────┐
          ▼          ▼                  ▼              ▼
   ts-user-service  ts-train-service  ts-order-service  ts-station-service
                     │                  │
                     ▼                  ▼
              ts-route-service   ts-payment-service
                                        │
                                        ▼
                                ts-inventory-service
```

### 4.2 What one service does per request

In `generate_yaml.py` (which produced `phase1-workload.yaml`):

1. If `ROUTES` matches the URL, forward the same request to that service (5 s timeout).
2. For each service in `DOWNSTREAM`, send a `GET /` to it.
3. Burn a little CPU: `for _ in range(5000): x += 1`. This makes CPU rise with traffic, which the HPA and our model can see.
4. Reply with JSON.

These calls are **synchronous**: the caller waits for the callee. So if `ts-route-service` becomes slow, `ts-train-service` becomes slow, and so does the dashboard. This is exactly how a fault spreads to users.

### 4.3 CPU requests, limits and autoscaling

Every service container has:

```yaml
resources:
  requests: { cpu: 10m,  memory: 32Mi }    # what the scheduler reserves
  limits:   { cpu: 150m, memory: 128Mi }   # hard ceiling (cgroup throttling)
```

- **Request = 10m** (1% of a core). The HPA measures utilisation *as a percentage of the request*.
- **HPA target = 50%**, so about **5m of CPU use** is enough to trigger scale-out. That's why the autoscaler reacts within about 2 ticks during a fault.
- **Limit = 150m.** A pod can never use more than 15% of a core. During a fault the stressed pod sits flat at **150m**, because the limit caps Chaos Mesh's stress.

### 4.4 The load generator (`manifests/loadgen.yaml`)

It runs **inside** the cluster, so requests pass through Istio sidecars and get measured.

```python
rps = 6.0 + 4.0 * sin(2π · t / 360 s)      # 2 → 10 requests/second, 6-minute wave
paths = train ×4, user ×3, order ×2, station ×1   # 40% / 30% / 20% / 10%
```

- The wave makes load, and therefore pod counts, **move up and down**, which is the autoscaling condition the project is about.
- The path weights explain a later result: `ts-order-service` receives only **20%** of user requests, which is why its fault was hidden when latency was averaged across the whole mesh (Stage 06).

### 4.5 Istio sidecars

Labelling the namespace with `istio-injection=enabled` makes Kubernetes add an **Envoy proxy container** to every pod. Envoy sees every request in and out, and publishes:

- `istio_requests_total{source_workload, destination_workload, response_code, reporter}`
- `istio_request_duration_milliseconds_bucket{...}`, a latency histogram

We use these for two things: **discovering the service graph** (Stage 03) and **measuring user-facing latency and errors** (Stage 06).

---

## 5. Real example from our data

During the order-service fault in the second recording (Stage 06), the HPA did its job:

```
tick  8   1 replica    user p95 18 ms   <- fault injected
tick 10   3 replicas   user p95 52 ms   <- autoscaler reacted within 2 ticks
ticks 13-16            36 11 30 34 ms   <- extra replicas absorbed some load
ticks 17-22            53 58 42 41 52 68 ms <- fault re-emerged, disruption at tick 19
```

Chaos Mesh stressed only the pod that existed at injection time. The new replicas were healthy, but the stressed pod stayed in the Service's rotation, so a share of requests kept reaching it. **Autoscaling delayed the disruption from 5 to 11 minutes but did not prevent it.**

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Service definitions (generated) | `phase1-workload.yaml` |
| Generator for that YAML | `generate_yaml.py` |
| Load generator | `manifests/loadgen.yaml` |
| Cluster + metrics-server setup | `scripts/01_setup_local_cluster.ps1` |
| Istio + Prometheus setup | `scripts/03_deploy_telemetry.sh` |

## 7. Limits and known issues

- **Mock services**: no database, no business logic. Real failures such as memory leaks, lock contention and slow queries are not reproduced.
- **One node**: node-level failures and rescheduling to another node cannot really be tested.
- **Tiny CPU numbers**: a 10m request and a 150m limit make the autoscaler very sensitive. That's good for demonstrating autoscaling, but not realistic sizing.
- **Leftover script**: `02_deploy_trainticket_core.sh` refers to a Helm chart that does not exist.

## 8. Questions a reviewer may ask

**Q: Why not use the real TrainTicket benchmark like the paper?**
A: It has dozens of Java services and needs far more memory than a laptop kind cluster. We kept its naming and call structure but used 8 lightweight Python services. This is listed as a limitation.

**Q: Why is autoscaling important to the project?**
A: PREFACE's contribution is handling a varying number of pods. Without an HPA the pod count would be constant and the Rectifier would never be tested.

**Q: Why run the load generator inside the cluster?**
A: So that requests pass through Istio sidecars. Traffic sent from outside through a port-forward would bypass the destination metrics we need.

**Q: What does 10m / 150m mean?**
A: 10 millicores are reserved (1% of a core), and 150 millicores are the maximum (15%). The HPA targets 50% of the request, about 5m.

**Q: How does a fault in one service reach users?**
A: Calls are synchronous. A slow `ts-route-service` makes `ts-train-service` wait, which makes the dashboard wait, which is what the user sees.
