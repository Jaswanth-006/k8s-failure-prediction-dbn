# Stage 03: The Service Graph (Who Calls Whom)

> How do we build the graph of dependencies between services, how is it stored, and where does the rest of the pipeline use it?

---

## 1. In simple words

If service A calls service B, a problem in B can make A look sick too. To blame the right service we need a **map of who calls whom**. Instead of writing that map by hand and possibly getting it wrong, we **read it from the real traffic**. Istio sees every request, so we ask Prometheus: *"for every pair of services, how many requests went from one to the other?"* Every pair with requests becomes an arrow in the graph.

## 2. Why this stage exists

- **Root-cause analysis** (Stage 09) needs to know parents and children: "Is this service suffering because its dependency is broken?"
- **The DBN** (Stage 07) has a slot for letting a sick parent raise the chance that its child gets sick.
- A **wrong graph silently gives wrong answers**. In the earlier simulated evaluation, a hand-drawn "star" graph (dashboard → everyone) made all 28 wrong root-cause answers blame the dashboard. With the discovered graph, the same simulated signals scored 100% (5 runs).

---

## 3. Step by step

```
Step 1  Traffic flows (loadgen → dashboard → ...) for a while, so Istio has counted requests
Step 2  python scripts/26_discover_service_graph.py
          - PromQL: sum by (source_workload, destination_workload)(istio_requests_total{reporter="source"})
          - keep pairs where both ends are among our 8 services, and source != destination
          - add a directed edge source → destination with request_count
Step 3  Save to data/experiments/discovered_service_graph.json
Step 4  Every consumer loads that JSON into a NetworkX DiGraph:
          - evaluator / calibrator: scripts/36_evaluate_goal6.py::graph_for_services
          - live operator:           src/inference_adapter.py::load_service_graph
Step 5  Check it is a DAG (no cycles); if not, fall back / condense
Step 6  DBN orders services topologically; RCA walks predecessors/successors
```

---

## 4. Technical depth

### 4.1 The discovery query

```promql
sum by (source_workload, destination_workload) (
    istio_requests_total{reporter="source"}
)
```

- `istio_requests_total` is a **counter** of requests since the sidecar started. No `rate()` is needed, because we only want to know whether an edge exists and roughly how busy it is.
- `reporter="source"` counts each request once, from the caller's sidecar.
- `sum by (source_workload, destination_workload)` collapses pods and response codes into one number per service pair.

Filtering in `discover_service_graph()`:
- Both ends must be in the known 8-service set. This drops `loadgen → ts-ui-dashboard`, Prometheus scrapes, and so on.
- Self-calls are dropped.

### 4.2 The discovered graph

`data/experiments/discovered_service_graph.json`:

| Source → Destination | Requests counted |
|---|---|
| ts-ui-dashboard → ts-train-service | 860 |
| ts-train-service → ts-route-service | 875 |
| ts-ui-dashboard → ts-user-service | 688 |
| ts-ui-dashboard → ts-order-service | 461 |
| ts-payment-service → ts-inventory-service | 436 |
| ts-order-service → ts-payment-service | 429 |
| ts-ui-dashboard → ts-station-service | 221 |

These numbers match the load generator's path weights (train 40%, user 30%, order 20%, station 10%) and the call chains in Stage 01. Discovery reproduced the design exactly: **8 nodes, 7 edges**.

```
ts-ui-dashboard ──► ts-train-service ──► ts-route-service
       │
       ├──► ts-user-service
       ├──► ts-order-service ──► ts-payment-service ──► ts-inventory-service
       └──► ts-station-service
```

### 4.3 Why it must be a DAG

A **DAG** (Directed Acyclic Graph) has arrows but no loops. We need that because:

1. The DBN updates services in **topological order** (parents before children): `nx.topological_sort(G)`. With a loop, there is no "first".
2. The words "upstream" and "downstream" only make sense without loops.

What happens if a cycle appears:
- **Live operator** (`src/inference_adapter.py`): logs a warning and uses `FALLBACK_EDGES`, the same 7 edges written in code, so node names stay intact.
- **Evaluator** (`scripts/36_evaluate_goal6.py`): uses `nx.condensation`, which merges each loop into one node. This never triggered on our data.

If the JSON file is missing, the operator uses `FALLBACK_EDGES` and the evaluator uses a star. Both print a warning.

### 4.4 Where the graph is used

| Consumer | How it uses the graph | Stage |
|---|---|---|
| **DBN transition step** | For each service, find its *worst parent state* in each particle. If a parent is Degrading or Critical, add a "topological modifier" to the child's transition probabilities. | 07 |
| **Root-cause analysis** | *Upstream evidence*: my children are unhealthy and started after me. *Victim evidence*: my parent is unhealthy and started before me. | 09 |
| **Calibration** | Tries to learn the topological modifiers from labelled runs. | 08 |

### 4.5 Graph data structure in code

```python
import networkx as nx
G = nx.DiGraph()
G.add_nodes_from(SERVICES)
G.add_edge("ts-train-service", "ts-route-service")   # train calls route
list(G.predecessors("ts-route-service"))  # ['ts-train-service']   -> parents
list(G.successors("ts-ui-dashboard"))     # user, train, order, station -> children
list(nx.topological_sort(G))              # parents always appear before children
```

---

## 5. Real example: blame on the train-service fault

Train-service fault, second recording. The fault was injected at tick 8:

```
tick   train   dashboard   route    PREFACE (alarm if any > 1.386)
  8     0.00     1.65       1.39    ALARM -> blames ts-ui-dashboard   (fault not visible yet)
  9     9.73     0.74       0.00    (already alarmed)
 10    10.54     0.88       1.17
```

- At tick 8 the stress had just started, and the 2-minute `rate()` window hadn't caught it yet. The dashboard happened to have a noise value of 1.65, above PREFACE's threshold, so **PREFACE alarmed at the injection tick and blamed the dashboard**.
- The DBN did not react to that single value: the dashboard's P(Critical) stayed at 0.00. At tick 9, train's P(Critical) rose to 0.69 (seed 0), and root-cause analysis named **ts-train-service**. It did so on every seed.
- **The graph's part:** at tick 9, `ts-route-service` (which train calls) received *victim evidence* of 2.50, because its parent in the graph was unhealthy. That kept route from being blamed.

So this win came mainly from **temporal belief** (ignoring a one-tick blip). The graph's contribution was marking train's dependency as a victim.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Discovery script | `scripts/26_discover_service_graph.py` |
| Stored graph | `data/experiments/discovered_service_graph.json` (gitignored, regenerate) |
| Loading for evaluation/calibration | `scripts/36_evaluate_goal6.py::graph_for_services` |
| Loading for live operator + fallback edges | `src/inference_adapter.py::load_service_graph`, `FALLBACK_EDGES` |

## 7. Limits and known issues

- **Only observed edges exist.** A dependency that got no traffic during discovery is missing. Our load generator hits every path, so all 7 edges appeared.
- **Static snapshot.** The graph is discovered once, not updated live.
- **Request counts are stored but unused.** The JSON keeps `request_count`, but both loaders add edges without it, so the causal analyzer's request-count "pressure" term is always 0.
- **The graph's influence inside the DBN was not actually learned.** A calibration bug makes the learned topological modifiers all zero (Stage 08, section 7). In the live evaluation, **the graph mattered through root-cause analysis, not through the DBN's transitions.** Be ready to say this if asked.

## 8. Questions a reviewer may ask

**Q: How did you get the dependency graph? Did you draw it?**
A: No. We queried Istio's request counters in Prometheus and made an edge for every pair of our services that exchanged requests. The result matched the deployment configuration exactly.

**Q: Why must the graph be acyclic?**
A: The DBN processes parents before children using a topological order, and "upstream" is undefined in a loop. If a cycle appears, the operator falls back to the known edges.

**Q: How is the graph used in probability?**
A: In the DBN transition step, a service's next-state distribution can be shifted by its worst parent's current state. In root-cause analysis, parents and children decide who is a cause and who is a victim. (In our calibrated runs, the transition shift was zero because of a calibration bug, so the graph acted through root-cause analysis.)

**Q: What if a new service is added?**
A: Re-run the discovery script and add the service name to the service lists. The Rectifier's input width changes with the service count, so the autoencoder would also need retraining.
