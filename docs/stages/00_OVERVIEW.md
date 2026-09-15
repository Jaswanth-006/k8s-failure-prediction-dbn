# Stage 00: The Big Picture

> Start here. This page explains the whole project in plain words, defines every term the other pages use, and shows how the stages fit together.

---

## 1. The project in one paragraph

We run a small online application made of 8 cooperating programs ("microservices") on Kubernetes. Every minute we measure how hard each program is working. A neural network learns what "healthy" looks like, and each minute it reports how unusual each program looks. A **Dynamic Bayesian Network (DBN)** turns those scores into probabilities: *how likely is it that this service is Normal, Degrading or Critical right now?* When one service becomes very likely Critical, the system works out **which service started the problem** and **which repair action is worth the most**. A Kubernetes **operator** then applies that decision, safely. The goal is to catch a failure **before users notice it**.

## 2. An everyday analogy

Think of a hospital ward with 8 patients, some of whom depend on each other.

| Hospital | Our project |
|---|---|
| Patients | The 8 microservices |
| Heart-rate and blood-pressure monitors | Prometheus collecting CPU numbers every minute |
| A nurse who knows each patient's normal readings | The autoencoder, trained on healthy data |
| "This reading looks unusual" | The anomaly signal |
| A doctor who considers the patient's history, not just one reading | The DBN, which carries its belief from minute to minute |
| "Patient B is sick *because* patient A infected them" | Root-cause analysis over the service graph |
| Choosing a treatment by weighing its benefit against its risk | The decision policy (expected utility) |
| A senior doctor who approves before anyone acts | The operator's safety gates (shadow mode) |
| The patient actually collapsing | A "disruption": users see slow or failed requests |

The original research method, PREFACE, is like a nurse who raises an alarm whenever a single reading crosses a fixed line. Our addition is the doctor: it remembers the past, knows who depends on whom, and decides what to do.

---

## 3. Glossary (read once, refer back often)

### Kubernetes words
| Term | Meaning in plain words |
|---|---|
| **Container** | A packaged program together with everything it needs to run. |
| **Pod** | The smallest unit Kubernetes runs: one or more containers sharing a network address. |
| **Node** | A machine (real or virtual) that runs pods. We have exactly one. |
| **Deployment** | An instruction such as "always keep N copies of this pod running". |
| **Replica** | One copy of a pod. |
| **Service** | A stable name, such as `ts-route-service`, that forwards traffic to whichever pods are currently alive. |
| **HPA (Horizontal Pod Autoscaler)** | Adds or removes replicas automatically, based on CPU use. |
| **Namespace** | A folder-like grouping of resources. Our app lives in `default`. |
| **kind** | "Kubernetes IN Docker": a whole Kubernetes cluster running inside a Docker container on a laptop. |
| **CPU millicores (m)** | 1000m = one CPU core; `150m` = 15% of one core. |
| **CRD (Custom Resource Definition)** | A way to teach Kubernetes a new kind of object. We define `FailurePredictor`. |
| **Operator** | A program that watches custom objects and acts on the cluster. Ours is written with the Kopf library. |

### Monitoring words
| Term | Meaning |
|---|---|
| **Prometheus** | A database that collects numbers ("metrics") from the cluster every few seconds. |
| **PromQL** | Prometheus's query language, e.g. `rate(...[2m])`. |
| **cAdvisor** | Part of Kubernetes that reports each container's CPU and memory use. |
| **node-exporter** | Reports whole-machine metrics, such as total CPU busy. |
| **Istio / Envoy sidecar** | A small proxy placed next to every pod. It sees every request, so it can report who called whom, how long it took, and whether it failed. |
| **p95 latency** | 95% of requests were faster than this. It is a measure of the slow tail users notice. |
| **Tick** | One pipeline step. We use one tick every **60 seconds**. |

### Machine-learning and probability words
| Term | Meaning |
|---|---|
| **Autoencoder** | A neural network trained to copy its input through a narrow middle layer. It copies "normal" data well and unusual data badly. |
| **Reconstruction error** | How badly the autoencoder copied the input. High error means an unusual input. |
| **z-score** | "How many standard deviations above normal". z = 3 means 3σ above the healthy average. |
| **Anomaly signal** | Our per-service score, `log1p(z)`. 0 is normal; 10 or more is extremely unusual. |
| **Hidden state** | Something we cannot measure directly, such as the true health of a service (Normal / Degrading / Critical). |
| **Transition probability** | The chance of moving from one hidden state to another in one minute. |
| **Emission probability** | The chance of seeing a particular anomaly signal given the hidden state. |
| **Prior / posterior** | Belief *before* / *after* looking at the new evidence. |
| **Particle filter** | Tracks a belief by simulating many guesses ("particles") and keeping the ones that best match the evidence. |
| **Root cause** | The service where the problem started, as opposed to services that suffer because they depend on it. |
| **Expected utility (EU)** | The average benefit of an action, weighted by how likely each state is. |
| **Debounce** | Require a condition to hold for several ticks before acting, to ignore one-off blips. |
| **Shadow mode** | The system decides and logs "I *would* do X" but does not touch the cluster. |

### Evaluation words
| Term | Meaning |
|---|---|
| **Fault injection** | Deliberately breaking something (here: CPU stress) to test the system. |
| **Chaos Mesh** | A Kubernetes tool for injecting faults. |
| **Disruption** | The moment users are measurably affected (p95 latency or errors clearly worse). |
| **Error interval** | Time from fault injection until the disruption. This is the window in which prediction is useful. |
| **Warning time (earliness)** | Time from our alarm until the disruption. Positive means we warned before users were hurt. |
| **False alarm** | An alarm during a run where nothing was broken. |
| **Seed** | The starting number for random choices. The particle filter is random, so we repeat with 8 seeds. |

---

## 4. Two ways to look at the pipeline

### View A: the build workflow (what we did once, in order)

```
 1. Create the cluster and deploy the 8 services + load generator      -> Stage 01
 2. Install Prometheus, node-exporter, Istio                           -> Stage 02
 3. Discover who-calls-whom from Istio traffic                         -> Stage 03
 4. Collect 25 minutes (100 samples) of HEALTHY telemetry             -> Stage 04/05
 5. Train the autoencoder on it                                        -> Stage 05
 6. Record 3 healthy + 3 fault runs (Chaos Mesh), find disruptions     -> Stage 06
 7. Label those runs and calibrate the DBN's probabilities             -> Stage 08
 8. Replay the runs through PREFACE and PREFACE-DBN, compare           -> Stage 12
 9. Run the operator live and inject a fault while it watches          -> Stage 11
```

### View B: what happens every 60 seconds when the system is live

```
  Kubernetes cluster (8 services, autoscaling, traffic)
          │  metrics every few seconds
          ▼
  ┌──────────────┐   PromQL: per-pod CPU + node CPU
  │  Prometheus  │─────────────────────────────────────┐
  └──────────────┘                                     │
                                                       ▼
  Stage 04  RECTIFIER      pods vary (1..3 per service) -> fixed vector of 63 numbers
                                                       │
  Stage 05  AUTOENCODER    63 numbers -> 8 anomaly signals (one per service)
                                                       │
  Stage 07  DBN            8 signals + memory of past -> P(Normal/Degrading/Critical) per service
                                                       │
  Stage 09  ROOT CAUSE     probabilities + service graph -> "ts-route-service started it"
                                                       │
  Stage 10  DECISION       expected utility + debounce + threshold + cooldown -> "Reschedule_Pod"
                                                       │
  Stage 11  OPERATOR       two safety gates -> act, or log WOULD_EXECUTE; publish status
```

---

## 5. Stage map

| # | File | Question this stage answers | Main code |
|---|---|---|---|
| 00 | this page | What is the whole thing? | |
| 01 | [01_CLUSTER_AND_WORKLOAD.md](01_CLUSTER_AND_WORKLOAD.md) | What are we monitoring, and how is it built? | `phase1-workload.yaml`, `manifests/loadgen.yaml` |
| 02 | [02_TELEMETRY.md](02_TELEMETRY.md) | How do we measure it? | `scripts/03_deploy_telemetry.sh`, PromQL in `src/inference_adapter.py` |
| 03 | [03_SERVICE_GRAPH.md](03_SERVICE_GRAPH.md) | Who calls whom, and how does the graph get built? | `scripts/26_discover_service_graph.py` |
| 04 | [04_RECTIFIER.md](04_RECTIFIER.md) | How do we turn a changing number of pods into a fixed input? | `src/rectifier.py` |
| 05 | [05_AUTOENCODER.md](05_AUTOENCODER.md) | How do we score "how unusual" each service looks? | `src/autoencoder_phase3.py`, `scripts/40`, `scripts/20` |
| 06 | [06_FAULTS_RECORDING_DISRUPTION.md](06_FAULTS_RECORDING_DISRUPTION.md) | How do we create failures, record them, and know when users were hurt? | `scripts/37_record_runs.py`, `src/disruption.py` |
| 07 | [07_DBN.md](07_DBN.md) | How does the DBN compute probabilities? | `src/ddn_core_phase3.py` |
| 08 | [08_CALIBRATION.md](08_CALIBRATION.md) | Where do the DBN's numbers come from? | `src/weak_labels.py`, `src/dbn_learner.py`, `scripts/38` |
| 09 | [09_ROOT_CAUSE.md](09_ROOT_CAUSE.md) | Which service started it? | `src/causal_rca.py` |
| 10 | [10_DECISION_POLICY.md](10_DECISION_POLICY.md) | What action, and when? | `src/decision_policy.py`, utility matrix |
| 11 | [11_OPERATOR_AND_SAFETY.md](11_OPERATOR_AND_SAFETY.md) | How does it run live, safely? | `src/operator.py`, `src/action_executor.py` |
| 12 | [12_EVALUATION_AND_RESULTS.md](12_EVALUATION_AND_RESULTS.md) | Does it work, and how do we know? | `scripts/41`, `scripts/39`, `scripts/42` |
| 13 | [13_REVIEW_QUESTIONS.md](13_REVIEW_QUESTIONS.md) | What will the panel ask? | |

---

## 6. What PREFACE did, and what we added

**PREFACE** (Denaro et al., FSE 2024) contributed two ideas we keep:
1. The **Rectifier**: summarise a varying number of pods into fixed statistics, so a neural network can read an autoscaling system.
2. **An autoencoder trained only on healthy data**: high reconstruction error means "something is off".

PREFACE then decides with a **memoryless rule**: alarm if the error is more than 3 standard deviations above normal, and blame the service with the highest score.

**PREFACE-DBN** (this project) adds:

| Addition | Why it matters |
|---|---|
| **DBN with hidden health states** | Remembers the past, so a one-minute spike doesn't trigger an alarm; a sustained problem does. |
| **Service-graph-aware root cause** | Separates "started the problem" from "suffering because of a dependency". |
| **Calibration from recorded faults (weak supervision)** | The probabilities come from our cluster's real behaviour, not guesses. |
| **Disruption detection with the paper's statistics** | Lets us measure warning time honestly. |
| **Expected-utility decisions + Kubernetes operator** | Turns a prediction into a safe, gated action. PREFACE only raised alarms. |

## 7. The headline result

On live recordings (6 healthy runs, 6 CPU-fault runs across two recordings):

| | PREFACE (its own 3σ rule) | PREFACE-DBN |
|---|---|---|
| Faults detected (second recording) | 3/3 | 3/3 on every seed |
| Healthy runs with a false alarm (both recordings) | **6/6** | **0/6** on every seed |
| Correct root cause (second recording) | 2/3 | 3/3 on every seed |
| Warning before users were affected | not meaningful (always alarming) | 3.5 ± 0.5 minutes |

The operator, run live in shadow mode, passed all 9 end-to-end checks. It acted too late (10.9 minutes, against a 5-minute disruption) because of its 11-tick debounce. Details are in Stage 12.

## 8. How to read these pages

Every stage page has the same layout:

1. **In simple words**: no jargon.
2. **Why this stage exists**: the problem it solves.
3. **Step by step**: the workflow in order.
4. **Technical depth**: formulas, code and parameters.
5. **Real example from our data**: actual numbers from the recorded runs.
6. **Where it lives in the code**.
7. **Limits and known issues**: be ready to discuss these honestly.
8. **Questions a reviewer may ask**.

If you only have one hour, read 00, 07, 10 and 12, then 13.
