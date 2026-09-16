# PREFACE-DBN documentation guide

Start here to find the right document.

| If you want to… | Read |
|---|---|
| **Learn the whole project stage by stage, from plain words to technical depth** | [`stages/00_OVERVIEW.md`](stages/00_OVERVIEW.md) |
| Understand the project, its results and its limits | [`final_report.md`](final_report.md) |
| See every number from the live experiments, with evidence | [`RESULTS_LIVE.md`](RESULTS_LIVE.md) |
| Run the pipeline and see known gaps | [`../README.md`](../README.md) |
| Understand directional root-cause analysis | [`GOAL3_DIRECTIONAL_CAUSALITY.md`](GOAL3_DIRECTIONAL_CAUSALITY.md) |
| Understand multi-signal telemetry | [`GOAL4_MULTI_SIGNAL_TELEMETRY.md`](GOAL4_MULTI_SIGNAL_TELEMETRY.md) |
| See the simulated Goal 6 smoke test | [`GOAL6_EVALUATION.md`](GOAL6_EVALUATION.md) |
| Read the original design and plans | `1_` to `4_` planning documents in the repository root |

## Status of each document

- **`final_report.md` and `RESULTS_LIVE.md`** describe what was built and measured on the live cluster. Where they disagree with anything else, they are the current account.
- **`GOAL6_EVALUATION.md`** describes a simulated evaluation. Its numbers are a smoke test of the code, not system results.
- **The planning documents** describe intent, written before implementation. Some of what they propose was built differently: the DBN is a hand-written NumPy particle filter rather than pgmpy or JAX, and the testbed uses mock services modelled on TrainTicket rather than TrainTicket itself.

## Pipeline at a glance

```
Prometheus / Istio ─> Rectifier ─> autoencoder ─> DBN + causal root cause ─> decision policy ─> operator
      (live cluster)     fixed vector   anomaly signal   P(Normal/Degrading/Critical)    gated action   shadow by default
```

## Scripts, in the order you run them

| Step | Script |
|---|---|
| Collect healthy telemetry | `scripts/40_collect_healthy.py` |
| Train the autoencoder | `scripts/20_audit_and_train_cpu_only.py` |
| Discover the service graph | `scripts/26_discover_service_graph.py` |
| Record healthy and fault runs | `scripts/37_record_runs.py` |
| Calibrate the DBN | `scripts/38_calibrate_from_runs.py` |
| Evaluate | `scripts/36_evaluate_goal6.py` |
| Compare with PREFACE | `scripts/39_compare_baseline.py` |
| Full analysis over seeds | `scripts/41_analyze_rerun.py` |
| Run the operator | `kopf run src/operator.py --standalone` |
