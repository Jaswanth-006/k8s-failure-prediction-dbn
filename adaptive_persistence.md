# Walkthrough: Adaptive Persistence (Goal 1)

I have successfully implemented Adaptive Persistence to fix the detection latency issue while preserving the safety of the DBN.

## 1. What was Changed?

### The New Component: `AdaptivePersistence`
I created a clean, testable state machine in `src/adaptive_persistence.py`. It tracks consecutive abnormal ticks per microservice independently. It exposes an `update` function that dynamically determines how many ticks are required based on the severity of the anomaly and the DBN's confidence:
- **HIGH** (Anomaly $\ge$ 8.0, P(Crit) $\ge$ 0.9) $\rightarrow$ Requires 2 ticks
- **MODERATE** (Anomaly $\ge$ 4.0, P(Crit) $\ge$ 0.6) $\rightarrow$ Requires 5 ticks
- **LOW/NORMAL** $\rightarrow$ Requires 11 ticks

### Integration into the DBN (`src/ddn_core_phase3.py`)
I integrated the new component seamlessly into the Particle Filter logic. 
- I added a `persistence_mode` flag (defaulting to `"fixed"` for backward compatibility).
- In `"adaptive"` mode, the DBN evaluates `AdaptivePersistence` for every service *before* root cause localization.
- **Critical Fix:** `_localize_root_cause` was updated so that it only traverses the graph to find the root cause among **confirmed failures** instead of blindly looking at `P(Critical) > 0.4`.

### Decision Policy Update (`src/decision_policy.py`)
- Added the `persistence_mode` flag.
- When running in `"adaptive"` mode, the Decision Policy skips its rigid 11-tick internal counter. Since the DBN only outputs a root cause if it is a confirmed failure, the policy instantly proceeds to Expected Utility calculations and mitigation.

## 2. Unit Testing
I created `tests/test_adaptive_persistence.py` mapping directly to the requested test cases.
- **TEST A**: Normal system resets properly.
- **TEST B**: One-tick spike resets properly.
- **TEST C**: Severe failure confirms in 2 ticks.
- **TEST D**: Moderate failure confirms in 5 ticks.
- **TEST E**: Recovery resets the counter.
**Result:** All 5 tests successfully passed execution.

## 3. Experimentation & Missing Instrumentation
I wrote the full benchmarking script (`scripts/34_compare_adaptive_vs_fixed.py`) to run the `pilot_cpu_01` data head-to-head.

> [!WARNING]
> **Missing Instrumentation / Data Unavailable**
> I attempted to execute the experiment script in the background, but it failed with a `requests.exceptions.ConnectTimeout`. The script requires an active connection to the local Prometheus server (`localhost:9090`) to fetch historical per-pod telemetry and pipe it through the Rectifier. Because the Kubernetes cluster/Prometheus server is currently offline in this environment, the raw evaluation data cannot be reconstructed. 

As per the safety instructions, I am not fabricating metrics for this run.

## 4. How to Use
- **To use Fixed Mode (Legacy):** Instantiate `DynamicDecisionNetworkPhase3` and `DecisionPolicy` with `persistence_mode="fixed"` (or just rely on the defaults).
- **To use Adaptive Mode:** Pass `persistence_mode="adaptive"` to both classes.
- **To run the unit tests:** Execute `python -m unittest tests.test_adaptive_persistence`
- **To run the experiment (when the cluster is up):** Execute `python scripts/34_compare_adaptive_vs_fixed.py`
