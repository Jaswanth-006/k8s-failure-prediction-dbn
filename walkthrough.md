# Goal 4: Multi-Signal Telemetry Implementation Completed

Goal 4 extends PREFACE-DBN with multi-signal service and dependency telemetry for evidence-rich root-cause analysis.

## Work Accomplished

1. **Fixed Anomaly Saturation**

   - The original autoencoder could produce very large MSE Z-scores during severe faults.
   - Extreme anomaly values could saturate the DBN observation model and make multiple services appear equally critical.
   - **Fix:** Applied `np.log1p(z_score)` scaling in `src/autoencoder_phase3.py`.
   - The DBN observation signal is also clipped at `15.0` in `src/ddn_core_phase3.py`, allowing stronger anomalies to remain distinguishable without destabilizing the particle filter.

2. **Integrated Multi-Signal Telemetry into Causal RCA**

   - Added service-level telemetry:
     - CPU utilization
     - Memory usage
     - Request rate
     - Error rate
   - Added dependency-level telemetry:
     - Request rate
     - Request count
     - Error rate
     - P95 latency

   - Updated `src/causal_rca.py` to accept `ServiceTelemetry` and `EdgeTelemetry`.
   - Upstream causal evidence is gated by dependency health.
   - Victim evidence is also gated by dependency error rate and latency.
   - Physical service telemetry helps distinguish genuine service failures from false positives produced by an abstract anomaly detector.

3. **Experiment Results**

   - Ran `scripts/29_run_goal4_experiment.py`.
   - The RCA model correctly identified `ts-train-service` as the root cause for all **20 ticks** during the FAULT phase.
   - Root-cause accuracy: **100.0%**
   - Required target: **80%**

4. **Testing**

   - All 12 Goal 4 telemetry tests passed.
   - All 6 Goal 3 causality regression tests passed.
   - Goal 2 discovered dependency-graph integration remains functional.
   - Python compilation checks passed for the modified core modules and Goal 4 experiment.

## Verification

The Goal 4 experiment achieved:

- Expected root cause: `ts-train-service`
- Fault ticks: 20
- Correct root-cause ticks: 20
- Incorrect root-cause ticks: 0
- Root-cause accuracy: 100.0%
- Average train-service error rate during FAULT: 52.9%
- Average train-service error rate during HEALTHY: 0.0%

For detailed design information, see:

`docs/GOAL4_MULTI_SIGNAL_TELEMETRY.md`

> Goal 4 adds physical service and dependency evidence to the existing CPU autoencoder + DBN + directional RCA pipeline, improving the system's ability to distinguish root causes from propagated victims and correlated false positives.
