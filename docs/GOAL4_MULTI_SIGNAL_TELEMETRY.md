# GOAL 4: Multi-Signal Service and Dependency Telemetry for Evidence-Rich RCA

## 1. Goal 4 Objective
Extend the existing Goal 3 directional causal RCA system with principled, multi-signal telemetry at both the service level (CPU, Memory, Request Rate, Error Rate) and edge level (Traffic, Error Rate, P95 Latency) to enable evidence-rich Root Cause Analysis (RCA).

## 2. Existing Architecture (Goal 3)
In Goal 3, RCA was based entirely on the autoencoder anomaly score (driven solely by CPU metrics) and the topological precedence in the call graph. While effective for simple faults, it lacked the ability to distinguish whether a downstream failure was actually caused by an upstream failure beyond temporal correlation.

## 3. Goal 4 Architecture
Goal 4 introduces a parallel data ingestion track:
- **PrometheusTelemetryCollector** pulls multi-signal metrics.
- **TelemetrySnapshot** normalizes and structures the signals into `ServiceTelemetry` and `EdgeTelemetry`.
- **Autoencoder Anomaly Pipeline** continues to produce baseline signals but uses a principled `log1p` transformation to prevent extreme outliers from collapsing the particle filter.
- **DirectionalCausalAnalyzer** is enriched to heavily gate both upstream propagation and victim attribution based on the explicit health of the dependency edges.

## 4. Service Telemetry
Captures the intrinsic health of a service:
- `cpu_rate`
- `memory_bytes`
- `request_rate`
- `error_rate`

## 5. Edge Telemetry
Captures the health of a dependency path between two services:
- `request_rate`
- `request_count` (used for traffic weighting)
- `error_rate`
- `latency_p95`
This distinction is critical: latency and downstream errors belong to the edge, not intrinsically to the source service.

## 6. Telemetry Collection
Implemented in `src/telemetry_collector.py`. The collector dynamically queries Prometheus for CPU, Memory, Request Rates, Error Rates, and Latency Histograms. It degrades gracefully if latency metrics are absent or traffic is zero.

## 7. Normalization
Raw telemetry is clipped to physically meaningful boundaries (e.g., CPU bounded to 1.0 cores, Latency bounded to 10s) and Min-Max scaled to `[0, 1]` using `telemetry_schema.py`. This avoids historical statistical scaling issues.

## 8. Anomaly Calibration (CRITICAL ISSUE #1 FIXED)
The DBN observation space expects standard Normal to Critical signals (`0.0` to `5.0`). Previously, extreme anomalies (MSE producing Z-scores in the thousands) would all clip to `10.0`, destroying the relative severity of root cause vs. victim. This was fixed by applying `np.log1p(z_score)` in `autoencoder_phase3.py` and expanding the DBN clip bounds, allowing extreme faults to map gracefully into `[0, 15]`.

## 9. DBN Integration
The DBN continues to estimate `P(Normal)`, `P(Degrading)`, and `P(Critical)` for each service using a particle filter. It remains fully backward compatible and simply passes the enriched edge/service telemetry directly to the Causal Analyzer.

## 10. Causal RCA
The Root Cause score is defined as:
`RC_Score = IntrinsicEvidence + UpstreamCausalEvidence - VictimEvidence`
Service telemetry enriches intrinsic evidence (e.g., high memory/errors). Edge telemetry actively *gates* the causal evidence (e.g., a perfectly healthy edge cannot transmit victim status to a downstream service).

## 11. Temporal Evidence
Temporal ordering remains a prerequisite. A parent must degrade before or at the same time as a child to be considered its root cause.

## 12. Root Cause vs Victim
- **ROOT_CAUSE**: High intrinsic degradation and evidence of downstream propagation.
- **PROPAGATED_VICTIM**: Degraded, but heavily discounted by VictimEvidence (an upstream parent failed first, and the edge from parent->child is unhealthy).

## 13. Dependency Pressure
Edge stress (`error_rate + latency + traffic_rate`) drives Dependency Pressure. High pressure increases the causal evidence assigned to the upstream parent and the victim status assigned to the downstream child.

## 14. Intervention Integration
The computed `ROOT_CAUSE` feeds directly into the Goal 3 Expected Utility (MEU) decision policy and `KubernetesActionController`, preserving shadow-mode operations.

## 15. Healthy/Fault/Recovery Experiment
`scripts/29_run_goal4_experiment.py` executes a full synthetic telemetry simulation:
- **HEALTHY**: No RCA actions triggered.
- **FAULT**: `ts-train-service` experiences severe CPU and error anomalies. `ts-route-service` degrades consequently. The RCA correctly identifies `ts-train-service` as `ROOT_CAUSE` and `ts-route-service` as `PROPAGATED_VICTIM`.
- **RECOVERY**: Anomalies naturally fade, and confidence scores return to Normal.

## 16. Tests
- 12 deterministic unit tests added in `scripts/30_test_goal4_telemetry.py`.
- 6 backward-compatibility tests in `scripts/27_test_goal3_causality.py`.

## 17. Results
The experiment achieved **100% root-cause accuracy** across 20 Fault-phase ticks, successfully avoiding the DBN saturation collapse and reliably pinpointing `ts-train-service` despite the massive anomaly generated by the fault.

## 18. Limitations
- Does not automatically discover new metrics beyond the predefined PromQL templates.
- Requires Istio telemetry for edge metrics (relies on envoy proxy headers).

## 19. Backward Compatibility
The system is fully backward compatible. If `edge_telemetry=None`, it falls back to the static `request_count` graph edge attributes used in Goal 3.

## 20. Future Improvements
- Extend edge telemetry to track TCP-level metrics (e.g., connection resets) for databases without HTTP wrappers.
- Dynamically learn telemetry normalization bounds over a 7-day rolling window rather than static physical caps.
