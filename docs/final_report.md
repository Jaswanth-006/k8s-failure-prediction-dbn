# PREFACE-DBN Final Research Report

## 1. Abstract
Microservice architectures are prone to complex, cascading failures. While existing research (PREFACE) demonstrated that failures can be predicted by observing anomalies in cluster telemetry during an "error interval" before disruption, it relied on a memoryless binary threshold. PREFACE-DBN extends this framework by injecting a Dynamic Bayesian Network (DBN) to track failure propagation over time, and a Maximum Expected Utility (MEU) Kubernetes Operator to execute safe, autonomous mitigation. Our head-to-head backtest on a rapid CPU-stress fault revealed a critical trade-off: while temporal debouncing in PREFACE-DBN ensures safety and prevents thrashing, it delays intervention, allowing the original instantaneous baseline to achieve faster and more accurate root cause localization in high-signal scenarios.

## 2. Problem Statement
Cascading failures in microservices often propagate downstream, causing proxy services to exhibit severe symptoms while the causal root service appears stable or masked. By the time static threshold alarms trigger, human operators are forced to react to disruptive failures rather than pre-empting them.

## 3. Research Motivation
The original PREFACE approach proved that anomalies can be detected early via Deep Autoencoders. However, its binary `m_e + 3s_e` threshold and instantaneous Z-score localization ignored the topological reality of the service call graph. We hypothesized that filtering these anomalies through a Dynamic Bayesian Network aligned with the call graph would improve localization accuracy and yield a calibrated probability of disruption `P(H_t = Critical)` that an autonomous mitigation controller could trust.

## 4. System Architecture
The system consists of a robust pipeline:
1. **Telemetry & Rectifier**: Polling Prometheus for pod KPIs and imputing missing data.
2. **Autoencoder**: Computing per-service reconstruction error Z-scores (`a_t^s`).
3. **Dynamic Bayesian Network**: Tracking `P(H_t^s | a_{1:t})` over time.
4. **Decision Policy**: Filtering noise via an 11-tick debounce.
5. **MEU Operator**: Selecting the optimal intervention (`Reschedule_Pod` vs `Restart_Pod`) via utility maximization.

## 5. Telemetry Pipeline
Built on Prometheus and cAdvisor, the pipeline captures standard RED/USE metrics. The Rectifier component processes variable-length pod counts into a fixed-width vector by computing cross-pod statistics (mean, median, IQR, max) per service, ensuring dimensionality consistency for the downstream neural network.

## 6. Autoencoder
A deep symmetric autoencoder (`n -> n/2 -> n/4 -> n/8 -> n/4 -> n/2 -> n`) trained purely on healthy steady-state data. We resolved numerical instabilities in the original implementation by transitioning from standard `mean/std` normalization to robust `median/IQR` normalization, preventing massive outlier blowouts.

## 7. Dynamic Bayesian Network
The DBN models the hidden failure state `H_t^s` (Healthy, Anomaly, Critical) of each service. It uses a transition matrix `P(H_t | H_{t-1})` and conditional probability tables modeling symptom propagation (e.g. `H_t^{child} | H_t^{parent}`). It consumes the continuous `a_t^s` anomaly signals as soft evidence.

## 8. Expected Utility Decision Making
The decision engine does not rely on static IF-THEN rules. Instead, it computes the expected utility:
`A* = argmax EU(A)`
where `EU(A)` considers the utility of restoring the service versus the disruption cost of the action (e.g. Reschedule is less disruptive than Restart).

## 9. Temporal Persistence
To prevent mitigation thrashing caused by the dynamic Kubernetes Horizontal Pod Autoscaler or transient network blips, the system enforces an 11-tick temporal debounce. A service must maintain `P(Critical) > τ` for 11 consecutive polling cycles before action is eligible.

## 10. Kubernetes Operator
The Kopf-based operator acts as the executor. It consumes the `DecisionPolicy` output, executes the K8s API mutation (shadowed), and enforces a 300-second per-root-cause cooldown to allow the system to stabilize post-intervention.

## 11. Safety Architecture
Safety is the paramount requirement for autonomous mitigation.
- **Shadow Mode**: Hardcoded to `True` to prevent physical cluster mutations.
- **Rate Limiting**: Cooldowns prevent infinite loops of restarts.
- **Determinism**: Ties in utility selection default to the least disruptive action (`Do_Nothing`).

## 12. Experimental Methodology
We deployed the TrainTicket microservice benchmark on a local Kubernetes (kind) cluster. Using Chaos Mesh, we injected CPU stress faults into specific microservices. We collected the complete metric windows and fed them into both PREFACE-DBN and a reconstructed original PREFACE baseline.

## 13. Metrics
- **Reaction Interval**: Delay from fault injection to first critical detection.
- **Strong Localization**: True if the *first* service blamed is the actual root cause.
- **Weak Localization**: True if the actual root cause is eventually blamed before disruption.
- **Intervention Delay**: Delay introduced by the safety layer (debounce).

## 14. PREFACE Baseline
The baseline was rigorously reconstructed according to the original specification: a memoryless binary threshold that triggers if the cluster's total autoencoder reconstruction error exceeds `m_e + 3s_e`, and localizes by ranking the instantaneous Z-scores.

## 15. PREFACE vs PREFACE-DBN
On the `pilot_cpu_01` experiment, both systems were fed identical historical Prometheus windows.

## 16. CPU Pilot Results
- **Detection Time**: Identical (t=5.1s). The Autoencoder successfully detected the spike immediately in both systems.
- **Baseline Reaction Interval**: 0.0s (acted instantly on detection).
- **PREFACE-DBN Reaction Interval**: 5.1s (1 tick to transition DBN state to Critical).
- **Intervention Time**: Baseline intervened at 0.0s. DBN intervened at +56.8s due to the 11-tick debounce.

## 17. Localization Failure Analysis
Counterintuitively, the **baseline achieved Strong Localization**, while **PREFACE-DBN failed Strong Localization**.
At the moment of injection, the true root cause (`ts-train-service`) spiked massively. The memoryless baseline instantly detected this and blamed the correct service.
PREFACE-DBN correctly flagged `ts-train-service`, but forced it to wait 11 ticks. By tick 8, backpressure caused the proxy node (`ts-ui-dashboard`) to spike. The DBN, updating its beliefs, eventually let `ts-ui-dashboard` cross the 11-tick threshold first, triggering the 300-second cooldown on the wrong service and blocking the correct mitigation.

## 18. Safety vs Earliness Trade-off
This result perfectly highlights the fundamental tension in AIOps:
Temporal debouncing acts as a crucial safety net against false positives and thrashing. However, *waiting* for certainty gives the fault time to propagate. In fast-acting faults (like CPU stress), propagation outpaces the debounce window, muddying the waters and confusing the root cause localizer.

## 19. Network-Delay Results
*Not Executed. Awaiting broader automated experimental runs.*

## 20. Memory Fault Limitation
Memory experiments (OOM/Leak) were explicitly excluded. Current Kubernetes fault injectors (including Chaos Mesh) allocate memory in user-space containers which are brutally and un-reproducibly OOM-Killed by the kernel without providing a graceful measurable "error interval", rendering them unsuitable for this validation pipeline.

## 21. Risk Calibration
While the DBN computes `P(Critical) = 0.99`, statistical confidence requires extensive multi-run validation to prove that 99% of such predictions actually result in disruption. This remains unproven due to the limited sample size.

## 22. Limitations
1. **Sample Size**: Conclusions rest on limited single-fault runs.
2. **Missing Classes**: Memory and Network delays are absent or unvalidated.
3. **Earliness Calculus**: Lacking locust HTTP percentile logs, true disruptive timestamps couldn't be calculated for earliness bounds.
4. **Shadow Mode**: Actual mitigation was not physically tested against live traffic recovery.
5. **Prometheus Latency**: `rate(...[2m])` queries inherently introduce scraping and aggregation lag.

## 23. Threats to Validity
- **Synthetic Faults**: Chaos Mesh CPU stress (`yes > /dev/null`) is a brutal, instantly-saturating synthetic fault. Real-world degradation is often slower and subtler, where the DBN might drastically outperform the baseline.
- **Workload Independence**: TrainTicket is a specific RPC architecture; results may not generalize to event-driven architectures.

## 24. Future Work
1. **Adaptive Persistence**: Dynamic debounce thresholds (e.g. 2 ticks for CPU, 20 ticks for Network).
2. **Causal Distinction**: Refining the DBN transition matrices to structurally separate "origin" nodes from "proxy/victim" nodes.
3. **Closed-Loop Live Evaluation**: Moving from shadow mode to live intervention to measure actual `Recovery Time Objective (RTO)`.

## 25. Conclusion
PREFACE-DBN successfully demonstrates the theoretical viability of an end-to-end, utility-driven autonomous mitigation pipeline. While the rigorous comparison against the PREFACE baseline on a synthetic CPU fault favored the instantaneous heuristic, this result exposed the critical architectural trade-off between temporal safety (debounce) and diagnostic clarity before propagation occurs. This research paves the way for adaptive-persistence AIOps controllers in Kubernetes.
