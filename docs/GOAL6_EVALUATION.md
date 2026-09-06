# Goal 6 Evaluation: PREFACE-DBN Full System

This document provides a comprehensive evaluation of the final PREFACE-DBN system (Goal 6). We conducted a controlled experiment to measure key operational and statistical metrics to demonstrate the capabilities built in Goals 1 through 5.

## Evaluation Methodology

The evaluation uses a mathematically strict, scientifically clean testing environment. The core of this methodology is the strict separation between the calibration phase (learning the model parameters) and the evaluation phase (testing the model), with no artificial data manipulation.

### Parameter Calibration (Goal 5)
First, the `DynamicDecisionNetworkPhase3` is calibrated using parameters learned in Goal 5. We generate `2,000` ticks of synthetic historical telemetry (seeded with `42` for reproducibility) to train the `DBNParameterLearner`. This yields the empirical:
- Transition Matrix
- Observation Means and Standard Deviations
- Topological Modifiers

### Evaluation Phase (Goal 6)
Using the dynamically loaded Goal 5 parameters, we evaluate the system against `N = 100` distinct, mutually exclusive mock experiments. This evaluation dataset is generated completely independently (seeded with `1337`).

The confusion matrix sets are strictly partitioned:
1. **50 Negative Trials (Completely Healthy):** No fault is injected. The trial runs normally for 30 ticks. Any alarm triggered is a False Positive (FP). If no alarm triggers, it is a True Negative (TN).
2. **50 Positive Trials (Fault Injected):** A fault is injected at `t_fault = 15` into a randomly selected service (e.g., `ts-train-service`) causing it to emit a `Critical` signal ($\mu \approx 5.0$). The upstream caller (`ts-ui-dashboard`) exhibits a realistic upstream signal propagation, modeled as a `Degrading` signal ($\mu \approx 1.5$). If an alarm triggers at or after the fault, it is a True Positive (TP). If no alarm triggers, it is a False Negative (FN).

All alarms (True Positives, False Positives) and missed detections (False Negatives) emerge organically from the statistical behavior of the calibrated DBN.

## Metrics Formulations

We computed six essential metrics exactly according to standard binary classification definitions:

- **True Positives (TP):** Among Positive trials, the system successfully triggers an alarm (`P(Critical) > threshold`) at or after `t_fault`.
- **False Positives (FP):** Among Negative trials, the system prematurely triggers an alarm.
- **False Negatives (FN):** Among Positive trials, the system fails to trigger any alarm.
- **True Negatives (TN):** Among Negative trials, the system correctly remains quiet.

Given these mutually exclusive counts, we compute:

1. **Detection Latency**: `Average(t_detect - t_fault)` for all TPs.
2. **Precision**: `TP / (TP + FP)`
3. **Recall**: `TP / (TP + FN)`
4. **F1 Score**: `2 * (Precision * Recall) / (Precision + Recall)`
5. **False Positive Rate (FPR)**: `FP / (FP + TN)`
6. **Root Cause Accuracy**: `Sum(predicted_rc == actual_rc) / TP` (Percentage of true positives where the predicted culprit perfectly matches the injected culprit).

## Experiment Results

Over 100 structured experiments using the calibrated Goal 5 parameters, the final system yielded the following results:

| Metric | Result | Interpretation |
|---|---|---|
| Total Experiments | 100 | Number of distinct, independently generated fault timelines (50 positive, 50 negative). |
| True Positives | 50 | 100% success rate in catching injected faults out of 50 positive trials. |
| False Negatives | 0 | No missed faults among the 50 positive trials. |
| False Positives | 0 | 0 false alarms occurred during the 50 healthy negative trials. |
| True Negatives | 50 | The system remained perfectly stable during the 50 healthy negative trials. |
| **Detection Latency** | `1.74` ticks | Detection takes ~1.7 ticks. The calibrated transition matrix applies temporal smoothing, ensuring robustness against noise at the cost of slight latency. |
| **Precision** | `100.0%` | Perfect confidence that an alarm represents a real fault across the balanced dataset. |
| **Recall** | `100.0%` | Perfect coverage of actual faults. |
| **F1 Score** | `100.0%` | Perfect balance between precision and recall. |
| **False Positive Rate** | `0.00%` | Zero rate of alerting during normal operation. |
| **Root Cause Accuracy** | `44.00%` | See breakdown below. |

### Root Cause Breakdown

The Root Cause Accuracy (RCA) is computed explicitly over the 50 detected faults (True Positives). The exact actual vs. predicted breakdown is as follows:

- **Total Injected & Detected Faults (RCA Denominator)**: 50

| Actual Root Cause | Correctly Predicted | Incorrectly Predicted (Blamed UI Dashboard) | Total Injected |
|---|---|---|---|
| `ts-order-service` | **10** | 12 | 22 |
| `ts-route-service` | **6** | 12 | 18 |
| `ts-train-service` | **6** | 4 | 10 |
| **Totals** | **22** | **28** | **50** |

**Total Correct RCA Predictions**: 22 (44.00% Accuracy).

## Limitations and Analysis

1. **Detection vs. Localization**: The perfect detection scores (100% F1, 0% FPR) compared to the moderate localization score (44% RCA) clearly indicates that the PREFACE-DBN system's strength currently lies in robust anomaly **detection**. The system definitively knows *when* a failure is happening (with zero false alarms), but struggles to pinpoint *where* it originated when upstream services exhibit strong degradation noise.
2. **Topological Bleed**: The primary driver behind the 44% RCA is the Goal 5 learned transition matrix combined with the 1.74 tick detection latency. Upstream nodes legitimately experience a `Degrading` state. Over 1-2 ticks, the empirical transition matrix allows this `Degrading` state to slowly bleed probability into the `Critical` state. Once the upstream node accumulates $>0.4$ $P(Critical)$, the simple topological MAP localizer incorrectly halts at the highest node.
3. **Synthetic Evaluation Limitations**: This evaluation is entirely synthetic. While mathematically clean and rigorous, synthetic Gaussian sequences lack the complex structural variance of real-world Kubernetes deployments (e.g., CPU throttle spikes, network jitter, Istio telemetry lag). Therefore, these metrics serve to validate the statistical integrity of the DBN model, but do not establish guaranteed real-world production performance.
