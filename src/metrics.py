from prometheus_client import Gauge, Counter, Histogram

# Gauges (Current State)
preface_risk_score = Gauge(
    'preface_risk_score',
    'Overall anomaly max risk score from DDN'
)
preface_probability_critical = Gauge(
    'preface_probability_critical',
    'P(Critical) for the current root cause'
)
preface_root_cause_probability = Gauge(
    'preface_root_cause_probability',
    'P(Critical) exported by specific service',
    ['service']
)
preface_persistence_ticks = Gauge(
    'preface_persistence_ticks',
    'Current temporal debounce count'
)
preface_intervention_eligible = Gauge(
    'preface_intervention_eligible',
    '1 if an intervention is eligible and ready, 0 otherwise'
)
preface_expected_utility = Gauge(
    'preface_expected_utility',
    'Expected utility mapped by action type',
    ['action']
)
preface_delta_eu = Gauge(
    'preface_delta_eu',
    'Difference between top two primary actions (e.g. Reschedule vs Restart)'
)
preface_cooldown_remaining_seconds = Gauge(
    'preface_cooldown_remaining_seconds',
    'Seconds remaining in the active action cooldown'
)
preface_rate_limit_usage = Gauge(
    'preface_rate_limit_usage',
    'Number of actions consumed in the current rate limit window'
)

# Counters (Cumulative Events)
preface_would_execute_total = Counter(
    'preface_would_execute_total',
    'Cumulative count of shadow mode interventions',
    ['action']
)
preface_executed_total = Counter(
    'preface_executed_total',
    'Cumulative count of real live interventions',
    ['action']
)
preface_blocked_total = Counter(
    'preface_blocked_total',
    'Cumulative count of interventions blocked by safety rails',
    ['reason']
)
preface_ticks_total = Counter(
    'preface_ticks_total',
    'Total inference ticks completed by the operator',
    ['status']
)

# Histograms
preface_inference_duration_seconds = Histogram(
    'preface_inference_duration_seconds',
    'Latency of inference ticks in seconds',
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

def export_decision_metrics(decision: dict, shadow_mode: bool, rate_limit_len: int, elapsed_time: float, tick_status: str = "success"):
    """
    Exports state payload from DecisionPolicy.evaluate into Prometheus metrics
    without mutating the underlying state.
    """
    preface_ticks_total.labels(status=tick_status).inc()
    preface_inference_duration_seconds.observe(elapsed_time)
    
    if tick_status != "success":
        return

    # 1. Risk Metrics
    # In DDN output, risk is P(Critical). If needed, risk_score could be anomaly score, but here P(Crit) is primary.
    preface_probability_critical.set(decision.get("p_crit", 0.0))
    # For now map risk_score to p_crit as well since Phase 3 doesn't pass raw anomaly max upwards
    preface_risk_score.set(decision.get("p_crit", 0.0)) 
    
    # Root Cause Service
    root_cause = decision.get("root_cause", "None")
    # Reset all bounded labels to 0.0 first to avoid stale data
    # (In production with bounded services, we would zero all expected services. 
    # For this demo, we can just clear the gauge and set the new one).
    preface_root_cause_probability._metrics.clear() 
    if root_cause != "None":
        preface_root_cause_probability.labels(service=root_cause).set(decision.get("p_crit", 0.0))
        
    # 2. Persistence
    preface_persistence_ticks.set(decision.get("persistence_count", 0))
    
    # 3. Utilities
    utilities = decision.get("expected_utilities", {})
    preface_expected_utility._metrics.clear()
    for action, eu in utilities.items():
        preface_expected_utility.labels(action=action).set(eu)
        
    preface_delta_eu.set(decision.get("delta_eu", 0.0))
    
    # 4. State & Eligibility
    state = decision.get("state", "HEALTHY")
    
    preface_intervention_eligible.set(1 if state == "INTERVENE" else 0)
    preface_cooldown_remaining_seconds.set(decision.get("cooldown_remaining", 0.0))
    preface_rate_limit_usage.set(rate_limit_len)
    
    # 5. Events / Boundaries
    action = decision.get("action", "Do_Nothing")
    if state == "INTERVENE" and action != "Do_Nothing":
        if shadow_mode:
            preface_would_execute_total.labels(action=action).inc()
        else:
            preface_executed_total.labels(action=action).inc()
            
    if state in ["COOLDOWN", "RATE_LIMITED"]:
        reason = decision.get("blocked_reason", state)
        preface_blocked_total.labels(reason=reason).inc()
