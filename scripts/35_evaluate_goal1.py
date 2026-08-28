"""
Goal 1 Offline Evaluator: Fixed vs Adaptive Persistence
Runs both modes against the same collected live telemetry.
DO NOT modify AdaptivePersistence, DBN, DecisionPolicy, or controller code.
"""
import pandas as pd
import json
import os


def evaluate_persistence(df, t_fault, mode="fixed"):
    """
    Simulate persistence evaluation for Fixed or Adaptive mode.

    Returns:
      results_df         - per-tick trace
      first_detection    - tick number of first ts-train detection AFTER fault
      first_intervention - tick number of first eligible intervention
    """
    results = []
    persistence_counter = 0
    current_rc = "None"

    first_detection = None
    first_intervention = None

    for _, row in df.iterrows():
        rc = str(row['root_cause']) if not pd.isna(row['root_cause']) else "None"
        p_crit = row['p_critical']
        signal = row['ts_train_signal']
        tick = int(row['tick'])
        phase = row['phase']
        t_tick = pd.to_datetime(row['timestamp'])

        # ---- Persistence counter: consecutive ticks of ts-train as root cause ----
        if rc == "ts-train-service":
            persistence_counter += 1
            current_rc = rc
        else:
            persistence_counter = 0
            current_rc = "None"

        # ---- Required persistence threshold ----
        req_ticks = 11  # Fixed default
        if mode == "adaptive":
            if signal >= 8.0 and p_crit >= 0.9:
                req_ticks = 2   # HIGH severity
            elif signal >= 4.0 and p_crit >= 0.6:
                req_ticks = 5   # MODERATE severity
            else:
                req_ticks = 11  # LOW / normal

        intervene = (persistence_counter >= req_ticks)

        # ---- Detection: first tick AFTER fault where ts-train is confirmed RC ----
        if rc == "ts-train-service" and first_detection is None and t_tick >= t_fault:
            first_detection = tick

        # ---- Intervention: first tick where persistence meets threshold ----
        if intervene and first_intervention is None:
            first_intervention = tick

        results.append({
            "tick": tick,
            "phase": phase,
            "timestamp": t_tick,
            "signal": signal,
            "p_crit": p_crit,
            "root_cause": rc,
            "persistence_counter": persistence_counter,
            "required_ticks": req_ticks,
            "intervene": intervene
        })

    return pd.DataFrame(results), first_detection, first_intervention


def compute_localization(df):
    """
    Strong: ts-train-service is root_cause for >= 5 consecutive EXPERIMENT ticks.
    Weak:   ts-train-service appears as root_cause at any EXPERIMENT tick.
    """
    exp_df = df[df['phase'] == 'EXPERIMENT'].copy()
    exp_df['is_train'] = exp_df['root_cause'].apply(
        lambda x: str(x) == 'ts-train-service'
    )
    weak = exp_df['is_train'].any()

    # Max consecutive ts-train ticks in experiment phase
    max_consec = 0
    run = 0
    for v in exp_df['is_train']:
        if v:
            run += 1
            max_consec = max(max_consec, run)
        else:
            run = 0
    strong = max_consec >= 5

    return strong, weak


def main():
    res_path = "data/experiments/goal1_clean_experiment_results.csv"
    meta_path = "data/experiments/goal1_metadata.json"

    if not os.path.exists(res_path):
        print(f"Results file not found: {res_path}")
        return

    df = pd.read_csv(res_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    with open(meta_path, "r") as f:
        meta = json.load(f)

    t_fault = pd.to_datetime(meta['fault_injection_time'])

    print("=" * 70)
    print("GOAL 1 AUDIT — RAW DIAGNOSTIC OUTPUT")
    print("=" * 70)
    print(f"Fault injection time : {t_fault}")

    # ---- Print all ts-train-service ticks ----
    print("\n--- All ticks where root_cause == ts-train-service ---")
    for _, r in df[df['root_cause'] == 'ts-train-service'].iterrows():
        delta = (r['timestamp'] - t_fault).total_seconds()
        print(f"  Tick {r['tick']:3d} [{r['phase']:10s}]  ts={r['timestamp']}  "
              f"delta={delta:+.2f}s  sig={r['ts_train_signal']:.4f}  "
              f"p_crit={r['p_critical']:.4f}  pers(csv)={r['persistence']}")

    # ---- First experiment tick ----
    exp_first = df[df['phase'] == 'EXPERIMENT'].iloc[0]
    print(f"\nFirst EXPERIMENT tick: Tick {exp_first['tick']}  "
          f"ts={exp_first['timestamp']}  "
          f"delta={(exp_first['timestamp']-t_fault).total_seconds():+.2f}s from fault")

    # ---- Evaluate Fixed ----
    fixed_df, f_det, f_int = evaluate_persistence(df, t_fault, "fixed")

    # ---- Evaluate Adaptive ----
    adapt_df, a_det, a_int = evaluate_persistence(df, t_fault, "adaptive")

    print("\n--- Adaptive persistence trace (ts-train-service ticks only) ---")
    for _, r in adapt_df[adapt_df['root_cause'] == 'ts-train-service'].iterrows():
        delta = (r['timestamp'] - t_fault).total_seconds()
        marker = " *** INTERVENTION ***" if r['intervene'] else ""
        print(f"  Tick {r['tick']:3d} [{r['phase']:10s}]  delta={delta:+.2f}s  "
              f"sig={r['signal']:.4f}  p={r['p_crit']:.4f}  "
              f"pers={r['persistence_counter']}/{r['required_ticks']}{marker}")

    # ---- Latency helper ----
    def get_latency_seconds(intervention_tick):
        if intervention_tick is None:
            return None
        row = df[df['tick'] == intervention_tick].iloc[0]
        t_int = row['timestamp']
        if t_int < t_fault:
            return None  # pre-fault, not valid
        return (t_int - t_fault).total_seconds()

    f_det_lat = get_latency_seconds(f_det)
    a_det_lat = get_latency_seconds(a_det)  # same underlying detection event
    f_int_lat = get_latency_seconds(f_int)
    a_int_lat = get_latency_seconds(a_int)

    # If detection is the same for both models (same ts-train first post-fault tick),
    # use a single detection latency value.
    det_lat = f_det_lat if f_det_lat is not None else a_det_lat

    print(f"\nFirst ts-train detection tick after fault: Tick {f_det}  ({det_lat:.2f}s from fault)" if det_lat is not None else "\nNo ts-train detection after fault")

    if a_int is not None:
        a_int_ts = df[df['tick'] == a_int].iloc[0]['timestamp']
        print(f"Adaptive intervention tick: Tick {a_int}  ts={a_int_ts}  ({a_int_lat:.2f}s from fault)")
    else:
        print("Adaptive intervention: None (threshold not reached)")

    if f_int is not None:
        f_int_ts = df[df['tick'] == f_int].iloc[0]['timestamp']
        print(f"Fixed intervention tick: Tick {f_int}  ts={f_int_ts}  ({f_int_lat:.2f}s from fault)")
    else:
        print("Fixed intervention: None (11-tick threshold never reached)")

    # ---- Recovery ----
    recovery_df = df[df['phase'] == 'RECOVERY']
    # Recovery = p_critical < 0.2 sustained for at least 3 ticks in recovery
    recovery_ticks_below = (recovery_df['p_critical'] < 0.2).sum()
    recovery_detected = recovery_ticks_below >= 3
    recovery_first = recovery_df[recovery_df['p_critical'] < 0.2]
    rec_tick = recovery_first.iloc[0]['tick'] if len(recovery_first) else None
    rec_ts = recovery_first.iloc[0]['timestamp'] if len(recovery_first) else None
    print(f"\nRecovery: {recovery_ticks_below}/{len(recovery_df)} RECOVERY ticks had p_crit < 0.2")
    print(f"First recovery tick (p_crit < 0.2): Tick {rec_tick}  ts={rec_ts}")
    print(f"Recovery Detected: {recovery_detected}")

    # ---- FPR ----
    def get_fpr(mode_df):
        pre = mode_df[mode_df['phase'] == 'PRE-FAULT']
        fp = int(pre['intervene'].sum())
        total = len(pre)
        fpr = fp / total if total > 0 else 0.0
        return fpr, fp, total

    f_fpr, f_fp, f_pre_n = get_fpr(fixed_df)
    a_fpr, a_fp, a_pre_n = get_fpr(adapt_df)

    # ---- Localization (shared — same telemetry, same DBN output) ----
    strong_loc, weak_loc = compute_localization(df)
    overall_loc = weak_loc

    # ---- Max persistence ----
    # Maximum consecutive ts-train ticks reached during experiment
    exp_train = adapt_df[(adapt_df['phase'] == 'EXPERIMENT') &
                         (adapt_df['root_cause'] == 'ts-train-service')]
    max_pers_exp = int(exp_train['persistence_counter'].max()) if len(exp_train) else 0
    max_pers_fixed = int(fixed_df['persistence_counter'].max())

    # ---- Build comparison table ----
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return round(v, 2)
        return v

    comp_df = pd.DataFrame({
        "Metric": [
            "Fault Injection Time",
            "Detection Latency (s)",
            "Reaction/Intervention Time (s)",
            "False Positive Rate",
            "False Positive Ticks / 30",
            "Strong Localization",
            "Weak Localization",
            "Overall Localization",
            "Maximum P(Critical)",
            "Max Consecutive Persistence (ts-train)",
            "Required Persistence Threshold",
            "Intervention Triggered",
            "Recovery Detected",
        ],
        "Fixed": [
            str(t_fault),
            fmt(det_lat),
            fmt(f_int_lat),
            fmt(f_fpr),
            f"{f_fp}/{f_pre_n}",
            strong_loc,
            weak_loc,
            overall_loc,
            fmt(df['p_critical'].max()),
            max_pers_fixed,
            11,
            f_int is not None,
            recovery_detected,
        ],
        "Adaptive": [
            str(t_fault),
            fmt(det_lat),
            fmt(a_int_lat),
            fmt(a_fpr),
            f"{a_fp}/{a_pre_n}",
            strong_loc,
            weak_loc,
            overall_loc,
            fmt(df['p_critical'].max()),
            max_pers_exp,
            "2 (HIGH) / 5 (MOD) / 11 (LOW)",
            a_int is not None,
            recovery_detected,
        ]
    })

    os.makedirs("data/experiments", exist_ok=True)
    comp_df.to_csv("data/experiments/goal1_fixed_vs_adaptive_results.csv", index=False)

    print("\n" + "=" * 70)
    print("CORRECTED GOAL 1 COMPARISON TABLE")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("\nSaved to: data/experiments/goal1_fixed_vs_adaptive_results.csv")


if __name__ == "__main__":
    main()
