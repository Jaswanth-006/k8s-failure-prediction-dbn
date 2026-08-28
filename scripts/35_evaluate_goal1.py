import pandas as pd
import json
import os

def evaluate_persistence(df, t_fault, mode="fixed"):
    results = []
    persistence_counter = 0
    current_rc = "None"
    
    first_detection = None
    first_intervention = None
    
    for idx, row in df.iterrows():
        rc = row['root_cause']
        p_crit = row['p_critical']
        signal = row['ts_train_signal']
        tick = row['experiment_tick'] if 'experiment_tick' in row else row['tick']
        phase = row['phase']
        
        if rc == "None" or rc != "ts-train-service":
            persistence_counter = 0
            current_rc = "None"
        else:
            current_rc = rc
            persistence_counter += 1
            
        req_ticks = 11
        if mode == "adaptive":
            if signal >= 8.0 and p_crit >= 0.9:
                req_ticks = 2
            elif signal >= 4.0 and p_crit >= 0.6:
                req_ticks = 5
                
        intervene = False
        if persistence_counter >= req_ticks:
            intervene = True
            
        t_tick = pd.to_datetime(row['timestamp'])
        
        if rc == "ts-train-service" and first_detection is None and t_tick >= t_fault:
            first_detection = tick
            
        if intervene and first_intervention is None:
            first_intervention = tick
            
        results.append({
            "tick": tick,
            "phase": phase,
            "signal": signal,
            "p_crit": p_crit,
            "persistence_counter": persistence_counter,
            "required_ticks": req_ticks,
            "intervene": intervene
        })
        
    return pd.DataFrame(results), first_detection, first_intervention

def main():
    res_path = "data/experiments/goal1_clean_experiment_results.csv"
    meta_path = "data/experiments/goal1_metadata.json"
    
    if not os.path.exists(res_path):
        print(f"Results file not found: {res_path}")
        return
        
    df = pd.read_csv(res_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    t_fault = pd.to_datetime(meta['fault_injection_time'])
    
    fixed_df, f_det, f_int = evaluate_persistence(df, t_fault, "fixed")
    adapt_df, a_det, a_int = evaluate_persistence(df, t_fault, "adaptive")
    
    print("Fixed Intervention Tick:", f_int)
    print("Adaptive Intervention Tick:", a_int)
    
    def get_fpr(mode_df):
        pre_fault = mode_df[mode_df['phase'] == 'PRE-FAULT']
        fp = pre_fault['intervene'].sum()
        return fp / len(pre_fault) if len(pre_fault) > 0 else 0, fp
        
    f_fpr, f_fp = get_fpr(fixed_df)
    a_fpr, a_fp = get_fpr(adapt_df)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    def get_latency(intervention_tick):
        if intervention_tick is None:
            return None
        t_int = df[df['tick'] == intervention_tick].iloc[0]['timestamp']
        if t_int < t_fault:
            return -1
        return (t_int - t_fault).total_seconds()
        
    f_lat = get_latency(f_int)
    a_lat = get_latency(a_int)
    
    det_lat = get_latency(f_det)
    
    max_p_crit = df['p_critical'].max()
    max_pers = fixed_df['persistence_counter'].max()
    
    recovery_detected = df.iloc[-1]['p_critical'] < 0.2
    overall_loc = "ts-train-service" in df['root_cause'].values
    
    comp_df = pd.DataFrame({
        "Metric": [
            "Detection Latency (s)",
            "Reaction/Intervention Time (s)",
            "False Positive Rate",
            "False Positive Ticks",
            "Strong Localization",
            "Weak Localization",
            "Overall Localization",
            "Maximum P(Critical)",
            "Maximum Persistence",
            "Intervention Triggered",
            "Recovery Detected"
        ],
        "Fixed": [
            det_lat if det_lat else 0,
            f_lat if f_lat else 0,
            f_fpr,
            f_fp,
            False,
            overall_loc,
            overall_loc,
            max_p_crit,
            max_pers,
            f_int is not None,
            recovery_detected
        ],
        "Adaptive": [
            det_lat if det_lat else 0,
            a_lat if a_lat else 0,
            a_fpr,
            a_fp,
            False,
            overall_loc,
            overall_loc,
            max_p_crit,
            adapt_df['persistence_counter'].max(),
            a_int is not None,
            recovery_detected
        ]
    })
    
    comp_df.to_csv("data/experiments/goal1_fixed_vs_adaptive_results.csv", index=False)
    print("Comparison Table:")
    print(comp_df.to_string())

if __name__ == "__main__":
    main()
