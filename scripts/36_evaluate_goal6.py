import os
import sys
import numpy as np
import networkx as nx
import random
import pandas as pd
import importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.goal6_evaluator import Goal6Evaluator
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.dbn_learner import DBNParameterLearner

def setup_mock_graph():
    G = nx.DiGraph()
    services = ["ts-ui-dashboard", "ts-train-service", "ts-route-service", "ts-order-service"]
    G.add_nodes_from(services)
    G.add_edge("ts-ui-dashboard", "ts-train-service")
    G.add_edge("ts-ui-dashboard", "ts-route-service")
    G.add_edge("ts-ui-dashboard", "ts-order-service")
    return G

def get_learned_parameters(G):
    # Import Goal 5 script dynamically since it starts with a number
    spec = importlib.util.spec_from_file_location("goal5", os.path.join(os.path.dirname(__file__), "31_run_goal5_experiment.py"))
    goal5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(goal5)
    
    # Seed for deterministic calibration
    np.random.seed(42)
    random.seed(42)
    
    state_seqs, anom_scores = goal5.generate_synthetic_data(G, num_ticks=2000)
    all_states = np.concatenate([np.array(v) for v in state_seqs.values()])
    all_scores = np.concatenate([np.array(v) for v in anom_scores.values()])
    
    learner = DBNParameterLearner()
    learned_mu, learned_sigma = learner.calibrate_emissions(all_states, all_scores)
    learned_T = learner.calibrate_transitions(state_seqs)
    learned_topological = learner.calibrate_topological_influences(state_seqs, G)
    
    return learned_mu, learned_sigma, learned_T, learned_topological

def run_goal6_experiments(num_experiments=100):
    print(f"--- Starting Goal 6 Evaluation: Running {num_experiments} Controlled Experiments ---")
    
    G = setup_mock_graph()
    evaluator = Goal6Evaluator()
    services = list(G.nodes())
    possible_faults = ["ts-train-service", "ts-route-service", "ts-order-service"]
    
    # Load Goal 5 Learned Parameters
    print("Loading Goal 5 learned parameters...")
    learned_mu, learned_sigma, learned_T, learned_topological = get_learned_parameters(G)
    
    print("\nLearned Parameters summary:")
    print(f"Mu: {learned_mu}")
    print(f"Sigma: {learned_sigma}")
    
    # For reporting breakdown
    rca_breakdown = []
    
    # Ensure reproducible evaluation
    np.random.seed(1337)
    random.seed(1337)
    
    for i in range(100):
        is_positive = (i >= 50)
        
        # Instantiate DDN with learned parameters
        ddn = DynamicDecisionNetworkPhase3(
            service_graph=G, 
            num_particles=500,
            learned_T=learned_T,
            learned_mu=learned_mu,
            learned_sigma=learned_sigma,
            learned_topological=learned_topological
        )
        
        total_ticks = 30
        fp_occurred = False
        t_detect = None
        predicted_rc = "None"
        
        if is_positive:
            t_fault = 15
            actual_rc = random.choice(possible_faults)
        else:
            t_fault = None
            actual_rc = None
        
        for tick in range(total_ticks):
            # Generate signals
            anomaly_signals = {}
            for s in services:
                if is_positive and tick >= t_fault:
                    if s == actual_rc:
                        # Faulty (Critical)
                        anomaly_signals[s] = np.random.normal(5.0, 0.5)
                    elif s == "ts-ui-dashboard":
                        # Upstream impact (Degrading)
                        anomaly_signals[s] = np.random.normal(1.5, 0.5)
                    else:
                        # Healthy
                        anomaly_signals[s] = max(0.0, np.random.normal(0.1, 0.1))
                else:
                    # Healthy tick (either a negative trial, or before t_fault in a positive trial)
                    anomaly_signals[s] = max(0.0, np.random.normal(0.1, 0.1))
                        
            out = ddn.step(anomaly_signals)
            
            # Check for alarm: P(Critical) > 0.5 on any service
            alarm_triggered = False
            for s, probs in out["posteriors"].items():
                if probs["Critical"] > 0.5:
                    alarm_triggered = True
                    break
                    
            if alarm_triggered:
                if is_positive:
                    if tick >= t_fault and t_detect is None:
                        t_detect = tick
                        predicted_rc = out["root_cause"]
                else:
                    # In a purely negative trial, any alarm is a false positive
                    fp_occurred = True
                    
        evaluator.record_experiment(
            experiment_id=f"exp_{i}",
            is_positive=is_positive,
            t_fault=t_fault,
            t_detect=t_detect,
            actual_rc=actual_rc,
            predicted_rc=predicted_rc,
            fp_occurred=fp_occurred
        )
        
        if is_positive and t_detect is not None:
            rca_breakdown.append({
                "actual": actual_rc,
                "predicted": predicted_rc
            })
        
    metrics = evaluator.compute_metrics()
    
    print("\n--- Final Goal 6 Evaluation Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k.replace('_', ' ').title()}: {v:.4f}")
        else:
            print(f"{k.replace('_', ' ').title()}: {v}")
            
    # RCA Breakdown
    print("\n--- Root Cause Breakdown ---")
    counts = {}
    for item in rca_breakdown:
        pair = (item["actual"], item["predicted"])
        counts[pair] = counts.get(pair, 0) + 1
        
    total_detected = len(rca_breakdown)
    print(f"Total Detected Faults (RCA Denominator): {total_detected}")
    correct = 0
    injected_counts = {}
    for item in rca_breakdown:
        injected_counts[item["actual"]] = injected_counts.get(item["actual"], 0) + 1
        
    print(f"\nInjected Breakdown (Only Detected):")
    for rc, count in injected_counts.items():
        print(f"  {rc}: {count}")
        
    print(f"\nPrediction Details:")
    for (a, p), c in counts.items():
        print(f"  Actual: {a} -> Predicted: {p} (Count: {c})")
        if a == p:
            correct += c
            
    print(f"\nTotal Correct RCA Predictions: {correct}")
    print(f"Exact RCA Denominator (Detected Faults): {total_detected}")
            
    # Save to CSV
    os.makedirs("data/experiments/goal6", exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv("data/experiments/goal6/evaluation_summary.csv", index=False)
    print("\nResults saved to data/experiments/goal6/evaluation_summary.csv")

if __name__ == "__main__":
    run_goal6_experiments(100)
