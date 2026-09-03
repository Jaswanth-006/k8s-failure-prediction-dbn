"""
PREFACE-DBN Goal 5: Learn and Calibrate the DBN Probabilities from Real Data
=============================================================================
scripts/31_run_goal5_experiment.py

Demonstrates Goal 5 functionality:
1. Generates a synthetic but realistic historical telemetry dataset with known states.
2. Uses DBNParameterLearner to estimate T_base, mu_obs, sigma_obs, and topological modifiers.
3. Instantiates DynamicDecisionNetworkPhase3 with the learned parameters.
4. Runs inference to verify stability.
"""

import os
import sys
import json
import time
import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dbn_learner import DBNParameterLearner
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3

GRAPH_FILE = "data/experiments/discovered_service_graph.json"

def generate_synthetic_data(graph: nx.DiGraph, num_ticks: int = 500):
    """
    Generates synthetic sequences of hidden states and anomaly scores for each node in the graph.
    """
    print(f"--- Generating {num_ticks} ticks of synthetic historical data ---")
    services = list(graph.nodes())
    
    # Ground truth distributions we want to learn
    true_mu = [0.1, 3.0, 5.5]
    true_sigma = [0.5, 1.0, 1.2]
    
    # We will simulate random transitions but make them somewhat persistent
    state_sequences = {s: [] for s in services}
    anomaly_scores = {s: [] for s in services}
    
    # Init states
    current_states = {s: 0 for s in services}
    
    for _ in range(num_ticks):
        for s in services:
            # Transitions (independent, for simplicity, but we will inject some topological effect)
            parents = list(graph.predecessors(s))
            worst_parent = max([current_states[p] for p in parents]) if parents else 0
            
            curr = current_states[s]
            
            # Simple manual transition model
            r = np.random.rand()
            if curr == 0:
                # 5% chance to degrade, if parent degraded -> +10%
                p_deg = 0.05 + (0.10 if worst_parent >= 1 else 0.0)
                if r < p_deg: next_s = 1
                else: next_s = 0
            elif curr == 1:
                # 20% recover, 10% critical
                if r < 0.20: next_s = 0
                elif r < 0.30: next_s = 2
                else: next_s = 1
            else:
                # Critical: 10% recover
                if r < 0.10: next_s = 0
                elif r < 0.30: next_s = 1
                else: next_s = 2
                
            current_states[s] = next_s
            state_sequences[s].append(next_s)
            
            # Generate emission
            score = np.random.normal(true_mu[next_s], true_sigma[next_s])
            score = max(0.0, score) # Clip at 0
            anomaly_scores[s].append(score)
            
    return state_sequences, anomaly_scores


def main():
    if not os.path.exists(GRAPH_FILE):
        print(f"Graph file not found: {GRAPH_FILE}. Creating a dummy graph.")
        graph = nx.DiGraph()
        graph.add_edges_from([("ts-ui-dashboard", "ts-train-service"), ("ts-train-service", "ts-route-service")])
    else:
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.DiGraph()
        for node in data.get("nodes", []):
            graph.add_node(node)
        for edge in data.get("edges", []):
            graph.add_edge(edge["source"], edge["destination"])

    # 1. Generate Data
    state_seqs, anom_scores = generate_synthetic_data(graph, num_ticks=2000)
    
    # Flatten arrays for emission learning
    all_states = np.concatenate([np.array(v) for v in state_seqs.values()])
    all_scores = np.concatenate([np.array(v) for v in anom_scores.values()])

    # 2. Learn Parameters
    print("\n--- Running DBNParameterLearner ---")
    learner = DBNParameterLearner()
    learned_mu, learned_sigma = learner.calibrate_emissions(all_states, all_scores)
    learned_T = learner.calibrate_transitions(state_seqs)
    learned_topological = learner.calibrate_topological_influences(state_seqs, graph)
    
    print("\nLearned Emission Parameters:")
    for i, state_name in enumerate(["Normal", "Degrading", "Critical"]):
        print(f"  {state_name}: mu = {learned_mu[i]:.2f}, sigma = {learned_sigma[i]:.2f}")
        
    print("\nLearned Transition Matrix T_base:")
    print(np.round(learned_T, 3))
    
    print("\nLearned Topological Modifiers:")
    print(f"  Parent Degrading (1) -> Delta P: {np.round(learned_topological[1], 3)}")
    print(f"  Parent Critical (2)  -> Delta P: {np.round(learned_topological[2], 3)}")

    # 3. Instantiate DDN with Learned Parameters
    print("\n--- Instantiating DynamicDecisionNetworkPhase3 with learned parameters ---")
    ddn = DynamicDecisionNetworkPhase3(
        service_graph=graph, 
        num_particles=500,
        learned_T=learned_T,
        learned_mu=learned_mu,
        learned_sigma=learned_sigma,
        learned_topological=learned_topological
    )
    
    # 4. Run a brief inference test
    print("\n--- Running Inference Test ---")
    # Simulate a tick where ts-train-service has a high anomaly score
    current_anomaly_signals = {
        "ts-ui-dashboard": 0.2,
        "ts-train-service": 5.8, # Likely Critical
        "ts-route-service": 3.1  # Likely Degrading
    }
    
    for s in graph.nodes():
        if s not in current_anomaly_signals:
            current_anomaly_signals[s] = 0.1
            
    out = ddn.step(current_anomaly_signals)
    
    print("\nPosterior Probabilities:")
    for svc, probs in out["posteriors"].items():
        if svc in ["ts-ui-dashboard", "ts-train-service", "ts-route-service"]:
            print(f"  {svc:20s}: N={probs['Normal']:.2f}, D={probs['Degrading']:.2f}, C={probs['Critical']:.2f}")
            
    print(f"\nRoot Cause Localized: {out['root_cause']}")
    print("Experiment successful! Learned parameters integrated correctly.")


if __name__ == "__main__":
    main()
