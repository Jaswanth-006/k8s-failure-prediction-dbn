import sys
import os
import networkx as nx
import numpy as np

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3

def build_sample_service_graph() -> nx.DiGraph:
    """Builds TrainTicket core service call graph (DAG)."""
    G = nx.DiGraph()
    G.add_edge("ts-ui-dashboard", "ts-user-service")
    G.add_edge("ts-ui-dashboard", "ts-train-service")
    G.add_edge("ts-ui-dashboard", "ts-route-service")
    G.add_edge("ts-ui-dashboard", "ts-order-service")
    G.add_edge("ts-order-service", "ts-payment-service")
    G.add_edge("ts-order-service", "ts-inventory-service")
    G.add_edge("ts-order-service", "ts-station-service")
    return G

def test_ddn_stability():
    print("================================================================")
    print(" Phase 3 DDN Numerical Stability Test")
    print("================================================================\n")
    
    G = build_sample_service_graph()
    ddn = DynamicDecisionNetworkPhase3(G, num_particles=1000)
    
    # 1. Test Ordinary Anomaly
    print("--- Test 1: Ordinary Anomaly (Score = 1.0) ---")
    signals_ordinary = {s: 1.0 for s in ddn.services}
    out_ordinary = ddn.step(signals_ordinary)
    
    for s in ddn.services:
        p_norm = out_ordinary["posteriors"][s]["Normal"]
        p_deg = out_ordinary["posteriors"][s]["Degrading"]
        p_crit = out_ordinary["posteriors"][s]["Critical"]
        total = p_norm + p_deg + p_crit
        assert np.isclose(total, 1.0), f"Posteriors do not sum to 1.0: {total}"
        
        # With anomaly = 1.0, it should not be entirely Critical
        assert p_crit < 0.9, f"P(Critical) is too high for ordinary anomaly: {p_crit}"
        
    print("[PASS] Ordinary anomalies process correctly. Posteriors sum to 1.0.\n")

    # 2. Test Extreme Anomaly (51 million, previously caused underflow)
    print("--- Test 2: Extreme Anomaly (Score = 51,000,000.0) ---")
    # Reset particles
    ddn.particles = np.zeros((ddn.num_particles, ddn.num_services), dtype=int)
    signals_extreme = {s: 51000000.0 for s in ddn.services}
    
    # Run the particle filter for 5 ticks to allow the state to converge
    # (Because predicting 'Critical' for all 8 services simultaneously from state 0
    # has a probability of 0.005^8, no single particle will guess it in tick 1.
    # It takes a few ticks for the Critical state to propagate).
    for tick in range(5):
        out_extreme = ddn.step(signals_extreme)
    
    for s in ddn.services:
        p_norm = out_extreme["posteriors"][s]["Normal"]
        p_deg = out_extreme["posteriors"][s]["Degrading"]
        p_crit = out_extreme["posteriors"][s]["Critical"]
        total = p_norm + p_deg + p_crit
        assert np.isclose(total, 1.0), f"Posteriors do not sum to 1.0: {total}"
        
        # With massive anomaly over 5 ticks, Critical should dominate
        assert p_crit > 0.9, f"Extreme anomaly did not map to Critical state over 5 ticks. p_crit={p_crit}"
        assert not np.isnan(p_crit), "Encountered NaN in posteriors"
        assert not np.isinf(p_crit), "Encountered Inf in posteriors"
        
    print("[PASS] Extreme anomaly maps to Critical state without NaN/Inf/Underflow.\n")
    
    print("================================================================")
    print(" All DDN numerical stability tests passed.")
    print("================================================================")

if __name__ == "__main__":
    test_ddn_stability()
