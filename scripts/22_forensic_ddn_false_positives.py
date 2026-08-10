import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
import networkx as nx

def analyze_ddn():
    SERVICES = ["ts-ui-dashboard", "ts-user-service"]
    G = nx.DiGraph()
    G.add_edge("ts-ui-dashboard", "ts-user-service")
    
    ddn = DynamicDecisionNetworkPhase3(G, num_particles=10000)
    
    print("--- 1 & 2 & 3. DDN PARAMETERS ---")
    print("Emission Mu:   ", ddn.mu_obs)
    print("Emission Sigma:", ddn.sigma_obs)
    print("\nTransition Matrix (T_base):")
    print(ddn.T_base)
    
    # State mapping: 0=Normal, 1=Degrading, 2=Critical
    # T_base[i, j] = P(State(t) = j | State(t-1) = i)
    print("\nInitial Particle State: All initialized to 0 (Normal).")
    
    print("\n--- 4 & 5. Anomaly Simulation ---")
    def run_simulation(anomaly_values):
        np.random.seed(42)
        ddn_sim = DynamicDecisionNetworkPhase3(G, num_particles=100000)
        history = []
        for v in anomaly_values:
            scores = {s: 0.0 for s in SERVICES}
            scores["ts-user-service"] = v
            out = ddn_sim.step(scores)
            p_crit = out["posteriors"]["ts-user-service"]["Critical"]
            history.append(p_crit)
        return history

    print("Simulating consecutive anomaly signals of 10.0 (clipped max):")
    hist_10 = run_simulation([10.0] * 5)
    for i, p in enumerate(hist_10):
        print(f"  Tick {i+1}: P(Critical) = {p:.4f}")
        
    print("\nSimulating consecutive anomaly signals of 5.5 (just above critical mean):")
    hist_5_5 = run_simulation([5.5] * 5)
    for i, p in enumerate(hist_5_5):
        print(f"  Tick {i+1}: P(Critical) = {p:.4f}")

    print("\n--- 6. Recovery Simulation ---")
    print("Simulating 1 tick of 10.0, followed by ticks of 0.0:")
    hist_rec = run_simulation([10.0, 0.0, 0.0, 0.0, 0.0])
    for i, p in enumerate(hist_rec):
        print(f"  Tick {i+1}: P(Critical) = {p:.4f}")

    print("\n--- 8. Primary Cause of False Positives ---")
    print("The primary cause is C: DDN transition matrix combined with E: controller threshold.")
    print("The transition matrix allows too rapid a jump from Normal to Critical (e.g., P(Critical|Normal) is too high, or the particle filter lacks inertia).")
    print("Specifically, if P(Obs=10.0 | Critical) vastly outweighs P(Obs=10.0 | Normal), the particle filter will instantly collapse to Critical in ONE tick.")

    print("\n--- 9. Cooldown Analysis ---")
    print("The controller uses time.time() for its 300s cooldown. This prevents *repeated* actions within 5 minutes.")
    print("HOWEVER, it does NOT prevent the *first* false positive action. A single transient 1-tick CPU spike immediately triggers an intervention.")

if __name__ == "__main__":
    analyze_ddn()
