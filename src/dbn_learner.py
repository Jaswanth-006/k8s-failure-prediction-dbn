import numpy as np
from typing import Dict, List, Tuple
import networkx as nx

class DBNParameterLearner:
    """
    Learns and calibrates DBN probabilities (Transitions, Emissions, Topological influences)
    from historical or simulated telemetry/state data.
    """
    def __init__(self, num_states: int = 3, min_variance: float = 0.01, min_samples_emission: int = 5):
        self.num_states = num_states
        self.min_variance = min_variance
        self.min_samples_emission = min_samples_emission

        # Hardcoded baseline parameters to fall back to if data is insufficient
        self.baseline_mu = np.array([0.0, 2.5, 5.0], dtype=np.float32)
        self.baseline_sigma = np.array([1.0, 1.2, 1.5], dtype=np.float32)
        self.baseline_T = np.array([
            [0.950, 0.045, 0.005],
            [0.200, 0.650, 0.150],
            [0.020, 0.180, 0.800]
        ], dtype=np.float32)

    def calibrate_emissions(self, states: np.ndarray, anomaly_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrates the Gaussian emission parameters (mu, sigma) for each state.
        
        :param states: 1D array of state labels (0=Normal, 1=Degrading, 2=Critical).
        :param anomaly_scores: 1D array of corresponding anomaly scores.
        :return: (mu_obs, sigma_obs) arrays of length num_states.
        """
        mu_obs = np.zeros(self.num_states, dtype=np.float32)
        sigma_obs = np.zeros(self.num_states, dtype=np.float32)

        for s in range(self.num_states):
            mask = (states == s)
            scores_for_state = anomaly_scores[mask]
            
            if len(scores_for_state) >= self.min_samples_emission:
                mu_obs[s] = float(np.mean(scores_for_state))
                # Apply Bessel's correction (ddof=1) for sample variance
                variance = np.var(scores_for_state, ddof=1)
                # Safeguard against zero variance
                sigma_obs[s] = float(np.sqrt(max(variance, self.min_variance)))
            else:
                # Fallback to baseline if insufficient data
                mu_obs[s] = self.baseline_mu[s]
                sigma_obs[s] = self.baseline_sigma[s]
                
        return mu_obs, sigma_obs

    def calibrate_transitions(self, state_sequences: Dict[str, List[int]]) -> np.ndarray:
        """
        Calibrates the base transition matrix P(H_t | H_{t-1}) using Laplace smoothing.
        
        :param state_sequences: Dictionary mapping service_name -> list of states over time.
        :return: (num_states, num_states) transition matrix.
        """
        # Initialize counts with 1 (Laplace smoothing) to prevent zero probabilities
        transition_counts = np.ones((self.num_states, self.num_states), dtype=np.float32)

        for seq in state_sequences.values():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                s_prev = seq[i]
                s_curr = seq[i+1]
                transition_counts[s_prev, s_curr] += 1.0

        # Normalize rows to sum to 1
        T_learned = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        return T_learned

    def calibrate_topological_influences(self, state_sequences: Dict[str, List[int]], graph: nx.DiGraph) -> Dict[int, np.ndarray]:
        """
        Estimates the increased probability of degradation based on the worst upstream parent state.
        
        :param state_sequences: Dictionary mapping service_name -> list of states over time.
        :param graph: NetworkX DiGraph representing service dependencies.
        :return: Dictionary mapping parent state (1 or 2) -> modifier array [delta_P(N), delta_P(D), delta_P(C)]
        """
        # transition_counts[parent_state][prev_state][curr_state]
        # Using add-1 smoothing
        counts = {
            0: np.ones((self.num_states, self.num_states), dtype=np.float32),
            1: np.ones((self.num_states, self.num_states), dtype=np.float32),
            2: np.ones((self.num_states, self.num_states), dtype=np.float32)
        }
        
        for node in graph.nodes():
            if node not in state_sequences:
                continue
            
            parents = list(graph.predecessors(node))
            seq = state_sequences[node]
            
            if len(seq) < 2:
                continue
                
            for t in range(len(seq) - 1):
                s_prev = seq[t]
                s_curr = seq[t+1]
                
                if len(parents) == 0:
                    worst_parent = 0
                else:
                    parent_states = [state_sequences[p][t] for p in parents if p in state_sequences]
                    worst_parent = max(parent_states) if parent_states else 0
                    
                counts[worst_parent][s_prev, s_curr] += 1.0
                
        # Normalize
        P_given_parent = {}
        for wp in [0, 1, 2]:
            P_given_parent[wp] = counts[wp] / np.sum(counts[wp], axis=1, keepdims=True)
            
        modifiers = {}
        for wp in [1, 2]:
            diff = P_given_parent[wp] - P_given_parent[0]
            avg_diff = np.mean(diff, axis=0)
            avg_diff -= np.mean(avg_diff) # correction to ensure exactly 0 sum
            modifiers[wp] = avg_diff
            
        return modifiers
