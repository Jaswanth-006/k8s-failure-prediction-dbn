"""
PREFACE-DDN Module 3 (Phase 3): Dynamic Bayesian & Decision Network (DDN) Core
Includes:
- Health State Tracking: Normal (0), Degrading (1), Critical (2)
- Service Call Graph Topological Dependencies (DAG)
- Vectorized JAX/NumPy Particle Filter Forward Engine
- Phase 3 Update: Log-Likelihood Log-Sum-Exp AND dynamic signal clipping.
- Decision Nodes (Do Nothing, Scale-Out, Restart, Reschedule, Traffic Shift)
- Expected Utility Calculations: EU(A) = sum P(State) * U(State, Action)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any

# Health states: 0 = Normal, 1 = Degrading, 2 = Critical
HEALTH_STATES = [0, 1, 2]
STATE_NAMES = ["Normal", "Degrading", "Critical"]

# Candidate Actions
ACTIONS = ["Do_Nothing", "Scale_Out", "Restart_Pod", "Reschedule_Pod", "Traffic_Shift"]

class DynamicDecisionNetworkPhase3:
    def __init__(self, service_graph: nx.DiGraph, num_particles: int = 500):
        """
        :param service_graph: Directed graph of microservices (NetworkX DAG).
        :param num_particles: Number of Monte Carlo particles for fast vectorized inference.
        """
        self.graph = service_graph
        self.services = list(nx.topological_sort(service_graph)) if nx.is_directed_acyclic_graph(service_graph) else list(service_graph.nodes())
        self.num_services = len(self.services)
        self.service_to_idx = {s: i for i, s in enumerate(self.services)}
        self.num_particles = num_particles
        
        # Initialize Goal 3 Directional Causal Analyzer
        try:
            from src.causal_rca import DirectionalCausalAnalyzer
            self.causal_analyzer = DirectionalCausalAnalyzer(service_graph)
        except ImportError:
            self.causal_analyzer = None

        # Parent mapping for topological causality
        self.parent_indices = []
        for s in self.services:
            parents = list(self.graph.predecessors(s))
            self.parent_indices.append([self.service_to_idx[p] for p in parents])

        # Baseline 3x3 Transition Matrix P(H_t | H_{t-1})
        self.T_base = np.array([
            [0.950, 0.045, 0.005], # Normal -> N, D, C
            [0.200, 0.650, 0.150], # Degrading -> N, D, C
            [0.020, 0.180, 0.800]  # Critical -> N, D, C
        ], dtype=np.float32)

        # Gaussian Emission parameters per state: P(a_t | State = k) ~ N(mu_k, sigma_k^2)
        # Expected observation ranges: Normal ~ 0.0, Degrading ~ 2.5, Critical ~ 5.0.
        self.mu_obs = np.array([0.0, 2.5, 5.0], dtype=np.float32)
        self.sigma_obs = np.array([1.0, 1.2, 1.5], dtype=np.float32)

        # Initialize particle filter state: (num_particles, num_services) integers in {0, 1, 2}
        self.particles = np.zeros((self.num_particles, self.num_services), dtype=int)

        # Utility Function Matrix U(State, Action)
        # Rows: Normal, Degrading, Critical
        # Cols: Do_Nothing, Scale_Out, Restart_Pod, Reschedule_Pod, Traffic_Shift
        self.utility_matrix = np.array([
            [ 10.0,   0.0,  -5.0, -15.0,  -2.0], # State = Normal
            [ -5.0,  15.0,  10.0,   5.0,  12.0], # State = Degrading
            [-50.0,  20.0,  30.0,  45.0,  25.0]  # State = Critical
        ], dtype=np.float32)

    def step(
        self,
        anomaly_signals: Dict[str, float],
        node_pressure_flag: bool = False,
        edge_telemetry: Dict = None,
        service_telemetry: Dict = None,
    ) -> Dict[str, Any]:
        """
        Processes 1-minute tick: updates particle belief state, computes P(Critical),
        performs MAP root-cause localization, and computes Expected Utilities EU(A).
        """
        # 1. State Transition Step (Predict)
        for s_idx in range(self.num_services):
            parents = self.parent_indices[s_idx]
            for p_idx in range(self.num_particles):
                prev_state = self.particles[p_idx, s_idx]
                t_row = np.copy(self.T_base[prev_state])

                if len(parents) > 0:
                    parent_states = [self.particles[p_idx, parent_i] for parent_i in parents]
                    worst_parent = max(parent_states)
                    if worst_parent == 1: 
                        t_row[1] += 0.10
                        t_row[2] += 0.05
                    elif worst_parent == 2: 
                        t_row[1] += 0.05
                        t_row[2] += 0.20
                    t_row = t_row / np.sum(t_row)

                self.particles[p_idx, s_idx] = np.random.choice([0, 1, 2], p=t_row)

        # 2. Observation Update & Resampling Step (Weight)
        log_particle_weights = np.zeros(self.num_particles, dtype=np.float32)
        for s_idx, s in enumerate(self.services):
            obs_a = anomaly_signals.get(s, 0.0)
            
            # --- PHASE 3 NUMERICAL STABILITY FIX: Signal Clipping ---
            # Rationale: While Log-Sum-Exp prevents float32 underflow mathematically,
            # particle filters suffer from "weight degeneracy" if the observations are
            # impossibly far from the predicted states.
            # In Goal 4, the anomaly signals are already log1p-transformed.
            # We clip at 15.0 to allow extreme signals to remain distinct
            # without causing particle filter collapse.
            obs_a = np.clip(obs_a, 0.0, 15.0)
            
            states = self.particles[:, s_idx]
            mus = self.mu_obs[states]
            sigmas = self.sigma_obs[states]
            
            # Compute Gaussian log-likelihood for each particle
            log_likelihoods = -np.log(np.sqrt(2 * np.pi) * sigmas) - 0.5 * ((obs_a - mus) / sigmas) ** 2
            log_particle_weights += log_likelihoods

        # Log-Sum-Exp Trick
        max_log_w = np.max(log_particle_weights)
        if np.isneginf(max_log_w):
            particle_weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles
        else:
            shifted_log_weights = log_particle_weights - max_log_w
            particle_weights = np.exp(shifted_log_weights)
            
            total_w = np.sum(particle_weights)
            if total_w > 0:
                particle_weights /= total_w
            else:
                particle_weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

        # Resample particles
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=particle_weights)
        self.particles = self.particles[indices]

        # 3. Compute Posterior Probabilities P(State) per Service
        posteriors: Dict[str, Dict[str, float]] = {}
        for s_idx, s in enumerate(self.services):
            states = self.particles[:, s_idx]
            p_normal = float(np.mean(states == 0))
            p_degrading = float(np.mean(states == 1))
            p_critical = float(np.mean(states == 2))
            posteriors[s] = {
                "Normal": p_normal,
                "Degrading": p_degrading,
                "Critical": p_critical
            }

        # 4. Topological Root Cause Localization
        causal_data = None

        if self.causal_analyzer:
            causal_data = self.causal_analyzer.step(
                anomaly_signals,
                posteriors,
                edge_telemetry=edge_telemetry,
                service_telemetry=service_telemetry,
            )
            root_cause_service = causal_data["root_cause"]
        else:
            root_cause_service = self._localize_root_cause(posteriors)
        # 5. Compute Expected Utilities EU(A) per Service (Decision Nodes)
        expected_utilities: Dict[str, Dict[str, float]] = {}
        for s in self.services:
            p_vec = np.array([
                posteriors[s]["Normal"],
                posteriors[s]["Degrading"],
                posteriors[s]["Critical"]
            ], dtype=np.float32)

            eu_vec = p_vec @ self.utility_matrix

            if node_pressure_flag and posteriors[s]["Critical"] > 0.3:
                resched_idx = ACTIONS.index("Reschedule_Pod")
                restart_idx = ACTIONS.index("Restart_Pod")
                eu_vec[resched_idx] += 35.0
                eu_vec[restart_idx] -= 25.0

            expected_utilities[s] = {action: float(eu_vec[idx]) for idx, action in enumerate(ACTIONS)}

        return {
            "posteriors": posteriors,
            "root_cause": root_cause_service,
            "expected_utilities": expected_utilities,
            "causal_data": causal_data
        }

    def _localize_root_cause(self, posteriors: Dict[str, Dict[str, float]]) -> str:
        critical_services = [s for s in self.services if posteriors[s]["Critical"] > 0.4]
        if not critical_services:
            return "None"

        for s in critical_services:
            parents = list(self.graph.predecessors(s))
            has_critical_parent = any(p in critical_services for p in parents)
            if not has_critical_parent:
                return s

        return critical_services[0]
