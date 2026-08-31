import networkx as nx
from typing import Dict, Any
import numpy as np

class DirectionalCausalAnalyzer:
    """
    Goal 3: Directional Causality and Backpressure-Aware Root Cause Analysis
    
    This module analyzes a directed service graph (A -> B means A calls B).
    It computes a causal root cause score based on:
    - IntrinsicEvidence: the service's own health (anomaly, posterior).
    - UpstreamCausalEvidence: how strongly its downstream dependents' degradations are explained by it.
    - VictimEvidence: how strongly its own degradation is explained by an unhealthy upstream parent.
    """
    def __init__(self, service_graph: nx.DiGraph):
        self.graph = service_graph
        self.services = list(service_graph.nodes())
        
        # Maintain temporal history for each service
        self.history = {
            s: {
                "prev_anomaly": 0.0,
                "prev_critical_prob": 0.0,
                "degradation_start_tick": -1
            } for s in self.services
        }
        self.current_tick = 0
        
    def step(self, anomaly_signals: Dict[str, float], posteriors: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        self.current_tick += 1
        
        # 1. Update temporal degradation start times
        for s in self.services:
            p_crit = posteriors.get(s, {}).get("Critical", 0.0)
            p_deg = posteriors.get(s, {}).get("Degrading", 0.0)
            unhealthy_prob = p_crit + p_deg
            
            # Threshold to consider the service "degrading"
            if unhealthy_prob > 0.4:
                if self.history[s]["degradation_start_tick"] == -1:
                    self.history[s]["degradation_start_tick"] = self.current_tick
            else:
                # Reset if healthy
                self.history[s]["degradation_start_tick"] = -1
                
        # 2. Calculate scores for all services
        scores = {}
        for s in self.services:
            scores[s] = self._calculate_scores(s, anomaly_signals, posteriors)
            
        # 3. Update history for the next tick
        for s in self.services:
            self.history[s]["prev_anomaly"] = anomaly_signals.get(s, 0.0)
            self.history[s]["prev_critical_prob"] = posteriors.get(s, {}).get("Critical", 0.0)
            
        # 4. Rank candidates to find the most plausible root cause
        ranked = sorted(scores.items(), key=lambda item: item[1]["root_cause_score"], reverse=True)
        
        if ranked:
            best_service = ranked[0][0]
            best_score = ranked[0][1]["root_cause_score"]
            intrinsic = ranked[0][1]["intrinsic_evidence"]
            
            # Ensure there's actually a problem before declaring a root cause
            if best_score > 0 and intrinsic > 0.3:
                root_cause = best_service
            else:
                root_cause = "None"
        else:
            root_cause = "None"
            
        return {
            "root_cause": root_cause,
            "scores": scores,
            "ranked": ranked
        }
        
    def _calculate_scores(self, s: str, anomaly_signals: Dict[str, float], posteriors: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        p_crit = posteriors.get(s, {}).get("Critical", 0.0)
        p_deg = posteriors.get(s, {}).get("Degrading", 0.0)
        anomaly = anomaly_signals.get(s, 0.0)
        
        # Intrinsic Evidence: how strongly its own telemetry indicates it is unhealthy
        intrinsic_evidence = (p_crit * 2.0) + (p_deg * 1.0) + (min(anomaly, 10.0) / 5.0)
        
        upstream_causal_evidence = 0.0
        victim_evidence = 0.0
        
        s_start = self.history[s]["degradation_start_tick"]
        
        # Upstream Causal Evidence: s is parent, looking at children (downstream)
        for child in self.graph.successors(s):
            c_crit = posteriors.get(child, {}).get("Critical", 0.0)
            c_deg = posteriors.get(child, {}).get("Degrading", 0.0)
            c_unhealthy = c_crit + c_deg
            
            if c_unhealthy > 0.3:
                c_start = self.history[child]["degradation_start_tick"]
                
                # Backpressure / edge evidence
                edge_data = self.graph.get_edge_data(s, child)
                reqs = edge_data.get("request_count", 0) if edge_data else 0
                backpressure_bonus = 0.0
                if reqs > 0:
                    backpressure_bonus = np.log1p(reqs) * 0.1
                
                # Temporal relationship: parent must degrade before or at same time as child
                if s_start != -1 and c_start != -1 and s_start <= c_start:
                    upstream_causal_evidence += c_unhealthy * (1.0 + backpressure_bonus)
                elif s_start != -1 and c_start == -1:
                    upstream_causal_evidence += c_unhealthy * 0.5
                    
        # Victim Evidence: s is child, looking at parents (upstream)
        for parent in self.graph.predecessors(s):
            p_crit = posteriors.get(parent, {}).get("Critical", 0.0)
            p_deg = posteriors.get(parent, {}).get("Degrading", 0.0)
            p_unhealthy = p_crit + p_deg
            
            if p_unhealthy > 0.3:
                p_start = self.history[parent]["degradation_start_tick"]
                
                # Temporal relationship: upstream parent degraded before or at same time as s
                if p_start != -1 and (s_start == -1 or p_start <= s_start):
                    victim_evidence += p_unhealthy * 2.5
                    
        # Final root cause score
        rc_score = intrinsic_evidence + upstream_causal_evidence - victim_evidence
        
        # Classify the service
        if victim_evidence > 1.0 and victim_evidence > upstream_causal_evidence * 0.5:
            classification = "PROPAGATED_VICTIM"
        elif rc_score > 0 and intrinsic_evidence > 0.4 and rc_score >= intrinsic_evidence * 0.5:
            classification = "ROOT_CAUSE"
        else:
            classification = "NORMAL"
            
        return {
            "intrinsic_evidence": intrinsic_evidence,
            "upstream_causal_evidence": upstream_causal_evidence,
            "victim_evidence": victim_evidence,
            "root_cause_score": rc_score,
            "classification": classification
        }
