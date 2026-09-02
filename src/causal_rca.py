"""
PREFACE-DBN Goal 3 + Goal 4: Directional Causality and Multi-Signal RCA
=========================================================================

Goal 3 (existing, unchanged in principle):
  Directional root-cause analysis using temporal precedence and the
  directed service-dependency graph.

Goal 4 (new, backward-compatible extension):
  Optional enrichment of RCA evidence with multi-signal telemetry:
    - Service-level: memory pressure, error rate
    - Edge-level:    dependency error rate, latency P95

The step() method signature is extended with two optional keyword arguments:

  step(anomaly_signals, posteriors,
       edge_telemetry=None,      # Dict[(src,dst), EdgeTelemetry] or None
       service_telemetry=None)   # Dict[service, ServiceTelemetry] or None

When both are None (the default), Goal 3 behavior is identical to before.
The Goal 3 regression tests (scripts/27_test_goal3_causality.py) pass without
any changes.

Scoring design
--------------
IntrinsicEvidence (per service):
  = (p_crit × 2.0) + (p_deg × 1.0) + clip(anomaly,10)/5
    + WEIGHT_MEMORY × norm(memory_pressure)   [if service_telemetry available]
    + WEIGHT_ERROR  × norm(error_rate)         [if service_telemetry available]

  When service_telemetry is available, the intrinsic evidence is SCALED by a
  physical_confidence factor derived from actual telemetry. This prevents the
  DBN posterior (which can saturate at Critical for any anomaly > 7) from
  dominating when real telemetry shows the service is healthy.

UpstreamCausalEvidence (per service s, summed over children):
  = Σ child_unhealthy_prob
      × temporal_weight
      × (base_evidence_from_edge_stress + dependency_pressure)

  base_evidence is gated by actual edge stress so that a healthy edge
  (zero error rate, low latency) cannot transmit causal blame from parent
  to child.

VictimEvidence (per service s, summed over parents):
  = Σ parent_unhealthy_prob × gated_victim_weight
    [gated by actual edge stress from parent → s]

RootCauseScore = IntrinsicEvidence + UpstreamCausalEvidence - VictimEvidence

Root-cause declaration requires BOTH:
  1. best_score > 0 AND intrinsic_evidence > INTRINSIC_THRESHOLD
  2. If service_telemetry is available: physical_stress > PHYSICAL_STRESS_MIN
     (at least some direct physical evidence of the problem)

Evidence weights (explicit named constants)
-------------------------------------------
WEIGHT_MEMORY       = 0.3  — memory pressure adds supporting intrinsic evidence
WEIGHT_SVC_ERROR    = 0.4  — service error rate is a strong intrinsic signal
WEIGHT_EDGE_ERROR   = 0.3  — edge error rate increases dependency pressure
WEIGHT_EDGE_LATENCY = 0.2  — edge latency P95 increases dependency pressure

INTRINSIC_THRESHOLD  = 0.5   — minimum intrinsic score to declare root cause
PHYSICAL_STRESS_MIN  = 0.05  — minimum physical stress to declare root cause when telemetry available
TELEMETRY_CONFIDENCE_WEIGHT = 0.7  — how much physical telemetry modulates the anomaly-posterior score
"""

import networkx as nx
from typing import Dict, Any, Optional
import numpy as np

# Explicit enrichment weights (Goal 4). Documented above.
WEIGHT_MEMORY       = 0.3
WEIGHT_SVC_ERROR    = 0.4
WEIGHT_EDGE_ERROR   = 0.3
WEIGHT_EDGE_LATENCY = 0.2

# Root-cause declaration thresholds
INTRINSIC_THRESHOLD         = 0.5   # minimum intrinsic evidence to declare root cause
PHYSICAL_STRESS_MIN         = 0.05  # minimum physical stress required when telemetry available
TELEMETRY_CONFIDENCE_WEIGHT = 0.7   # weight of physical telemetry vs anomaly-posterior in intrinsic


class DirectionalCausalAnalyzer:
    """
    Goal 3 + Goal 4: Directional Causality and Backpressure-Aware Root Cause Analysis

    This module analyzes a directed service graph (A → B means A calls B).
    It computes a causal root cause score based on:
    - IntrinsicEvidence: the service's own health (anomaly, posterior, +memory/error).
    - UpstreamCausalEvidence: how strongly its downstream dependents' degradations
      are explained by it (enriched with multi-signal dependency pressure).
    - VictimEvidence: how strongly its own degradation is explained by an unhealthy
      upstream parent.

    Goal 4 adds telemetry-confidence scaling:
    - When physical telemetry (CPU, memory, error rate) is available, the intrinsic
      evidence is scaled by a confidence factor derived from actual measurements.
    - This prevents the DBN posterior (which saturates at P(Critical)≈1 for any
      anomaly > ~7) from producing false root causes for healthy services.
    - Edge stress gates both upstream causal evidence and victim evidence so that
      a clean dependency path cannot transmit blame.
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

    def step(
        self,
        anomaly_signals: Dict[str, float],
        posteriors: Dict[str, Dict[str, float]],
        edge_telemetry: Optional[Dict] = None,
        service_telemetry: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process one tick and return RCA results.

        Parameters
        ----------
        anomaly_signals   : service → autoencoder anomaly score
        posteriors        : service → {Normal, Degrading, Critical} probs
        edge_telemetry    : (src, dst) → EdgeTelemetry (Goal 4, optional)
        service_telemetry : service → ServiceTelemetry (Goal 4, optional)

        Returns
        -------
        dict with keys: root_cause, scores, ranked
        """
        self.current_tick += 1

        # 1. Update temporal degradation start times.
        #    We use physical stress (if available) OR posterior to decide if a
        #    service is actually degrading. This prevents the history from being
        #    permanently set during the healthy phase just because the observation
        #    model saturated.
        for s in self.services:
            p_crit = posteriors.get(s, {}).get("Critical", 0.0)
            p_deg  = posteriors.get(s, {}).get("Degrading", 0.0)
            unhealthy_prob = p_crit + p_deg
            anomaly = anomaly_signals.get(s, 0.0)
            print(
                f"[RCA DEBUG] {s}: "
                f"anomaly={anomaly:.3f}, "
                f"p_crit={p_crit:.3f}, "
                f"p_deg={p_deg:.3f}"
            )

            # Goal 4: When physical telemetry is available, require BOTH
            # posterior unhealthy AND physical stress to set degradation history.
            # This prevents the history being set during healthy phases when
            # the DDN posterior saturates due to observation model mismatch.
            actually_degrading = False
            if service_telemetry is not None:
                svc_tel = service_telemetry.get(s)
                if svc_tel is not None and svc_tel.available:
                    norm = svc_tel.normalized()
                    physical_stress = norm["cpu_rate"] + norm["memory_bytes"] + norm["error_rate"]
                    # Require posterior AND physical evidence
                    actually_degrading = (unhealthy_prob > 0.4) and (physical_stress > PHYSICAL_STRESS_MIN)
                else:
                    actually_degrading = (unhealthy_prob > 0.4)
            else:
                # Goal 3 path: use posterior only
                actually_degrading = (unhealthy_prob > 0.4)

            if actually_degrading:
                if self.history[s]["degradation_start_tick"] == -1:
                    self.history[s]["degradation_start_tick"] = self.current_tick
            else:
                self.history[s]["degradation_start_tick"] = -1

        # 2. Calculate scores for all services
        scores = {}
        for s in self.services:
            scores[s] = self._calculate_scores(
                s,
                anomaly_signals,
                posteriors,
                edge_telemetry=edge_telemetry,
                service_telemetry=service_telemetry,
            )

        # 3. Update history for the next tick
        for s in self.services:
            self.history[s]["prev_anomaly"]       = anomaly_signals.get(s, 0.0)
            self.history[s]["prev_critical_prob"]  = posteriors.get(s, {}).get("Critical", 0.0)

        # 4. Rank candidates to find the most plausible root cause
        ranked = sorted(
            scores.items(),
            key=lambda item: item[1]["root_cause_score"],
            reverse=True,
        )

        root_cause = "None"
        if ranked:
            best_service = ranked[0][0]
            best_score   = ranked[0][1]["root_cause_score"]
            intrinsic    = ranked[0][1]["intrinsic_evidence"]
            physical_stress = ranked[0][1].get("physical_stress", -1.0)

            # Ensure there's actually a problem before declaring a root cause.
            # Goal 4 requires physical evidence when telemetry is available.
            score_ok = (best_score > 0 and intrinsic > INTRINSIC_THRESHOLD)
            if score_ok:
                if service_telemetry is not None and physical_stress >= 0:
                    # Physical evidence required: the service must show actual stress
                    if physical_stress > PHYSICAL_STRESS_MIN:
                        root_cause = best_service
                    # else: anomaly without physical backing → no root cause declared
                else:
                    # Goal 3 path (no telemetry): use score threshold only
                    root_cause = best_service

        return {
            "root_cause": root_cause,
            "scores": scores,
            "ranked": ranked,
        }

    # ------------------------------------------------------------------
    # Private scoring
    # ------------------------------------------------------------------

    def _calculate_scores(
        self,
        s: str,
        anomaly_signals: Dict[str, float],
        posteriors: Dict[str, Dict[str, float]],
        edge_telemetry: Optional[Dict],
        service_telemetry: Optional[Dict],
    ) -> Dict[str, Any]:

        p_crit  = posteriors.get(s, {}).get("Critical", 0.0)
        p_deg   = posteriors.get(s, {}).get("Degrading", 0.0)
        anomaly = anomaly_signals.get(s, 0.0)

        # --- INTRINSIC EVIDENCE (Goal 3 core) ---
        anomaly_posterior_score = (
            (p_crit * 2.0)
            + (p_deg * 1.0)
            + (min(anomaly, 10.0) / 5.0)
        )
        intrinsic_evidence = anomaly_posterior_score
        physical_stress = -1.0  # sentinel: no telemetry

        # --- INTRINSIC ENRICHMENT (Goal 4 service-level telemetry) ---
        if service_telemetry is not None:
            svc_tel = service_telemetry.get(s)
            if svc_tel is not None and svc_tel.available:
                norm = svc_tel.normalized()
                physical_stress = norm["cpu_rate"] + norm["memory_bytes"] + norm["error_rate"]

                # Physical signal contribution (direct, not affected by posterior saturation)
                physical_intrinsic = (
                    WEIGHT_MEMORY    * norm["memory_bytes"]
                    + WEIGHT_SVC_ERROR * norm["error_rate"]
                )

                # Telemetry-confidence blending:
                # When physical telemetry is available, the intrinsic evidence is a
                # weighted combination of:
                #   - anomaly_posterior_score: captures CPU anomaly via autoencoder + DBN
                #   - physical_intrinsic: captures memory + error rate directly
                #
                # The blending prevents the observation model saturation problem:
                # if a healthy service produces anomaly=7 (causing P(Critical)≈1),
                # but physical telemetry shows cpu=0.05, mem=50MB, error=0%,
                # then physical_intrinsic ≈ 0 pulls the blended score toward 0.
                #
                # Blend formula:
                #   intrinsic = (1 - w) * anomaly_posterior + w * physical_intrinsic_scaled
                # where w = TELEMETRY_CONFIDENCE_WEIGHT
                #
                # physical_intrinsic_scaled is multiplied by the max anomaly_posterior_score
                # so the two components live on the same scale.
                max_ap_score = 2.0 + 1.0 + 2.0  # p_crit=1 → 2.0, p_deg=1 → 1.0, anomaly=10 → 2.0
                physical_scaled = physical_intrinsic * (max_ap_score / max(0.7, 0.7))  # normalized ref

                intrinsic_evidence = (
                    (1.0 - TELEMETRY_CONFIDENCE_WEIGHT) * anomaly_posterior_score
                    + TELEMETRY_CONFIDENCE_WEIGHT * physical_scaled
                )

                print(
                    f"[RCA DEBUG] {s}: "
                    f"memory_norm={norm['memory_bytes']:.3f}, "
                    f"error_norm={norm['error_rate']:.3f}, "
                    f"phys_stress={physical_stress:.3f}, "
                    f"intrinsic={intrinsic_evidence:.3f}"
                )

        upstream_causal_evidence = 0.0
        victim_evidence          = 0.0

        s_start = self.history[s]["degradation_start_tick"]

        # --- UPSTREAM CAUSAL EVIDENCE (Goal 3 + Goal 4 enrichment) ---
        # Goal 4: When service_telemetry is available, the parent must also show
        # physical stress to accumulate upstream causal evidence. This prevents
        # a healthy hub service (e.g. ts-ui-dashboard) from accumulating causal
        # blame purely because the DDN posterior saturated at Critical for all
        # services (due to high anomaly scores across the board).
        parent_has_physical_stress = True  # default: assume stressed (Goal 3 path)
        if service_telemetry is not None:
            s_tel = service_telemetry.get(s)
            if s_tel is not None and s_tel.available:
                s_norm = s_tel.normalized()
                s_physical_stress = s_norm["cpu_rate"] + s_norm["memory_bytes"] + s_norm["error_rate"]
                parent_has_physical_stress = (s_physical_stress > PHYSICAL_STRESS_MIN)

        for child in self.graph.successors(s):
            c_crit    = posteriors.get(child, {}).get("Critical", 0.0)
            c_deg     = posteriors.get(child, {}).get("Degrading", 0.0)
            c_unhealthy = c_crit + c_deg

            if c_unhealthy > 0.3:
                c_start = self.history[child]["degradation_start_tick"]

                # Compute dependency pressure (backpressure bonus)
                pressure = self._dependency_pressure(s, child, edge_telemetry)

                # Goal 4: Gate upstream evidence by BOTH parent physical stress
                # AND edge stress (error rate + latency only — request_rate alone
                # does not indicate an unhealthy edge).
                if edge_telemetry is not None and (s, child) in edge_telemetry:
                    et = edge_telemetry[(s, child)]
                    norm_et = et.normalized()
                    # Only error_rate and latency signal actual edge health problems.
                    # request_rate is excluded: normal traffic on a healthy edge
                    # should NOT generate upstream causal credit.
                    edge_stress = (
                        norm_et["error_rate"]
                        + norm_et.get("latency_p95", 0.0)
                    )
                    # base_evidence scales from 0 (clean edge) to 1.0 (full stress)
                    base_evidence = min(1.0, edge_stress * 3.0)
                    # If parent shows no physical stress: further suppress (it's a hub victim)
                    if not parent_has_physical_stress:
                        base_evidence *= 0.2
                else:
                    # Goal 3 path: full evidence (no edge telemetry to gate with)
                    base_evidence = 1.0

                # Temporal relationship: parent must degrade before or at same time as child
                if s_start != -1 and c_start != -1 and s_start <= c_start:
                    upstream_causal_evidence += c_unhealthy * (base_evidence + pressure)
                elif s_start != -1 and c_start == -1:
                    upstream_causal_evidence += c_unhealthy * (base_evidence * 0.5)

        # --- VICTIM EVIDENCE (Goal 3, unchanged base, Goal 4 gated) ---
        for parent in self.graph.predecessors(s):
            par_crit    = posteriors.get(parent, {}).get("Critical", 0.0)
            par_deg     = posteriors.get(parent, {}).get("Degrading", 0.0)
            p_unhealthy = par_crit + par_deg

            if p_unhealthy > 0.3:
                p_start = self.history[parent]["degradation_start_tick"]

                # Goal 4: Gate victim evidence by edge stress (error + latency only).
                # A healthy edge cannot propagate victim status.
                if edge_telemetry is not None and (parent, s) in edge_telemetry:
                    et = edge_telemetry[(parent, s)]
                    norm_et = et.normalized()
                    edge_stress = (
                        norm_et["error_rate"]
                        + norm_et.get("latency_p95", 0.0)
                    )
                    base_victim = min(2.5, edge_stress * 5.0)
                else:
                    # Goal 3 path: full victim weight (no edge telemetry)
                    base_victim = 2.5

                if p_start != -1 and (s_start == -1 or p_start <= s_start):
                    victim_evidence += p_unhealthy * base_victim

        # Final root cause score
        rc_score = intrinsic_evidence + upstream_causal_evidence - victim_evidence

        # Classify the service
        if victim_evidence > 1.0 and victim_evidence > upstream_causal_evidence * 0.5:
            classification = "PROPAGATED_VICTIM"
        elif rc_score > 0 and intrinsic_evidence > INTRINSIC_THRESHOLD and rc_score >= intrinsic_evidence * 0.5:
            classification = "ROOT_CAUSE"
        else:
            classification = "NORMAL"

        return {
            "intrinsic_evidence":       intrinsic_evidence,
            "upstream_causal_evidence": upstream_causal_evidence,
            "victim_evidence":          victim_evidence,
            "root_cause_score":         rc_score,
            "classification":           classification,
            "physical_stress":          physical_stress,
        }

    def _dependency_pressure(
        self,
        source: str,
        destination: str,
        edge_telemetry: Optional[Dict],
    ) -> float:
        """
        Compute the dependency pressure scalar for edge source → destination.

        Goal 3 baseline:
          log1p(request_count) × 0.1

        Goal 4 enrichment (if edge_telemetry is available):
          Uses EdgeTelemetry.dependency_pressure() which adds error_rate and
          latency_p95 components via WEIGHT_EDGE_ERROR and WEIGHT_EDGE_LATENCY.

        If edge_telemetry is None, falls back to the Goal 3 graph edge attribute
        "request_count", preserving exact backward compatibility.
        """
        if edge_telemetry is not None:
            et = edge_telemetry.get((source, destination))
            if et is not None:
                return et.dependency_pressure(
                    weight_error=WEIGHT_EDGE_ERROR,
                    weight_latency=WEIGHT_EDGE_LATENCY,
                )

        # Goal 3 fallback: read request_count from graph edge attributes
        edge_data = self.graph.get_edge_data(source, destination)
        reqs = edge_data.get("request_count", 0) if edge_data else 0
        if reqs > 0:
            return np.log1p(reqs) * 0.1
        return 0.0
