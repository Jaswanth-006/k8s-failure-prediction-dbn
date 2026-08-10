"""
PREFACE-DDN Module 4: Kubernetes Action Controller & Intervention Utility Test
Evaluates Maximum Expected Utility (MEU) decision rules: A* = argmax EU(A)
Runs explicit Intervention Utility Test: delta_EU = EU(Reschedule) - EU(Restart)
Supports Shadow/Dry-Run Mode (Default-on) for safe evaluation.
"""

import time
import logging
from typing import Dict, Any

try:
    from src.decision_policy import DecisionPolicy
except ImportError:
    from decision_policy import DecisionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class KubernetesActionController:
    def __init__(self, shadow_mode: bool = True, cooldown_seconds: int = 300):
        self.shadow_mode = shadow_mode
        self.cooldown_seconds = cooldown_seconds
        self.decision_policy = DecisionPolicy(debounce_ticks=11, cooldown_seconds=cooldown_seconds)

    def reconcile_tick(self, ddn_output: Dict[str, Any], node_pressure_flag: bool = False):
        """
        Reconciles DDN output each tick, performing MEU optimization & Intervention Utility Test.
        Requires 11-tick consecutive persistence for the same root cause before intervening.
        """
        root_cause = ddn_output.get("root_cause", "None")
        
        decision = self.decision_policy.evaluate(ddn_output)
        
        if decision["state"] == "HEALTHY":
            logging.info("[Controller] System healthy. No intervention required.")
            return
            
        if decision["state"] == "PENDING":
            logging.info(f"[Controller] Intervention pending for '{root_cause}'. Persistence: {decision['persistence_count']}/11 ticks.")
            return

        p_crit = decision["p_crit"]
        eu_reschedule = decision["eu_reschedule"]
        eu_restart = decision["eu_restart"]
        delta_eu_intervene = eu_reschedule - eu_restart
        best_action = decision["best_action"]
        max_eu = decision["max_eu"]

        logging.info(f"--- [INTERVENTION UTILITY TEST for '{root_cause}'] ---")
        logging.info(f"P(Critical): {p_crit:.2f} | Node Pressure: {node_pressure_flag}")
        logging.info(f"EU(Reschedule_Pod): {eu_reschedule:.2f} | EU(Restart_Pod): {eu_restart:.2f}")
        logging.info(f"Delta EU (Reschedule - Restart): {delta_eu_intervene:.2f}")
        logging.info(f"Optimal MEU Decision: '{best_action}' with EU = {max_eu:.2f}")

        if decision["state"] == "COOLDOWN":
            logging.info(f"[Cooldown Active] Skipping action for '{root_cause}' ({int(decision['cooldown_remaining'])}s remaining).")
            return

        if decision["action"] != "Do_Nothing":
            self._execute_action(root_cause, decision["action"], delta_eu_intervene)
            self.decision_policy.record_action(root_cause)

    def _execute_action(self, service_name: str, action: str, delta_eu: float):
        if self.shadow_mode:
            logging.info(f"[SHADOW MODE DRY-RUN] Would execute '{action}' on microservice '{service_name}' (Delta EU = {delta_eu:.2f}). No real K8s modification made.")
        else:
            logging.warning(f"[LIVE ACTION EXECUTED] Executing '{action}' on microservice '{service_name}' via Kubernetes API!")
            if action == "Reschedule_Pod":
                self._k8s_reschedule_pod(service_name)
            elif action == "Restart_Pod":
                self._k8s_restart_pod(service_name)
            elif action == "Scale_Out":
                self._k8s_scale_out(service_name)
            elif action == "Traffic_Shift":
                self._k8s_traffic_shift(service_name)

    def _k8s_reschedule_pod(self, service_name: str):
        """Simulates/calls K8s pod eviction to relocate pod to another node."""
        logging.info(f"kubectl evict pod -l app={service_name} (relocating to clean node)")

    def _k8s_restart_pod(self, service_name: str):
        """Simulates/calls K8s deployment rollout restart."""
        logging.info(f"kubectl rollout restart deployment/{service_name}")

    def _k8s_scale_out(self, service_name: str):
        """Simulates/calls K8s replica scaling."""
        logging.info(f"kubectl scale deployment/{service_name} --replicas+=2")

    def _k8s_traffic_shift(self, service_name: str):
        """Simulates/calls Istio VirtualService traffic shift."""
        logging.info(f"istioctl apply -f traffic-shift-{service_name}.yaml")
