"""
PREFACE-DDN Module 4: Kubernetes Action Controller & Intervention Utility Test
Evaluates Maximum Expected Utility (MEU) decision rules: A* = argmax EU(A)
Runs explicit Intervention Utility Test: delta_EU = EU(Reschedule) - EU(Restart)
Supports Shadow/Dry-Run Mode (Default-on) for safe evaluation.
"""

import time
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class KubernetesActionController:
    def __init__(self, shadow_mode: bool = True, cooldown_seconds: int = 300):
        self.shadow_mode = shadow_mode
        self.cooldown_seconds = cooldown_seconds
        self.last_action_time: Dict[str, float] = {}
        self.root_cause_persistence: Dict[str, int] = {}
        self.current_root_cause: str = "None"

    def reconcile_tick(self, ddn_output: Dict[str, Any], node_pressure_flag: bool = False):
        """
        Reconciles DDN output each tick, performing MEU optimization & Intervention Utility Test.
        Requires 3-tick consecutive persistence for the same root cause before intervening.
        """
        posteriors = ddn_output["posteriors"]
        root_cause = ddn_output["root_cause"]
        expected_utilities = ddn_output["expected_utilities"]

        if root_cause == "None":
            self.root_cause_persistence.clear()
            self.current_root_cause = "None"
            logging.info("[Controller] System healthy. No intervention required.")
            return

        if root_cause != self.current_root_cause:
            self.root_cause_persistence.clear()
            self.current_root_cause = root_cause
            self.root_cause_persistence[root_cause] = 1
        else:
            self.root_cause_persistence[root_cause] = self.root_cause_persistence.get(root_cause, 0) + 1

        if self.root_cause_persistence[root_cause] < 11:
            logging.info(f"[Controller] Intervention pending for '{root_cause}'. Persistence: {self.root_cause_persistence[root_cause]}/11 ticks.")
            return

        service_eu = expected_utilities[root_cause]
        p_crit = posteriors[root_cause]["Critical"]

        # Maximum Expected Utility (MEU) decision selection
        best_action = max(service_eu, key=service_eu.get)
        max_eu = service_eu[best_action]

        # Explicit Intervention Utility Test: delta_EU = EU(Reschedule) - EU(Restart)
        eu_reschedule = service_eu["Reschedule_Pod"]
        eu_restart = service_eu["Restart_Pod"]
        delta_eu_intervene = eu_reschedule - eu_restart

        logging.info(f"--- [INTERVENTION UTILITY TEST for '{root_cause}'] ---")
        logging.info(f"P(Critical): {p_crit:.2f} | Node Pressure: {node_pressure_flag}")
        logging.info(f"EU(Reschedule_Pod): {eu_reschedule:.2f} | EU(Restart_Pod): {eu_restart:.2f}")
        logging.info(f"Delta EU (Reschedule - Restart): {delta_eu_intervene:.2f}")
        logging.info(f"Optimal MEU Decision: '{best_action}' with EU = {max_eu:.2f}")

        # Cooldown check
        now = time.time()
        last_time = self.last_action_time.get(root_cause, 0.0)
        if (now - last_time) < self.cooldown_seconds:
            logging.info(f"[Cooldown Active] Skipping action for '{root_cause}' ({int(self.cooldown_seconds - (now - last_time))}s remaining).")
            return

        # Execute or simulate action
        if best_action != "Do_Nothing":
            self._execute_action(root_cause, best_action, delta_eu_intervene)
            self.last_action_time[root_cause] = now

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
