import time
from typing import Dict, Any, List

class DecisionPolicy:
    """
    Centralized decision policy evaluating DDN output.
    Enforces temporal persistence (debounce), cooldowns, Maximum Expected Utility (MEU) selection,
    and a rate limiter safety rail.
    """
    def __init__(self, debounce_ticks: int = 11, cooldown_seconds: int = 300, threshold_risk: float = 0.95):
        self.debounce_ticks = debounce_ticks
        self.cooldown_seconds = cooldown_seconds
        self.threshold_risk = threshold_risk
        
        self.root_cause_persistence: Dict[str, int] = {}
        self.current_root_cause: str = "None"
        self.last_action_time: Dict[str, float] = {}
        
        # target -> list of timestamps
        self.rate_limit_history: Dict[str, List[float]] = {}

    def export_state(self) -> Dict[str, Any]:
        """Exports the internal state for CRD checkpointing."""
        return {
            "root_cause_persistence": self.root_cause_persistence,
            "current_root_cause": self.current_root_cause,
            "last_action_time": self.last_action_time,
            "rate_limit_history": self.rate_limit_history
        }

    def import_state(self, state: Dict[str, Any]):
        """Imports the internal state from CRD checkpointing."""
        if not state:
            return
        self.root_cause_persistence = state.get("root_cause_persistence", {})
        self.current_root_cause = state.get("current_root_cause", "None")
        self.last_action_time = state.get("last_action_time", {})
        self.rate_limit_history = state.get("rate_limit_history", {})

    def evaluate(self, ddn_output: Dict[str, Any], config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluates the current tick and returns a state dictionary containing the evaluation result.
        Returns:
            - state: "HEALTHY", "PENDING", "COOLDOWN", "RATE_LIMITED", "INTERVENE"
            - max_eu, best_action, p_crit, expected_utilities, delta_eu
        """
        posteriors = ddn_output.get("posteriors", {})
        root_cause = ddn_output.get("root_cause", "None")
        expected_utilities = ddn_output.get("expected_utilities", {})
        
        # Support dynamic config overrides from Phase 4 CRDs
        debounce = self.debounce_ticks
        cooldown = self.cooldown_seconds
        rate_limit_max = 3
        rate_limit_window = 3600
        
        if config_override:
            debounce = config_override.get("debounce_ticks", debounce)
            cooldown = config_override.get("cooldown_seconds", cooldown)
            rate_limit_max = config_override.get("rate_limit_max", rate_limit_max)
            rate_limit_window = config_override.get("rate_limit_window", rate_limit_window)

        result = {
            "action": "Do_Nothing",
            "state": "HEALTHY",
            "root_cause": root_cause,
            "persistence_count": 0,
            "blocked_reason": "None"
        }

        if root_cause == "None":
            self.root_cause_persistence.clear()
            self.current_root_cause = "None"
            return result

        if root_cause != self.current_root_cause:
            self.root_cause_persistence.clear()
            self.current_root_cause = root_cause
            self.root_cause_persistence[root_cause] = 1
        else:
            self.root_cause_persistence[root_cause] = self.root_cause_persistence.get(root_cause, 0) + 1

        persistence_count = self.root_cause_persistence[root_cause]
        result["persistence_count"] = persistence_count

        if persistence_count < debounce:
            result["state"] = "PENDING"
            return result

        # Persistence is met.
        service_eu = expected_utilities.get(root_cause, {})
        p_crit = posteriors.get(root_cause, {}).get("Critical", 0.0)
        
        if not service_eu:
            return result
            
        # 1. Determine Enabled Actions
        enabled_actions = ["Do_Nothing", "Reschedule_Pod", "Restart_Pod", "Scale_Out", "Traffic_Shift"]
        if config_override and "enabled_actions" in config_override:
            enabled_actions = config_override["enabled_actions"]
        
        # Always ensure Do_Nothing is present as a fallback
        if "Do_Nothing" not in enabled_actions:
            enabled_actions.append("Do_Nothing")

        # 2. Filter Utilities
        valid_eu = {a: service_eu.get(a, 0.0) for a in enabled_actions}

        # 3. Maximum Expected Utility (MEU) with Tie-Breaking
        precedence = ["Reschedule_Pod", "Restart_Pod", "Scale_Out", "Traffic_Shift", "Do_Nothing"]
        max_eu = max(valid_eu.values())
        best_actions = [a for a, eu in valid_eu.items() if eu == max_eu]
        
        best_action = "Do_Nothing"
        for p in precedence:
            if p in best_actions:
                best_action = p
                break
                
        # 4. Explicit Delta EU (Intervention Utility Test)
        eu_reschedule = service_eu.get("Reschedule_Pod", 0.0)
        eu_restart = service_eu.get("Restart_Pod", 0.0)
        delta_eu = eu_reschedule - eu_restart
        
        result.update({
            "eu_reschedule": eu_reschedule,
            "eu_restart": eu_restart,
            "delta_eu": delta_eu,
            "max_eu": max_eu,
            "p_crit": p_crit,
            "best_action": best_action,
            "enabled_actions": enabled_actions,
            "expected_utilities": service_eu
        })

        if best_action == "Do_Nothing":
            result["state"] = "HEALTHY"
            return result

        # Safety Check 1: Cooldown
        now = time.time()
        last_time = self.last_action_time.get(root_cause, 0.0)
        remaining = cooldown - (now - last_time)
        
        if remaining > 0:
            result["state"] = "COOLDOWN"
            result["cooldown_remaining"] = remaining
            result["blocked_reason"] = "Cooldown Active"
            return result
            
        # Safety Check 2: Rate Limiter
        history = self.rate_limit_history.get(root_cause, [])
        # Filter history to only include events within the window
        recent_history = [t for t in history if now - t <= rate_limit_window]
        self.rate_limit_history[root_cause] = recent_history # Clean up old state
        
        if len(recent_history) >= rate_limit_max:
            result["state"] = "RATE_LIMITED"
            result["blocked_reason"] = f"Rate Limit Exceeded ({len(recent_history)}/{rate_limit_max} in {rate_limit_window}s)"
            return result
            
        result["state"] = "INTERVENE"
        result["action"] = best_action
        return result

    def record_action(self, target: str):
        """Records that an action was executed to start the cooldown timer and update rate limit history."""
        now = time.time()
        self.last_action_time[target] = now
        
        history = self.rate_limit_history.get(target, [])
        history.append(now)
        self.rate_limit_history[target] = history
