import json
import logging
from datetime import datetime, timezone
import os

class AuditLogger:
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        # Setup standard logger
        self.logger = logging.getLogger("AuditLogger")
        self.logger.setLevel(logging.INFO)
        
    def log_decision(self, fp_name: str, namespace: str, decision: dict, shadow_mode: bool):
        """
        Logs a structured audit record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Determine execution status
        state = decision.get("state", "UNKNOWN")
        execution_status = "BLOCKED"
        
        if state == "INTERVENE":
            if shadow_mode:
                execution_status = "WOULD_EXECUTE"
            else:
                execution_status = "EXECUTED"
        
        record = {
            "timestamp": timestamp,
            "failure_predictor": f"{namespace}/{fp_name}",
            "risk": decision.get("p_crit", 0.0),
            "root_cause": decision.get("root_cause", "None"),
            "expected_utilities": decision.get("expected_utilities", {}),
            "selected_action": decision.get("action", "Do_Nothing"),
            "delta_eu": decision.get("delta_eu", 0.0),
            "persistence_count": decision.get("persistence_count", 0),
            "decision_state": state,
            "shadow_mode": shadow_mode,
            "execution_status": execution_status,
            "reason": decision.get("blocked_reason", "MEU Selected")
        }
        
        # Write to JSONL
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
            
        # Write to standard logger
        self.logger.info(f"[AUDIT] {execution_status}: {record['selected_action']} on {record['root_cause']} "
                         f"(State: {state}, Reason: {record['reason']}, Shadow: {shadow_mode})")
