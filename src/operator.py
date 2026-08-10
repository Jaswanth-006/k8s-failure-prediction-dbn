import kopf
import logging
import asyncio
import time
from kubernetes import client, config
from prometheus_client import start_http_server

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference_adapter import InferenceAdapter
from src.decision_policy import DecisionPolicy
from src.action_executor import ActionExecutor
from src.audit_logger import AuditLogger
from src.metrics import export_decision_metrics

# Global state objects
inference_adapter = None
decision_policy = None
action_executor = None
audit_logger = None

@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    # Operator settings
    settings.execution.max_workers = 5
    settings.networking.request_timeout = 10
    
    # Leader Election
    settings.peering.name = "failurepredictor"
    settings.peering.mandatory = True
    
    global inference_adapter, decision_policy, action_executor, audit_logger
    
    logging.info("Initializing Phase 4 Operator Components...")
    
    # These persist across timer ticks
    inference_adapter = InferenceAdapter(prometheus_url="http://localhost:9090")
    decision_policy = DecisionPolicy()
    action_executor = ActionExecutor()
    audit_logger = AuditLogger()
    
    # Start Prometheus HTTP server
    start_http_server(8000)
    logging.info("Prometheus metrics server started on port 8000")
    
    logging.info("Operator Components Initialized.")

@kopf.on.create('preface.example.com', 'v1alpha1', 'failurepredictors')
@kopf.on.update('preface.example.com', 'v1alpha1', 'failurepredictors')
def handle_config_update(spec, name, namespace, logger, **kwargs):
    logger.info(f"Loaded configuration for {name}")
    # We don't store it globally; we read it from the object in the timer

# Use a threading lock or rely on kopf's timer mechanism (it does not overlap by default)
@kopf.timer('preface.example.com', 'v1alpha1', 'failurepredictors', interval=5.0, sharp=True)
def inference_tick(spec, status, patch, name, namespace, logger, **kwargs):
    """
    Periodic trigger that runs every 5 seconds.
    Reads spec, runs inference, makes a decision, and patches status.
    """
    global inference_adapter, decision_policy, action_executor, audit_logger
    
    tick_start_time = time.time()
    tick_status = "success"

    # 1. State Restoration (Checkpointing)
    if not getattr(decision_policy, 'is_restored', False):
        checkpoint = status.get("checkpoint")
        if checkpoint:
            logger.info("Restoring DecisionPolicy state from CRD checkpoint.")
            decision_policy.import_state(checkpoint)
        decision_policy.is_restored = True

    # Parse enabled actions from spec
    actions_spec = spec.get("actions", {})
    enabled_actions = ["Do_Nothing"]
    if actions_spec.get("reschedule", {}).get("enabled", True):
        enabled_actions.append("Reschedule_Pod")
    if actions_spec.get("restart", {}).get("enabled", True):
        enabled_actions.append("Restart_Pod")
    if actions_spec.get("scale", {}).get("enabled", True):
        enabled_actions.append("Scale_Out")
    if actions_spec.get("traffic", {}).get("enabled", True):
        enabled_actions.append("Traffic_Shift")

    # Default Config
    config_dict = {
        "debounce_ticks": spec.get("debounce", {}).get("ticks", 11),
        "cooldown_seconds": spec.get("cooldown", {}).get("seconds", 300),
        "threshold_risk": spec.get("threshold", {}).get("risk", 0.95),
        "shadow_mode": spec.get("policy", {}).get("shadowMode", True),
        "enabled_actions": enabled_actions
    }
    
    decision = {}
    rate_limit_len = 0

    try:
        # 2. Run inference
        ddn_output = inference_adapter.run_tick()
        
        # 3. Evaluate MEU and Persistence
        decision = decision_policy.evaluate(ddn_output, config_dict)
        
        state = decision.get("state", "HEALTHY")
        action_name = decision.get("action", "Do_Nothing")
        root_cause = ddn_output.get("root_cause", "None")
        
        # Calculate rate limit usage for metrics
        if root_cause != "None":
            # Just count the items safely since it's pruned in evaluate()
            rate_limit_len = len(decision_policy.rate_limit_history.get(root_cause, []))
        
        # Audit Logging for the Action Boundary
        if state in ["INTERVENE", "COOLDOWN", "RATE_LIMITED"]:
            audit_logger.log_decision(name, namespace, decision, config_dict["shadow_mode"])
        
        # 4. Execute action (if any)
        if state == "INTERVENE" and action_name != "Do_Nothing":
            action_executor.execute(action_name, root_cause, shadow_mode=config_dict["shadow_mode"])
            decision_policy.record_action(root_cause)
            
            # Record action in status
            recent = status.get("recentActions", [])
            reason_str = "MEU Selected"
            recent.insert(0, f"[{root_cause}] {action_name} - {reason_str}")
            patch.status["recentActions"] = recent[:5] # Keep last 5

        # 5. Patch CRD Status (New Structured Format)
        posteriors = ddn_output.get("posteriors", {})
        risk = posteriors.get(root_cause, {}).get("Critical", 0.0) if root_cause != "None" else 0.0

        patch.status["risk"] = {
            "current": risk,
            "criticalProbability": risk
        }
        patch.status["rootCause"] = {
            "service": root_cause
        }
        patch.status["decision"] = {
            "selectedAction": decision.get("best_action", "Do_Nothing"),
            "deltaEu": decision.get("delta_eu", 0.0),
            "interventionEligible": state == "INTERVENE"
        }
        patch.status["persistence"] = {
            "currentTicks": decision.get("persistence_count", 0),
            "requiredTicks": config_dict["debounce_ticks"]
        }
        patch.status["safety"] = {
            "shadowMode": config_dict["shadow_mode"],
            "cooldownRemaining": decision.get("cooldown_remaining", 0.0),
            "rateLimitState": rate_limit_len
        }
        patch.status["health"] = {
            "lastSuccessfulTick": time.time(),
            "lastError": "",
            "inferenceLatency": time.time() - tick_start_time
        }
        
        # Save checkpoint
        patch.status["checkpoint"] = decision_policy.export_state()

    except Exception as e:
        logger.error(f"Inference tick failed: {e}")
        tick_status = "error"
        patch.status["health"] = {
            "lastError": str(e),
            "inferenceLatency": time.time() - tick_start_time
        }
        # Allow recovery on next tick
    finally:
        # 6. Export Metrics (Independent of whether it failed or not)
        elapsed_time = time.time() - tick_start_time
        export_decision_metrics(decision, config_dict["shadow_mode"], rate_limit_len, elapsed_time, tick_status)
