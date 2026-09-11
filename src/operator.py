"""
Kubernetes operator for PREFACE-DBN.

Run with:
    kopf run src/operator.py --verbose

This was previously an empty file, so nothing ever populated the FailurePredictor
CRD's status subresource despite the schema defining it in full. The reconcile
loop below closes that gap: every tick it runs inference, applies the decision
policy, optionally acts, and writes the result back to status so the system's
state is visible through `kubectl get failurepredictor`.

Reconcile loop
--------------
    Prometheus -> Rectifier -> autoencoder -> DBN     (InferenceAdapter)
      -> DecisionPolicy (debounce, cooldown, MEU)
        -> ActionExecutor (two safety gates)
          -> CRD status + audit log

Safety
------
Shadow mode is on unless `spec.policy.shadowMode` is explicitly false, and even
then the executor needs its own `allow_live` gate, which is opened only by
setting PREFACE_ALLOW_LIVE=true in the operator's environment. So a live
mutation requires a deliberate change in two different places: the custom
resource and the deployment environment. Neither alone is enough.

Degradation
-----------
Inference failures (Prometheus down, model missing) never crash the loop. They
are recorded in `status.health.lastError` and the belief state is left untouched,
so a transient scrape failure does not produce a spurious risk score.
"""

import os
import time
from datetime import datetime, timezone

import kopf

from src.action_executor import ActionExecutor, EXECUTED, WOULD_EXECUTE
from src.audit_logger import AuditLogger
from src.decision_policy import DecisionPolicy

GROUP = "preface.example.com"
VERSION = "v1alpha1"
PLURAL = "failurepredictors"

# Seconds between reconcile ticks. Matches the recorder's default cadence and the
# 1-minute scrape interval the pipeline is designed around.
TICK_INTERVAL = float(os.environ.get("PREFACE_TICK_SECONDS", "60"))

# Second safety gate for live mutation; see module docstring.
ALLOW_LIVE = os.environ.get("PREFACE_ALLOW_LIVE", "").lower() in ("1", "true", "yes")

MAX_RECENT_ACTIONS = 10

# Per-resource runtime state, keyed by (namespace, name). Holds the inference
# adapter and decision policy so temporal context survives across ticks.
_state = {}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(namespace, name):
    return "%s/%s" % (namespace, name)


def _config(spec):
    """Read the CRD spec, applying defaults for anything unset."""
    return {
        "risk_threshold": float(spec.get("threshold", {}).get("risk", 0.95)),
        "debounce_ticks": int(spec.get("debounce", {}).get("ticks", 11)),
        "cooldown_seconds": int(spec.get("cooldown", {}).get("seconds", 300)),
        # Shadow unless explicitly disabled - an absent field must never mean live.
        "shadow_mode": spec.get("policy", {}).get("shadowMode", True) is not False,
        "actions": spec.get("actions", {}),
    }


def _action_enabled(cfg, action):
    """Map an action name onto its spec.actions.<x>.enabled flag."""
    mapping = {
        "Restart_Pod": "restart",
        "Reschedule_Pod": "reschedule",
        "Scale_Out": "scale",
    }
    field = mapping.get(action)
    if field is None:
        return False
    return bool(cfg["actions"].get(field, {}).get("enabled", False))


def _ensure_state(namespace, name, cfg, status, logger):
    """Build (or fetch) the runtime state for one FailurePredictor."""
    key = _key(namespace, name)
    if key in _state:
        return _state[key]

    policy = DecisionPolicy(
        debounce_ticks=cfg["debounce_ticks"],
        cooldown_seconds=cfg["cooldown_seconds"],
        threshold_risk=cfg["risk_threshold"],
    )

    # Resume temporal context after an operator restart, so a rescheduled
    # operator does not re-enter an incident with a flat prior.
    checkpoint = (status or {}).get("checkpoint") or {}
    if checkpoint:
        try:
            policy.import_state(checkpoint)
            logger.info("restored decision-policy checkpoint")
        except Exception as exc:
            logger.warning("could not restore checkpoint: %s", exc)

    entry = {
        "policy": policy,
        "executor": ActionExecutor(allow_live=ALLOW_LIVE, namespace=namespace),
        "audit": AuditLogger(),
        "adapter": None,
        "adapter_error": None,
        "recent": list((status or {}).get("recentActions") or []),
    }
    _state[key] = entry
    return entry


def _get_adapter(entry, logger):
    """
    Lazily construct the inference adapter.

    Deferred because it loads a torch model and needs Prometheus; doing it at
    startup would make the operator crash-loop when either is unavailable,
    instead of reporting the problem through status.
    """
    if entry["adapter"] is not None:
        return entry["adapter"]

    from src.inference_adapter import InferenceAdapter

    entry["adapter"] = InferenceAdapter()
    entry["adapter_error"] = None
    logger.info("inference adapter ready")
    return entry["adapter"]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
@kopf.on.create(GROUP, VERSION, PLURAL)
@kopf.on.resume(GROUP, VERSION, PLURAL)
def on_ready(spec, status, name, namespace, patch, logger, **_):
    """Initialise a FailurePredictor and report its starting configuration."""
    cfg = _config(spec)
    _ensure_state(namespace, name, cfg, status, logger)

    logger.info(
        "FailurePredictor %s ready (threshold=%.2f debounce=%d cooldown=%ds shadow=%s)",
        name, cfg["risk_threshold"], cfg["debounce_ticks"],
        cfg["cooldown_seconds"], cfg["shadow_mode"],
    )
    patch.status["safety"] = {
        "shadowMode": cfg["shadow_mode"],
        "cooldownRemaining": 0.0,
        "rateLimitState": 0,
    }
    patch.status["health"] = {"lastError": "", "lastSuccessfulTick": 0.0,
                              "inferenceLatency": 0.0}


@kopf.on.update(GROUP, VERSION, PLURAL, field="spec")
def on_spec_change(spec, name, namespace, logger, **_):
    """Rebuild runtime state so config edits take effect without a restart."""
    cfg = _config(spec)
    key = _key(namespace, name)
    entry = _state.get(key)
    if entry is not None:
        entry["policy"].debounce_ticks = cfg["debounce_ticks"]
        entry["policy"].cooldown_seconds = cfg["cooldown_seconds"]
        entry["policy"].threshold_risk = cfg["risk_threshold"]
    logger.info("spec updated: threshold=%.2f debounce=%d shadow=%s",
                cfg["risk_threshold"], cfg["debounce_ticks"], cfg["shadow_mode"])


@kopf.on.delete(GROUP, VERSION, PLURAL)
def on_delete(name, namespace, logger, **_):
    _state.pop(_key(namespace, name), None)
    logger.info("FailurePredictor %s removed", name)


@kopf.timer(GROUP, VERSION, PLURAL, interval=TICK_INTERVAL, sharp=True)
def reconcile(spec, status, name, namespace, patch, logger, **_):
    """One monitoring tick: infer, decide, maybe act, then publish status."""
    cfg = _config(spec)
    entry = _ensure_state(namespace, name, cfg, status, logger)

    started = time.time()
    try:
        adapter = _get_adapter(entry, logger)
        ddn_output = adapter.run_tick()
    except Exception as exc:
        # Carry the belief state forward untouched rather than inventing an
        # observation; a scrape failure is missing evidence, not good news.
        message = "%s: %s" % (exc.__class__.__name__, exc)
        logger.error("inference failed: %s", message)
        entry["adapter"] = None
        patch.status["health"] = {
            "lastError": message,
            "lastSuccessfulTick": (status or {}).get("health", {}).get("lastSuccessfulTick", 0.0),
            "inferenceLatency": round(time.time() - started, 3),
        }
        return

    latency = time.time() - started

    # evaluate() reads the risk threshold off the instance rather than from
    # config_override, so apply spec changes directly.
    entry["policy"].threshold_risk = cfg["risk_threshold"]
    decision = entry["policy"].evaluate(ddn_output, config_override={
        "debounce_ticks": cfg["debounce_ticks"],
        "cooldown_seconds": cfg["cooldown_seconds"],
    })

    root_cause = ddn_output.get("root_cause", "None")
    action = decision.get("action", "Do_Nothing")
    state = decision.get("state", "HEALTHY")
    # DecisionPolicy states: HEALTHY, PENDING, COOLDOWN, RATE_LIMITED, INTERVENE.
    # Only INTERVENE has cleared debounce, cooldown and rate limiting.
    eligible = state == "INTERVENE" and action != "Do_Nothing"

    result = None
    if eligible:
        if not _action_enabled(cfg, action):
            logger.info("action '%s' disabled in spec.actions; skipping", action)
            result = "DISABLED"
        else:
            result = entry["executor"].execute(
                action, root_cause, shadow_mode=cfg["shadow_mode"]
            )
            if result in (EXECUTED, WOULD_EXECUTE):
                entry["policy"].record_action(root_cause)
            entry["audit"].log_decision(name, namespace, decision, cfg["shadow_mode"])
            entry["recent"].insert(0, "%s %s on %s -> %s"
                                   % (_now(), action, root_cause, result))
            del entry["recent"][MAX_RECENT_ACTIONS:]
            logger.warning("action %s on %s -> %s", action, root_cause, result)

    patch.status["risk"] = {
        "current": float(decision.get("p_crit", 0.0)),
        "criticalProbability": float(decision.get("p_crit", 0.0)),
    }
    patch.status["rootCause"] = {"service": root_cause}
    patch.status["decision"] = {
        "selectedAction": action,
        "deltaEu": float(decision.get("eu_reschedule", 0.0)
                         - decision.get("eu_restart", 0.0)),
        "interventionEligible": bool(eligible),
    }
    patch.status["persistence"] = {
        "currentTicks": int(decision.get("persistence_count", 0)),
        "requiredTicks": cfg["debounce_ticks"],
    }
    patch.status["safety"] = {
        "shadowMode": cfg["shadow_mode"],
        "cooldownRemaining": float(decision.get("cooldown_remaining", 0.0)),
        "rateLimitState": len(entry["recent"]),
    }
    patch.status["health"] = {
        "lastError": "",
        "lastSuccessfulTick": time.time(),
        "inferenceLatency": round(latency, 3),
    }
    patch.status["recentActions"] = entry["recent"]
    # Checkpoint every tick so a rescheduled operator resumes mid-incident.
    patch.status["checkpoint"] = entry["policy"].export_state()

    if latency > 5.0:
        # NFR-1 in the design docs budgets 5s per tick.
        logger.warning("inference took %.1fs, over the 5s budget", latency)
