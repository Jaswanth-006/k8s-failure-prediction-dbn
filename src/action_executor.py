"""
Executes mitigation actions against Kubernetes.

Safety model
------------
Live cluster mutation requires TWO independent gates:

  1. ActionExecutor(allow_live=True)   - set once, deliberately, at construction
  2. execute(..., shadow_mode=False)   - passed per call

Both must be open. A single flag flipped by accident, or a config default
changed somewhere upstream, cannot by itself cause a real mutation. With either
gate closed the action is logged and skipped.

This replaces the previous arrangement, where the live branch was unreachable
(an unconditional `return "BLOCKED"` sat above commented-out code) and the
controller's own fallbacks merely printed the kubectl command they would have
run. "Shadow mode is safe" was therefore trivially true - there was no live path
to guard. One genuinely working action with real gates is worth more than four
that print strings.

Return codes
------------
    WOULD_EXECUTE    shadow mode; nothing touched
    BLOCKED          live requested but allow_live=False
    EXECUTED         the cluster was mutated
    NOT_IMPLEMENTED  no live implementation for this action yet
    FAILED           the API call raised
"""

import logging

from kubernetes import client, config

logger = logging.getLogger(__name__)

WOULD_EXECUTE = "WOULD_EXECUTE"
BLOCKED = "BLOCKED"
EXECUTED = "EXECUTED"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
FAILED = "FAILED"

# Actions with a real implementation below. Anything else returns
# NOT_IMPLEMENTED rather than logging a command and claiming success.
LIVE_ACTIONS = ("Restart_Pod", "Scale_Out")


class ActionExecutor:
    """Applies (or simulates) a mitigation action on one microservice."""

    def __init__(self, allow_live=False, namespace="default", max_replicas=10):
        """
        Parameters
        ----------
        allow_live
            First safety gate. Leave False unless operating on a cluster you are
            willing to have mutated.
        max_replicas
            Ceiling for Scale_Out, so a runaway risk score cannot scale a
            deployment without bound.
        """
        self.allow_live = bool(allow_live)
        self.namespace = namespace
        self.max_replicas = int(max_replicas)

        self._apps = None
        self._core = None
        self._api_error = None
        try:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            self._apps = client.AppsV1Api()
            self._core = client.CoreV1Api()
        except Exception as exc:
            # Not fatal: shadow mode must still work with no cluster present,
            # which is how the evaluation harness runs.
            self._api_error = exc
            logger.warning("[ActionExecutor] no Kubernetes client (%s); "
                           "shadow mode only", exc.__class__.__name__)

    # ------------------------------------------------------------------
    @property
    def live_available(self):
        return self.allow_live and self._apps is not None

    def execute(self, action, target_service, shadow_mode=True):
        """Apply `action` to `target_service`, honouring both safety gates."""
        if shadow_mode:
            logger.info("[SHADOW] would execute '%s' on '%s'; no mutation performed",
                        action, target_service)
            return WOULD_EXECUTE

        if not self.allow_live:
            logger.warning(
                "[BLOCKED] live '%s' on '%s' requested, but allow_live=False. "
                "Construct ActionExecutor(allow_live=True) to permit mutation.",
                action, target_service,
            )
            return BLOCKED

        # Checked before the client, because whether an action is implemented
        # does not depend on cluster availability, and NOT_IMPLEMENTED is the
        # more informative answer.
        if action not in LIVE_ACTIONS:
            logger.warning("[NOT_IMPLEMENTED] '%s' has no live implementation; "
                           "implemented actions are %s",
                           action, ", ".join(LIVE_ACTIONS))
            return NOT_IMPLEMENTED

        if self._apps is None:
            logger.error("[FAILED] no Kubernetes client available (%s)", self._api_error)
            return FAILED

        logger.warning("[LIVE] executing '%s' on '%s' in namespace '%s'",
                       action, target_service, self.namespace)
        try:
            if action == "Restart_Pod":
                self._restart_deployment(target_service)
            elif action == "Scale_Out":
                self._scale_out(target_service)
            return EXECUTED
        except Exception as exc:
            logger.error("[FAILED] %s on %s: %s", action, target_service, exc)
            return FAILED

    # ------------------------------------------------------------------
    # Live implementations
    # ------------------------------------------------------------------
    def _restart_deployment(self, deployment_name):
        """
        Trigger a rolling restart.

        Uses the same mechanism as `kubectl rollout restart`: stamp a changing
        annotation on the pod template so the deployment controller rolls pods
        gradually. Deleting pods directly would drop capacity all at once.
        """
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {"kubectl.kubernetes.io/restartedAt": now}
                    }
                }
            }
        }
        self._apps.patch_namespaced_deployment(deployment_name, self.namespace, body)
        logger.info("[LIVE] rollout restart issued for %s", deployment_name)

    def _scale_out(self, deployment_name, step=1):
        """
        Add replicas, up to `max_replicas`.

        Reads current replicas first rather than assuming, so this cooperates
        with whatever the HPA has already decided instead of overwriting it.
        """
        scale = self._apps.read_namespaced_deployment_scale(
            deployment_name, self.namespace
        )
        current = scale.spec.replicas or 0
        target = min(current + step, self.max_replicas)

        if target <= current:
            logger.info("[LIVE] %s already at %d replicas (max %d); no scale-out",
                        deployment_name, current, self.max_replicas)
            return

        self._apps.patch_namespaced_deployment_scale(
            deployment_name, self.namespace, {"spec": {"replicas": target}}
        )
        logger.info("[LIVE] scaled %s from %d to %d replicas",
                    deployment_name, current, target)
