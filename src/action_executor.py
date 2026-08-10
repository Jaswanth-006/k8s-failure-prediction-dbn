import logging
from kubernetes import client, config

class ActionExecutor:
    """
    Executes Kubernetes actions based on decisions.
    Strictly enforces shadow mode in Phase 4.1.
    """
    def __init__(self):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            try:
                config.load_kube_config()
            except Exception as e:
                logging.warning(f"[ActionExecutor] Failed to load kubeconfig: {e}")
        
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()

    def execute(self, action: str, target_service: str, shadow_mode: bool = True):
        """
        Executes or simulates the specified action on the target service.
        """
        if shadow_mode:
            logging.info(f"[SHADOW MODE] Would execute '{action}' on '{target_service}'. No cluster mutation performed.")
            return "WOULD_EXECUTE"

        # SAFETY RAIL: Fail safely if shadow_mode=False is attempted in Phase 4
        logging.warning("Safety Rail Triggered: LIVE EXECUTION IS DISABLED IN PHASE 4 SPRINT 4.3. shadow_mode MUST be True.")
        return "BLOCKED"

        # LIVE EXECUTION (Protected by shadow_mode flag)
        # logging.warning(f"[LIVE ACTION] Executing '{action}' on '{target_service}'")
        # try:
        #     if action == "Restart_Pod":
        #         self._restart_deployment(target_service)
        #         return "EXECUTED"
        #     elif action == "Reschedule_Pod":
        #         logging.info(f"Reschedule not fully implemented for {target_service} yet.")
        #         return "FAILED"
        #     elif action == "Scale_Out":
        #         logging.info(f"Scale_Out not fully implemented for {target_service} yet.")
        #         return "FAILED"
        #     else:
        #         logging.warning(f"Unknown action: {action}")
        #         return "FAILED"
        # except Exception as e:
        #     logging.error(f"[ActionExecutor] Failed to execute {action} on {target_service}: {e}")
        #     return "FAILED"

    def _restart_deployment(self, deployment_name: str, namespace: str = "default"):
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": now.isoformat()
                        }
                    }
                }
            }
        }
        self.k8s_apps.patch_namespaced_deployment(deployment_name, namespace, body)
