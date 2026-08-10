import logging
import requests
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, Any

from src.rectifier import Rectifier
from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3

SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service"
]

class InferenceAdapter:
    """
    Adapter that orchestrates the Phase 3 inference pipeline:
    Prometheus -> Rectifier -> Autoencoder -> DDN.
    Keeps state (Rectifier EMA, DDN Particles) across ticks.
    """
    def __init__(self, prometheus_url: str = "http://localhost:9090", model_path: str = "models/phase3_autoencoder_cpu_only.pth"):
        self.prometheus_url = prometheus_url
        self.services = SERVICES
        
        # Build service graph for DDN
        G = nx.DiGraph()
        G.add_edge("ts-ui-dashboard", "ts-user-service")
        G.add_edge("ts-ui-dashboard", "ts-train-service")
        G.add_edge("ts-ui-dashboard", "ts-route-service")
        G.add_edge("ts-ui-dashboard", "ts-order-service")
        G.add_edge("ts-order-service", "ts-payment-service")
        G.add_edge("ts-order-service", "ts-inventory-service")
        G.add_edge("ts-order-service", "ts-station-service")
        
        # Initialize Phase 3 components
        self.rectifier = Rectifier(self.services, ["cpu_usage"], ["node_cpu"])
        self.pipeline = RobustAnomalyScorePipeline(self.rectifier.feature_names, self.services)
        
        logging.info(f"[InferenceAdapter] Loading Phase 3 Autoencoder from {model_path}")
        self.pipeline.load_model(model_path)
        
        self.ddn = DynamicDecisionNetworkPhase3(G, num_particles=1000)

    def _query_prometheus(self) -> pd.DataFrame:
        """Fetches the latest metrics snapshot from Prometheus."""
        records = []
        timestamp = datetime.now().isoformat()
        
        query = 'sum(rate(container_cpu_usage_seconds_total{container!=""}[2m])) by (pod, namespace)'
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=2)
            result = response.json()
            if result.get("status") == "success":
                for item in result["data"]["result"]:
                    pod_name = item["metric"].get("pod", "unknown")
                    val = float(item["value"][1])
                    srv_name = "unknown"
                    for s in self.services:
                        if s in pod_name:
                            srv_name = s
                            break
                    records.append({
                        "timestamp": timestamp,
                        "pod_name": pod_name,
                        "service_name": srv_name,
                        "pod_phase": "Running",
                        "is_ready": True,
                        "kpi_name": "cpu_usage",
                        "value": val
                    })
        except Exception as e:
            logging.error(f"[InferenceAdapter] Prometheus query failed: {e}")
            raise

        # Dummy node_cpu to satisfy Rectifier input dimensions (as established in Phase 3 experiments)
        records.append({
            "timestamp": timestamp,
            "pod_name": "node",
            "service_name": "node",
            "pod_phase": "Running",
            "is_ready": True,
            "kpi_name": "node_cpu",
            "value": 0.1
        })
        
        return pd.DataFrame(records)

    def run_tick(self) -> Dict[str, Any]:
        """
        Executes one full inference tick.
        Raises exception if Prometheus is down or inference fails.
        """
        # 1. Fetch telemetry
        df_tick = self._query_prometheus()
        
        # 2. Rectifier
        x_t, _ = self.rectifier.process_tick(df_tick)
        
        # 3. Autoencoder
        anomaly_signals = self.pipeline.compute_anomaly_signals(x_t)
        
        # 4. DDN
        ddn_output = self.ddn.step(anomaly_signals, node_pressure_flag=False)
        
        return ddn_output
