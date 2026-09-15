"""
Operator inference path: Prometheus -> Rectifier -> autoencoder -> DDN.

This is what the Kubernetes operator runs every tick, so it has to match how
the model was trained and evaluated. It previously diverged in four ways, each
of which meant the operator was not running the model the results describe:

  * node_cpu was a hardcoded 0.1. The autoencoder was trained on real
    node-exporter values (about 0.02), so every tick fed it a feature value it
    never saw in training.
  * The CPU query counted the pod sandbox ("POD") container; training and the
    run recorder exclude it.
  * The service graph was hardcoded and did not match the cluster. It routed
    ts-ui-dashboard to route and ts-order-service to inventory and station,
    whereas live Istio traffic shows train->route, order->payment and
    payment->inventory. Root-cause analysis walks this graph.
  * The DDN used its built-in default parameters instead of the ones calibrated
    from recorded fault runs.

It now uses the same CPU and node queries as scripts/37_record_runs.py, the
service graph discovered from Istio, and calibrated parameters. A missing graph
or parameter file falls back to defaults with a warning, so the operator still
starts. Missing node telemetry raises instead: substituting a constant is
exactly the mismatch this replaces, and the operator records the error in
status.health.lastError.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
import requests

from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
from src.rectifier import Rectifier

logger = logging.getLogger(__name__)

SERVICES = [
    "ts-ui-dashboard", "ts-user-service", "ts-train-service",
    "ts-route-service", "ts-order-service", "ts-payment-service",
    "ts-inventory-service", "ts-station-service",
]

DISCOVERED_GRAPH = "data/experiments/discovered_service_graph.json"
DEFAULT_PARAMS = "data/experiments/params/live_v2.json"

# Same particle count as the evaluation runs, so operator posteriors are
# comparable with the reported results.
NUM_PARTICLES = 500

# Used only when no discovered graph exists. Mirrors the topology observed from
# live Istio traffic rather than the earlier hardcoded guess.
FALLBACK_EDGES = [
    ("ts-ui-dashboard", "ts-user-service"),
    ("ts-ui-dashboard", "ts-train-service"),
    ("ts-ui-dashboard", "ts-order-service"),
    ("ts-ui-dashboard", "ts-station-service"),
    ("ts-train-service", "ts-route-service"),
    ("ts-order-service", "ts-payment-service"),
    ("ts-payment-service", "ts-inventory-service"),
]


def cpu_query(rate_window="2m"):
    """Per-pod CPU, identical to the query the training data was collected with."""
    return ('sum(rate(container_cpu_usage_seconds_total'
            '{container!="",container!="POD"}[%s])) by (pod)' % rate_window)


def node_cpu_query(rate_window="2m"):
    """Real node utilisation, from node-exporter."""
    return '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[%s]))' % rate_window


def load_service_graph(path=DISCOVERED_GRAPH):
    """Discovered topology if available, otherwise the fallback edges."""
    graph = nx.DiGraph()
    graph.add_nodes_from(SERVICES)

    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for edge in data.get("edges", []):
            src, dst = edge.get("source"), edge.get("destination")
            if src in SERVICES and dst in SERVICES:
                graph.add_edge(src, dst)
        source = path
    else:
        logger.warning("[InferenceAdapter] %s not found; using fallback topology", path)
        graph.add_edges_from(FALLBACK_EDGES)
        source = "fallback"

    if not nx.is_directed_acyclic_graph(graph):
        # The DDN and root-cause analysis need a DAG keyed by service name, so
        # rather than condensing (which renames nodes) fall back to known edges.
        logger.warning("[InferenceAdapter] discovered graph has a cycle; using fallback topology")
        graph = nx.DiGraph()
        graph.add_nodes_from(SERVICES)
        graph.add_edges_from(FALLBACK_EDGES)
        source = "fallback (discovered graph was cyclic)"

    return graph, source


def load_params(path):
    """Calibrated DDN parameters as constructor kwargs, or None to use defaults."""
    if not path or not os.path.exists(path):
        logger.warning("[InferenceAdapter] calibrated parameters %s not found; using DDN defaults", path)
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    topological = {int(k): np.array(v, dtype=np.float32)
                   for k, v in d.get("topological", {}).items()}
    return {
        "learned_mu": np.array(d["mu"], dtype=np.float32),
        "learned_sigma": np.array(d["sigma"], dtype=np.float32),
        "learned_T": np.array(d["transition"], dtype=np.float32),
        "learned_topological": topological or None,
    }


class InferenceAdapter:
    """
    Runs one inference tick at a time, keeping Rectifier and DDN state between
    ticks so temporal reasoning carries over.
    """

    def __init__(self, prometheus_url=None, model_path="models/phase3_autoencoder_cpu_only.pth",
                 params_path=None, graph_path=DISCOVERED_GRAPH, rate_window="2m"):
        # 127.0.0.1 rather than localhost: on Windows, localhost resolves to IPv6
        # ::1 first, and the kubectl port-forward listens only on IPv4, so every
        # request waited ~1.9s for the IPv6 attempt to fail. With two queries per
        # tick that pushed the operator past its 5s budget.
        self.prometheus_url = (prometheus_url
                               or os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090")).rstrip("/")
        self.services = SERVICES
        self.rate_window = rate_window

        self.rectifier = Rectifier(self.services, ["cpu_usage"], ["node_cpu"])
        self.pipeline = RobustAnomalyScorePipeline(self.rectifier.feature_names, self.services)
        logger.info("[InferenceAdapter] loading autoencoder from %s", model_path)
        self.pipeline.load_model(model_path)

        self.graph, self.graph_source = load_service_graph(graph_path)
        params_path = params_path or os.environ.get("PREFACE_PARAMS", DEFAULT_PARAMS)
        learned = load_params(params_path)
        self.params_source = params_path if learned else "DDN defaults"
        self.ddn = DynamicDecisionNetworkPhase3(self.graph, num_particles=NUM_PARTICLES, **(learned or {}))

        self.last_signals: Dict[str, float] = {}
        logger.info("[InferenceAdapter] graph: %s (%d edges); parameters: %s",
                    self.graph_source, self.graph.number_of_edges(), self.params_source)

    def _query(self, promql):
        response = requests.get("%s/api/v1/query" % self.prometheus_url,
                                params={"query": promql}, timeout=10)
        data = response.json()
        if data.get("status") != "success":
            raise RuntimeError("Prometheus query failed: %s" % data.get("error", data))
        return data["data"]["result"]

    def _query_prometheus(self) -> pd.DataFrame:
        """One snapshot of per-pod CPU and node CPU, in the Rectifier's input schema."""
        timestamp = datetime.now().isoformat()
        records = []

        for item in self._query(cpu_query(self.rate_window)):
            pod = item["metric"].get("pod", "")
            service = next((s for s in self.services if s in pod), None)
            if service is None:
                # Infrastructure pods; the Rectifier only models app services.
                continue
            records.append({
                "timestamp": timestamp, "pod_name": pod, "service_name": service,
                "pod_phase": "Running", "is_ready": True,
                "kpi_name": "cpu_usage", "value": float(item["value"][1]),
            })

        node = self._query(node_cpu_query(self.rate_window))
        if not node:
            raise RuntimeError("node_cpu unavailable: no node-exporter metrics in Prometheus")
        records.append({
            "timestamp": timestamp, "pod_name": "node", "service_name": "node",
            "pod_phase": "Running", "is_ready": True,
            "kpi_name": "node_cpu", "value": float(node[0]["value"][1]),
        })

        return pd.DataFrame(records)

    def run_tick(self) -> Dict[str, Any]:
        """
        Execute one inference tick. Raises if Prometheus or node telemetry is
        unavailable; the operator records that instead of inventing an
        observation.
        """
        df_tick = self._query_prometheus()
        x_t, _ = self.rectifier.process_tick(df_tick)
        self.last_signals = self.pipeline.compute_anomaly_signals(x_t)
        return self.ddn.step(self.last_signals, node_pressure_flag=False)
