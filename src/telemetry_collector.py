"""
PREFACE-DBN Goal 4: Prometheus Telemetry Collector
====================================================
Responsible ONLY for retrieving and normalizing raw telemetry from Prometheus.
Does NOT contain any RCA or causal reasoning logic.

Architecture position
---------------------
Telemetry (Prometheus / Istio)
   ↓
PrometheusTelemetryCollector      ← this module
   ↓
TelemetrySnapshot                 ← structured output for downstream use

Usage
-----
  collector = PrometheusTelemetryCollector(
      prometheus_url="http://localhost:9090",
      services=["ts-train-service", ...],
      service_graph=G,             # nx.DiGraph from Goal 2
  )
  snapshot = collector.collect()   # returns TelemetrySnapshot or None

Prometheus availability
-----------------------
All methods are written to degrade gracefully.
If Prometheus is unreachable, or if a specific metric family does not exist
in this cluster, the affected fields are set to 0.0 / latency_available=False
rather than raising exceptions.

Metric discovery
----------------
Before querying specific metrics, call verify_available_metrics() to obtain
the list of metrics actually present in this Prometheus instance.
This is important because Istio metric names and label schemas can differ
between versions.
"""

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import requests

from src.telemetry_schema import EdgeTelemetry, ServiceTelemetry, TelemetrySnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PromQL templates
# All use 5-minute rate windows, consistent with Goal 2's graph discovery.
# Labels filtered to exclude Prometheus/Istio internal workloads.
# ---------------------------------------------------------------------------

# CPU: container CPU usage rate per pod, summed per workload
_Q_CPU = (
    'sum by (pod) ('
    '  rate(container_cpu_usage_seconds_total{container!="",container!="POD"}[5m])'
    ')'
)

# Memory: working-set bytes per pod (preferred over RSS because it includes
#          page cache that the OOM killer would reclaim)
_Q_MEMORY = (
    'sum by (pod) ('
    '  container_memory_working_set_bytes{container!="",container!="POD"}'
    ')'
)

# Request rate per destination workload (as seen by the destination reporter)
_Q_REQ_RATE = (
    'sum by (destination_workload, source_workload) ('
    '  rate(istio_requests_total{reporter="destination"}[5m])'
    ')'
)

# Error rate per destination workload: 5xx responses / all responses
# Division-by-zero is avoided by using or vector(0) as the denominator guard.
_Q_ERR_RATE_NUM = (
    'sum by (destination_workload, source_workload) ('
    '  rate(istio_requests_total{reporter="destination",response_code=~"5.."}[5m])'
    ')'
)
_Q_ERR_RATE_DEN = (
    'sum by (destination_workload, source_workload) ('
    '  rate(istio_requests_total{reporter="destination"}[5m])'
    ')'
)

# Cumulative request count (for backpressure bonus; mirrors Goal 2 behavior)
_Q_REQ_COUNT = (
    'sum by (source_workload, destination_workload) ('
    '  istio_requests_total{reporter="source"}'
    ')'
)

# Latency P95: histogram_quantile over request duration.
# We attempt two metric names that appear in different Istio versions.
_LATENCY_METRIC_CANDIDATES = [
    "istio_request_duration_milliseconds_bucket",
    "istio_request_duration_seconds_bucket",  # older Istio versions
]

_Q_LATENCY_P95_TEMPLATE = (
    'histogram_quantile(0.95,'
    '  sum by (le, source_workload, destination_workload) ('
    '    rate({metric}{{reporter="destination"}}[5m])'
    '  )'
    ')'
)

# Request count via Istio (for edge telemetry, same label set as Goal 2)
_Q_EDGE_REQ_COUNT = (
    'sum by (source_workload, destination_workload) ('
    '  istio_requests_total{reporter="source"}'
    ')'
)


class PrometheusTelemetryCollector:
    """
    Collects multi-signal telemetry from Prometheus for Goal 4.

    Parameters
    ----------
    prometheus_url : str
        Base URL, e.g. "http://localhost:9090"
    services : List[str]
        Known workload names to collect service-level telemetry for.
    service_graph : nx.DiGraph
        The dependency graph discovered by Goal 2.
        Used to enumerate edges for edge-level telemetry.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        prometheus_url: str,
        services: List[str],
        service_graph: nx.DiGraph,
        timeout: float = 5.0,
    ):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.services: Set[str] = set(services)
        self.graph = service_graph
        self.timeout = timeout

        # Latency metric name, discovered at first collection.
        # None means not yet discovered; "" means not available.
        self._latency_metric: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_prometheus_available(self) -> bool:
        """Return True if Prometheus health endpoint responds OK."""
        try:
            r = requests.get(
                f"{self.prometheus_url}/-/healthy",
                timeout=self.timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    def verify_available_metrics(self) -> Set[str]:
        """
        Query Prometheus for all metric names currently present.
        Returns an empty set if Prometheus is unreachable.
        """
        try:
            r = requests.get(
                f"{self.prometheus_url}/api/v1/label/__name__/values",
                timeout=self.timeout,
            )
            data = r.json()
            if data.get("status") == "success":
                return set(data["data"])
        except Exception as exc:
            logger.warning("[TelemetryCollector] Cannot enumerate metrics: %s", exc)
        return set()

    def collect(self) -> Optional[TelemetrySnapshot]:
        """
        Collect a full multi-signal telemetry snapshot.

        Returns None if Prometheus is completely unreachable.
        Returns a TelemetrySnapshot with best-effort data otherwise
        (missing signals are 0.0 / latency_available=False).
        """
        if not self.is_prometheus_available():
            logger.warning(
                "[TelemetryCollector] Prometheus unreachable at %s",
                self.prometheus_url,
            )
            return None

        # Discover latency metric if not already done
        if self._latency_metric is None:
            self._discover_latency_metric()

        snapshot = TelemetrySnapshot()

        # --- Service-level ---
        cpu_map     = self._collect_cpu_by_service()
        memory_map  = self._collect_memory_by_service()
        req_map     = self._collect_request_rate_by_service()
        err_map     = self._collect_error_rate_by_service()

        for svc in self.services:
            snapshot.services[svc] = ServiceTelemetry(
                service=svc,
                cpu_rate=cpu_map.get(svc, 0.0),
                memory_bytes=memory_map.get(svc, 0.0),
                request_rate=req_map.get(svc, 0.0),
                error_rate=err_map.get(svc, 0.0),
                available=True,
            )

        # --- Edge-level ---
        edge_req_rate  = self._collect_edge_request_rate()
        edge_req_count = self._collect_edge_request_count()
        edge_err_rate  = self._collect_edge_error_rate()
        edge_latency   = self._collect_edge_latency_p95()

        for src, dst in self.graph.edges():
            key = (src, dst)
            lat_val, lat_avail = edge_latency.get(key, (0.0, False))
            snapshot.edges[key] = EdgeTelemetry(
                source=src,
                destination=dst,
                request_rate=edge_req_rate.get(key, 0.0),
                request_count=edge_req_count.get(key, 0.0),
                error_rate=edge_err_rate.get(key, 0.0),
                latency_p95=lat_val,
                latency_available=lat_avail,
            )

        return snapshot

    # ------------------------------------------------------------------
    # Private: metric discovery
    # ------------------------------------------------------------------

    def _discover_latency_metric(self):
        """Detect which latency histogram metric is available in this cluster."""
        available = self.verify_available_metrics()
        for candidate in _LATENCY_METRIC_CANDIDATES:
            if candidate in available:
                logger.info(
                    "[TelemetryCollector] Latency metric found: %s", candidate
                )
                self._latency_metric = candidate
                return
        # Check if we should scale seconds→ms
        # (older Istio uses _seconds, so multiply by 1000)
        logger.info(
            "[TelemetryCollector] No latency histogram metric found. "
            "latency_available will be False for all edges."
        )
        self._latency_metric = ""  # explicitly "not available"

    # ------------------------------------------------------------------
    # Private: service-level queries
    # ------------------------------------------------------------------

    def _raw_query(self, query: str) -> List[dict]:
        """Execute a PromQL instant query, returning result list."""
        try:
            r = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=self.timeout,
            )
            data = r.json()
            if data.get("status") == "success":
                return data["data"]["result"]
        except Exception as exc:
            logger.debug("[TelemetryCollector] Query failed: %s | %s", query[:80], exc)
        return []

    def _match_service(self, pod_name: str) -> Optional[str]:
        """Map a pod name to one of the known service names."""
        for svc in self.services:
            if svc in pod_name:
                return svc
        return None

    def _collect_cpu_by_service(self) -> Dict[str, float]:
        """Sum CPU rates per service (across pods)."""
        result: Dict[str, float] = {}
        for item in self._raw_query(_Q_CPU):
            pod = item["metric"].get("pod", "")
            svc = self._match_service(pod)
            if svc:
                val = _safe_float(item["value"][1])
                result[svc] = result.get(svc, 0.0) + val
        return result

    def _collect_memory_by_service(self) -> Dict[str, float]:
        """Sum working-set memory bytes per service (across pods)."""
        result: Dict[str, float] = {}
        for item in self._raw_query(_Q_MEMORY):
            pod = item["metric"].get("pod", "")
            svc = self._match_service(pod)
            if svc:
                val = _safe_float(item["value"][1])
                result[svc] = result.get(svc, 0.0) + val
        return result

    def _collect_request_rate_by_service(self) -> Dict[str, float]:
        """Sum incoming request rate per destination service."""
        result: Dict[str, float] = {}
        for item in self._raw_query(_Q_REQ_RATE):
            dst = item["metric"].get("destination_workload", "")
            if dst in self.services:
                val = _safe_float(item["value"][1])
                result[dst] = result.get(dst, 0.0) + val
        return result

    def _collect_error_rate_by_service(self) -> Dict[str, float]:
        """
        Compute per-service error rate = 5xx_rate / total_rate.
        Returns 0.0 when there is no traffic (avoids division-by-zero).
        """
        num_map: Dict[str, float] = {}
        den_map: Dict[str, float] = {}

        for item in self._raw_query(_Q_ERR_RATE_NUM):
            dst = item["metric"].get("destination_workload", "")
            if dst in self.services:
                num_map[dst] = num_map.get(dst, 0.0) + _safe_float(item["value"][1])

        for item in self._raw_query(_Q_ERR_RATE_DEN):
            dst = item["metric"].get("destination_workload", "")
            if dst in self.services:
                den_map[dst] = den_map.get(dst, 0.0) + _safe_float(item["value"][1])

        result: Dict[str, float] = {}
        for svc in self.services:
            den = den_map.get(svc, 0.0)
            num = num_map.get(svc, 0.0)
            result[svc] = (num / den) if den > 0.0 else 0.0
        return result

    # ------------------------------------------------------------------
    # Private: edge-level queries
    # ------------------------------------------------------------------

    def _collect_edge_request_rate(self) -> Dict[Tuple[str, str], float]:
        result: Dict[Tuple[str, str], float] = {}
        for item in self._raw_query(_Q_REQ_RATE):
            src = item["metric"].get("source_workload", "")
            dst = item["metric"].get("destination_workload", "")
            if src in self.services and dst in self.services and src != dst:
                key = (src, dst)
                result[key] = result.get(key, 0.0) + _safe_float(item["value"][1])
        return result

    def _collect_edge_request_count(self) -> Dict[Tuple[str, str], float]:
        result: Dict[Tuple[str, str], float] = {}
        for item in self._raw_query(_Q_EDGE_REQ_COUNT):
            src = item["metric"].get("source_workload", "")
            dst = item["metric"].get("destination_workload", "")
            if src in self.services and dst in self.services and src != dst:
                key = (src, dst)
                result[key] = result.get(key, 0.0) + _safe_float(item["value"][1])
        return result

    def _collect_edge_error_rate(self) -> Dict[Tuple[str, str], float]:
        """Edge error rate: 5xx / total, per (source, destination) pair."""
        num_map: Dict[Tuple[str, str], float] = {}
        den_map: Dict[Tuple[str, str], float] = {}

        for item in self._raw_query(_Q_ERR_RATE_NUM):
            src = item["metric"].get("source_workload", "")
            dst = item["metric"].get("destination_workload", "")
            if src in self.services and dst in self.services and src != dst:
                key = (src, dst)
                num_map[key] = num_map.get(key, 0.0) + _safe_float(item["value"][1])

        for item in self._raw_query(_Q_ERR_RATE_DEN):
            src = item["metric"].get("source_workload", "")
            dst = item["metric"].get("destination_workload", "")
            if src in self.services and dst in self.services and src != dst:
                key = (src, dst)
                den_map[key] = den_map.get(key, 0.0) + _safe_float(item["value"][1])

        result: Dict[Tuple[str, str], float] = {}
        for key in set(list(num_map.keys()) + list(den_map.keys())):
            den = den_map.get(key, 0.0)
            num = num_map.get(key, 0.0)
            result[key] = (num / den) if den > 0.0 else 0.0
        return result

    def _collect_edge_latency_p95(
        self,
    ) -> Dict[Tuple[str, str], Tuple[float, bool]]:
        """
        Collect P95 latency per edge.

        Returns dict mapping (src, dst) → (latency_ms, available).
        If the latency metric is not available, all values have available=False.

        Unit handling:
          - istio_request_duration_milliseconds_bucket → values already in ms
          - istio_request_duration_seconds_bucket      → multiply by 1000
        """
        result: Dict[Tuple[str, str], Tuple[float, bool]] = {}

        if not self._latency_metric:
            return result  # latency not available

        scale = 1.0
        if "seconds" in self._latency_metric and "milliseconds" not in self._latency_metric:
            scale = 1000.0  # convert seconds → ms

        q = _Q_LATENCY_P95_TEMPLATE.format(metric=self._latency_metric)
        for item in self._raw_query(q):
            src = item["metric"].get("source_workload", "")
            dst = item["metric"].get("destination_workload", "")
            if src in self.services and dst in self.services and src != dst:
                key = (src, dst)
                val = _safe_float(item["value"][1])
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                result[key] = (val * scale, True)

        return result


# ---------------------------------------------------------------------------
# Utility used inside this module only
# ---------------------------------------------------------------------------
def _safe_float(v, default: float = 0.0) -> float:
    """Convert Prometheus value string to float, returning default on failure."""
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default
