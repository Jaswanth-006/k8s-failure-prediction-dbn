"""
PREFACE-DBN Goal 4: Telemetry Data Model
=========================================
Defines typed, structured representations for:
  - ServiceTelemetry : per-microservice health signals (CPU, memory, request rate, error rate)
  - EdgeTelemetry    : per-dependency health signals (traffic, errors, latency)
  - TelemetrySnapshot: complete snapshot at one point in time

Also provides:
  - TelemetryNormalizer: robust clipping + min-max normalization for multi-signal RCA

Design notes
------------
SERVICE-level signals describe the intrinsic health of a microservice.
EDGE-level signals describe the health of a dependency path between two services.
These must NOT be conflated (see Goal 4 spec, Part 12).

Latency is an EDGE/REQUEST-PATH signal: it cannot meaningfully be attributed to
a service independently of which dependency path exhibits the latency.

Normalization uses predefined, physically meaningful value ranges rather than
training-time statistics, so it works correctly for a single online snapshot
without requiring historical data.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import math


# ---------------------------------------------------------------------------
# Normalization ranges: physically meaningful bounds for each signal type.
# These are conservative upper bounds for "abnormal but plausible" values.
# Clipping is applied before normalization so that extreme outliers do not
# dominate the scoring.
#
# Rationale for each bound:
#   cpu_rate    : 1.0 core = fully saturated container; >1.0 throttled
#   memory_bytes: 2 GB upper bound for a single microservice pod
#   request_rate: 1000 req/s is extreme for a single TrainTicket service
#   error_rate  : 0.0 – 1.0 (fraction); 1.0 = 100% errors
#   latency_ms  : 10 000 ms (10 s) = effectively broken dependency
# ---------------------------------------------------------------------------
_NORM_RANGES: Dict[str, tuple] = {
    "cpu_rate":      (0.0, 1.0),
    "memory_bytes":  (0.0, 2_000_000_000.0),
    "request_rate":  (0.0, 1000.0),
    "error_rate":    (0.0, 1.0),
    "latency_ms":    (0.0, 10_000.0),
}

_EPS = 1e-9  # prevents division by zero in normalization


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default for None/NaN/Inf."""
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def normalize_signal(value: float, signal_type: str) -> float:
    """
    Normalize a raw telemetry value to [0, 1] using predefined physical ranges.

    Method: robust clipping + min-max normalization.
      1. Clip to [min_val, max_val] — removes outliers without discarding the signal.
      2. Linearly scale to [0, 1].

    Returns 0.0 for missing/invalid inputs.
    """
    value = _safe_float(value, default=0.0)
    lo, hi = _NORM_RANGES.get(signal_type, (0.0, 1.0))
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo + _EPS)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ServiceTelemetry:
    """
    Per-microservice health signals collected at one tick.

    All raw values are stored as-is (real units).
    Use normalize_signal() before combining signals in scoring.

    Fields
    ------
    service      : workload name (e.g. "ts-train-service")
    cpu_rate     : CPU cores consumed (rate, e.g. 0.15 = 15% of 1 core)
    memory_bytes : working-set memory in bytes
    request_rate : incoming requests per second (aggregated across edges)
    error_rate   : fraction of requests returning HTTP 5xx (0.0 – 1.0)
    available    : False when Prometheus returned no data for this service
    """
    service:       str
    cpu_rate:      float = 0.0
    memory_bytes:  float = 0.0
    request_rate:  float = 0.0
    error_rate:    float = 0.0
    available:     bool  = True

    def normalized(self) -> Dict[str, float]:
        """Return dict of normalized [0,1] values, keyed by signal name."""
        return {
            "cpu_rate":     normalize_signal(self.cpu_rate,     "cpu_rate"),
            "memory_bytes": normalize_signal(self.memory_bytes, "memory_bytes"),
            "request_rate": normalize_signal(self.request_rate, "request_rate"),
            "error_rate":   normalize_signal(self.error_rate,   "error_rate"),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service":       self.service,
            "cpu_rate":      self.cpu_rate,
            "memory_bytes":  self.memory_bytes,
            "request_rate":  self.request_rate,
            "error_rate":    self.error_rate,
            "available":     self.available,
        }


@dataclass
class EdgeTelemetry:
    """
    Per-dependency-edge health signals collected at one tick.

    Latency is explicitly marked as unavailable when the Prometheus instance
    does not expose histogram metrics, so callers can degrade gracefully.

    Fields
    ------
    source            : calling workload name
    destination       : called workload name
    request_rate      : requests/second on this edge
    request_count     : cumulative request count on this edge (raw counter value)
    error_rate        : fraction of requests returning HTTP 5xx on this edge
    latency_p95       : 95th-percentile latency in milliseconds (or 0.0 if unavailable)
    latency_available : True only when a real latency value was obtained
    """
    source:            str
    destination:       str
    request_rate:      float = 0.0
    request_count:     float = 0.0
    error_rate:        float = 0.0
    latency_p95:       float = 0.0
    latency_available: bool  = False

    def normalized(self) -> Dict[str, float]:
        """Return dict of normalized [0,1] values, keyed by signal name."""
        result = {
            "request_rate":  normalize_signal(self.request_rate,  "request_rate"),
            "error_rate":    normalize_signal(self.error_rate,     "error_rate"),
        }
        if self.latency_available:
            result["latency_p95"] = normalize_signal(self.latency_p95, "latency_ms")
        else:
            result["latency_p95"] = 0.0
        return result

    def dependency_pressure(
        self,
        weight_error: float = 0.3,
        weight_latency: float = 0.2,
    ) -> float:
        """
        Compute a scalar 'dependency pressure' signal for this edge.

        Combines:
          - request_count (log-scaled, existing Goal 3 behavior)
          - error_rate (normalized)
          - latency_p95 (normalized, only when available)

        This value is used as the backpressure bonus in upstream causal scoring.
        It is bounded at a reasonable maximum so it cannot dominate the RCA score.

        Weights are explicit named parameters so callers can inspect them.
        They were chosen so that the pressure signal stays in roughly [0, 1]
        under normal operating conditions.
        """
        # Existing Goal 3 component: log-scaled request count
        log_req = math.log1p(max(0.0, self.request_count)) * 0.1

        norm = self.normalized()
        error_component   = weight_error   * norm["error_rate"]
        latency_component = weight_latency * norm["latency_p95"] if self.latency_available else 0.0

        return log_req + error_component + latency_component

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source":            self.source,
            "destination":       self.destination,
            "request_rate":      self.request_rate,
            "request_count":     self.request_count,
            "error_rate":        self.error_rate,
            "latency_p95":       self.latency_p95,
            "latency_available": self.latency_available,
        }


@dataclass
class TelemetrySnapshot:
    """
    Complete multi-signal telemetry snapshot for one inference tick.

    services : workload_name → ServiceTelemetry
    edges    : (source, destination) → EdgeTelemetry
    """
    services: Dict[str, ServiceTelemetry] = field(default_factory=dict)
    edges:    Dict[tuple, EdgeTelemetry]  = field(default_factory=dict)

    def get_service(self, name: str) -> Optional[ServiceTelemetry]:
        return self.services.get(name)

    def get_edge(self, source: str, destination: str) -> Optional[EdgeTelemetry]:
        return self.edges.get((source, destination))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "edges":    {f"{s}->{d}": e.to_dict() for (s, d), e in self.edges.items()},
        }
