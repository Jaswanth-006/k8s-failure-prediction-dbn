"""
PREFACE-DBN Goal 4: Deterministic Telemetry Tests
==================================================
Scripts: scripts/30_test_goal4_telemetry.py

All tests are deterministic and require no Prometheus connection.

Tests
-----
TEST 1  — Healthy service telemetry (all signals low)
TEST 2  — High CPU service
TEST 3  — High memory service
TEST 4  — High error-rate service
TEST 5  — High-latency dependency
TEST 6  — High request-rate dependency
TEST 7  — Upstream service unhealthy before downstream → upstream = ROOT_CAUSE
TEST 8  — Downstream unhealthy first → downstream = ROOT_CAUSE (not upstream)
TEST 9  — Missing latency metric does not crash
TEST 10 — Zero request traffic does not cause division-by-zero
TEST 11 — Goal 3 backward compat: no Goal 4 telemetry supplied
TEST 12 — Goal 2 discovered graph consumed without hardcoded edges
"""

import sys
import os
import json
import math

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
from src.causal_rca import DirectionalCausalAnalyzer
from src.telemetry_schema import (
    ServiceTelemetry,
    EdgeTelemetry,
    TelemetrySnapshot,
    normalize_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph_ab():
    """A → B service graph for simple two-node tests."""
    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)
    return G


def make_healthy_posteriors(*services):
    return {s: {"Normal": 1.0, "Degrading": 0.0, "Critical": 0.0} for s in services}


def make_critical_posteriors(critical_svc, all_services):
    result = make_healthy_posteriors(*all_services)
    result[critical_svc] = {"Normal": 0.0, "Degrading": 0.1, "Critical": 0.9}
    return result


def make_degrading_posteriors(degrading_svc, all_services):
    result = make_healthy_posteriors(*all_services)
    result[degrading_svc] = {"Normal": 0.2, "Degrading": 0.6, "Critical": 0.2}
    return result


def _pass(name):
    print(f"{name} ... PASSED")


def _fail(name, msg):
    print(f"{name} ... FAILED: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# TEST 1 — Healthy service telemetry (all signals low)
# ---------------------------------------------------------------------------
def test_01_healthy_service_telemetry():
    name = "TEST 1  (Healthy service telemetry)"

    svc_tel = ServiceTelemetry(
        service="ts-train-service",
        cpu_rate=0.05,
        memory_bytes=50_000_000,
        request_rate=2.0,
        error_rate=0.0,
        available=True,
    )

    norm = svc_tel.normalized()

    # All normalized values should be low
    assert norm["cpu_rate"]     < 0.15,  f"cpu_rate too high: {norm['cpu_rate']}"
    assert norm["memory_bytes"] < 0.05,  f"memory too high: {norm['memory_bytes']}"
    assert norm["request_rate"] < 0.01,  f"req_rate too high: {norm['request_rate']}"
    assert norm["error_rate"]   == 0.0,  f"error_rate should be 0.0: {norm['error_rate']}"

    _pass(name)


# ---------------------------------------------------------------------------
# TEST 2 — High CPU service
# ---------------------------------------------------------------------------
def test_02_high_cpu_service():
    name = "TEST 2  (High CPU service)"

    svc_high_cpu = ServiceTelemetry(
        service="ts-train-service",
        cpu_rate=0.95,       # Near 1 core — very high for a container
        memory_bytes=50_000_000,
        request_rate=2.0,
        error_rate=0.0,
        available=True,
    )

    norm = svc_high_cpu.normalized()
    assert norm["cpu_rate"] > 0.9, f"Expected high normalized CPU, got {norm['cpu_rate']}"

    # High CPU should raise intrinsic evidence when enrichment is applied
    G = make_graph_ab()
    analyzer = DirectionalCausalAnalyzer(G)
    service_telemetry = {"A": svc_high_cpu}

    posteriors = {
        "A": {"Normal": 0.1, "Degrading": 0.2, "Critical": 0.7},
        "B": {"Normal": 1.0, "Degrading": 0.0, "Critical": 0.0},
    }
    anomaly_signals = {"A": 4.0, "B": 0.0}

    result = analyzer.step(
        anomaly_signals, posteriors,
        service_telemetry=service_telemetry,
    )

    # A should be root cause
    assert result["root_cause"] == "A", f"Expected A, got {result['root_cause']}"
    _pass(name)


# ---------------------------------------------------------------------------
# TEST 3 — High memory service
# ---------------------------------------------------------------------------
def test_03_high_memory_service():
    name = "TEST 3  (High memory service)"

    svc_high_mem = ServiceTelemetry(
        service="ts-train-service",
        cpu_rate=0.05,
        memory_bytes=1_800_000_000,   # ~1.8 GB — very high
        request_rate=2.0,
        error_rate=0.0,
        available=True,
    )

    norm = svc_high_mem.normalized()
    assert norm["memory_bytes"] > 0.85, f"Expected high normalized memory, got {norm['memory_bytes']}"

    # High memory should contribute to intrinsic evidence
    G = make_graph_ab()
    analyzer = DirectionalCausalAnalyzer(G)

    service_telemetry = {"A": svc_high_mem}
    posteriors = {
        "A": {"Normal": 0.2, "Degrading": 0.4, "Critical": 0.4},
        "B": {"Normal": 1.0, "Degrading": 0.0, "Critical": 0.0},
    }
    anomaly_signals = {"A": 2.0, "B": 0.0}

    result = analyzer.step(
        anomaly_signals, posteriors,
        service_telemetry=service_telemetry,
    )
    score_A = result["scores"]["A"]["intrinsic_evidence"]

    # Compare with no enrichment (baseline)
    analyzer2 = DirectionalCausalAnalyzer(G)
    result2 = analyzer2.step(anomaly_signals, posteriors)
    score_A_base = result2["scores"]["A"]["intrinsic_evidence"]

    assert score_A > score_A_base, (
        f"Memory enrichment should raise intrinsic evidence: {score_A} vs {score_A_base}"
    )
    _pass(name)


# ---------------------------------------------------------------------------
# TEST 4 — High error-rate service
# ---------------------------------------------------------------------------
def test_04_high_error_rate_service():
    name = "TEST 4  (High error-rate service)"

    svc_errors = ServiceTelemetry(
        service="ts-train-service",
        cpu_rate=0.1,
        memory_bytes=100_000_000,
        request_rate=50.0,
        error_rate=0.85,   # 85% error rate
        available=True,
    )

    norm = svc_errors.normalized()
    assert norm["error_rate"] > 0.80, f"Expected high normalized error rate, got {norm['error_rate']}"

    G = make_graph_ab()
    analyzer_enriched = DirectionalCausalAnalyzer(G)
    analyzer_baseline = DirectionalCausalAnalyzer(G)

    posteriors = {
        "A": {"Normal": 0.3, "Degrading": 0.4, "Critical": 0.3},
        "B": {"Normal": 1.0, "Degrading": 0.0, "Critical": 0.0},
    }
    anomaly_signals = {"A": 1.5, "B": 0.0}

    result_enriched = analyzer_enriched.step(
        anomaly_signals, posteriors,
        service_telemetry={"A": svc_errors},
    )
    result_baseline = analyzer_baseline.step(anomaly_signals, posteriors)

    score_enriched = result_enriched["scores"]["A"]["intrinsic_evidence"]
    score_baseline = result_baseline["scores"]["A"]["intrinsic_evidence"]

    assert score_enriched > score_baseline, (
        f"Error-rate enrichment should raise intrinsic evidence: {score_enriched} vs {score_baseline}"
    )
    _pass(name)


# ---------------------------------------------------------------------------
# TEST 5 — High-latency dependency
# ---------------------------------------------------------------------------
def test_05_high_latency_dependency():
    name = "TEST 5  (High-latency dependency)"

    edge_high_lat = EdgeTelemetry(
        source="A",
        destination="B",
        request_rate=10.0,
        request_count=1000.0,
        error_rate=0.05,
        latency_p95=4500.0,   # 4.5 s — very high
        latency_available=True,
    )

    norm = edge_high_lat.normalized()
    assert norm["latency_p95"] > 0.40, f"Expected high normalized latency, got {norm['latency_p95']}"

    pressure = edge_high_lat.dependency_pressure()
    assert pressure > 0.5, f"High latency should produce meaningful pressure: {pressure}"

    _pass(name)


# ---------------------------------------------------------------------------
# TEST 6 — High request-rate dependency
# ---------------------------------------------------------------------------
def test_06_high_request_rate_dependency():
    name = "TEST 6  (High request-rate dependency)"

    edge_high_req = EdgeTelemetry(
        source="A",
        destination="B",
        request_rate=800.0,
        request_count=100_000.0,
        error_rate=0.0,
        latency_p95=0.0,
        latency_available=False,
    )

    norm = edge_high_req.normalized()
    assert norm["request_rate"] > 0.70, f"Expected high normalized request rate, got {norm['request_rate']}"

    pressure = edge_high_req.dependency_pressure()
    assert pressure > 1.0, f"High request count should produce large log pressure: {pressure}"

    _pass(name)


# ---------------------------------------------------------------------------
# TEST 7 — Upstream unhealthy BEFORE downstream -> upstream = ROOT_CAUSE
# ---------------------------------------------------------------------------
def test_07_upstream_fails_first():
    name = "TEST 7  (Upstream fails first -> upstream = ROOT_CAUSE)"

    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)
    G.add_edge("B", "C", request_count=100)

    analyzer = DirectionalCausalAnalyzer(G)

    edge_telemetry = {
        ("A", "B"): EdgeTelemetry(
            source="A", destination="B",
            request_rate=10.0, request_count=500.0,
            error_rate=0.3, latency_p95=2000.0, latency_available=True
        ),
        ("B", "C"): EdgeTelemetry(
            source="B", destination="C",
            request_rate=5.0, request_count=200.0,
            error_rate=0.0, latency_p95=0.0, latency_available=False
        ),
    }
    service_telemetry = {
        "A": ServiceTelemetry("A", cpu_rate=0.9, memory_bytes=1_000_000_000,
                               request_rate=10.0, error_rate=0.0, available=True),
        "B": ServiceTelemetry("B", cpu_rate=0.1, memory_bytes=50_000_000,
                               request_rate=5.0, error_rate=0.0, available=True),
        "C": ServiceTelemetry("C", cpu_rate=0.0, memory_bytes=50_000_000,
                               request_rate=0.0, error_rate=0.0, available=True),
    }

    # Tick 1: A fails
    analyzer.step(
        {"A": 5.0, "B": 0.0, "C": 0.0},
        {"A": {"Normal": 0.1, "Critical": 0.9, "Degrading": 0.0},
         "B": {"Normal": 1.0, "Critical": 0.0, "Degrading": 0.0},
         "C": {"Normal": 1.0, "Critical": 0.0, "Degrading": 0.0}},
        edge_telemetry=edge_telemetry,
        service_telemetry=service_telemetry,
    )

    # Tick 2: B also fails (downstream propagation)
    result = analyzer.step(
        {"A": 6.0, "B": 4.0, "C": 0.0},
        {"A": {"Normal": 0.0, "Critical": 1.0, "Degrading": 0.0},
         "B": {"Normal": 0.2, "Critical": 0.8, "Degrading": 0.0},
         "C": {"Normal": 1.0, "Critical": 0.0, "Degrading": 0.0}},
        edge_telemetry=edge_telemetry,
        service_telemetry=service_telemetry,
    )

    assert result["root_cause"] == "A", f"Expected A as root cause, got {result['root_cause']}"
    assert result["scores"]["B"]["classification"] == "PROPAGATED_VICTIM", \
        f"Expected B=PROPAGATED_VICTIM, got {result['scores']['B']['classification']}"
    _pass(name)


# ---------------------------------------------------------------------------
# TEST 8 — Downstream unhealthy FIRST -> downstream = ROOT_CAUSE
# ---------------------------------------------------------------------------
def test_08_downstream_fails_first():
    name = "TEST 8  (Downstream fails first -> downstream = ROOT_CAUSE)"

    G = make_graph_ab()
    analyzer = DirectionalCausalAnalyzer(G)

    service_telemetry = {
        "A": ServiceTelemetry("A", cpu_rate=0.0, memory_bytes=0,
                               request_rate=0.0, error_rate=0.0, available=True),
        "B": ServiceTelemetry("B", cpu_rate=0.9, memory_bytes=1_500_000_000,
                               request_rate=10.0, error_rate=0.6, available=True),
    }

    result = analyzer.step(
        {"A": 0.0, "B": 5.5},
        {"A": {"Normal": 1.0, "Degrading": 0.0, "Critical": 0.0},
         "B": {"Normal": 0.0, "Degrading": 0.1, "Critical": 0.9}},
        service_telemetry=service_telemetry,
    )

    assert result["root_cause"] == "B", f"Expected B as root cause, got {result['root_cause']}"
    assert result["scores"]["A"]["classification"] == "NORMAL", \
        f"Expected A=NORMAL, got {result['scores']['A']['classification']}"
    _pass(name)


# ---------------------------------------------------------------------------
# TEST 9 — Missing latency metric does NOT crash
# ---------------------------------------------------------------------------
def test_09_missing_latency_no_crash():
    name = "TEST 9  (Missing latency metric does not crash)"

    edge_no_lat = EdgeTelemetry(
        source="A",
        destination="B",
        request_rate=5.0,
        request_count=500.0,
        error_rate=0.1,
        latency_p95=0.0,
        latency_available=False,   # latency unavailable
    )

    # Must not raise any exception
    try:
        pressure = edge_no_lat.dependency_pressure()
        norm     = edge_no_lat.normalized()
    except Exception as e:
        _fail(name, f"Exception raised: {e}")
        return

    # latency_p95 in normalized output should be 0.0
    assert norm["latency_p95"] == 0.0, f"Expected 0.0 when unavailable, got {norm['latency_p95']}"
    # Pressure should still be non-zero (request_count + error_rate contribute)
    assert pressure > 0.0, "Pressure should be > 0 even without latency"

    _pass(name)


# ---------------------------------------------------------------------------
# TEST 10 — Zero request traffic does NOT cause division-by-zero
# ---------------------------------------------------------------------------
def test_10_zero_traffic_no_division_by_zero():
    name = "TEST 10 (Zero traffic — no division-by-zero)"

    zero_traffic_edge = EdgeTelemetry(
        source="A",
        destination="B",
        request_rate=0.0,
        request_count=0.0,
        error_rate=0.0,
        latency_p95=0.0,
        latency_available=False,
    )

    zero_svc = ServiceTelemetry(
        service="ts-train-service",
        cpu_rate=0.0,
        memory_bytes=0.0,
        request_rate=0.0,
        error_rate=0.0,
        available=True,
    )

    try:
        pressure = zero_traffic_edge.dependency_pressure()
        norm_edge = zero_traffic_edge.normalized()
        norm_svc  = zero_svc.normalized()
    except ZeroDivisionError as e:
        _fail(name, f"ZeroDivisionError: {e}")
        return
    except Exception as e:
        _fail(name, f"Unexpected exception: {e}")
        return

    assert pressure == 0.0, f"Zero traffic edge should have 0 pressure, got {pressure}"
    assert norm_edge["request_rate"] == 0.0
    assert norm_svc["error_rate"]    == 0.0

    # Also verify normalize_signal with zero values and its inverse
    assert normalize_signal(0.0, "error_rate")   == 0.0
    assert normalize_signal(0.0, "request_rate") == 0.0

    _pass(name)


# ---------------------------------------------------------------------------
# TEST 11 — Goal 3 backward compatibility (no Goal 4 telemetry)
# ---------------------------------------------------------------------------
def test_11_goal3_backward_compatibility():
    name = "TEST 11 (Goal 3 backward compat — no Goal 4 telemetry)"

    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)

    analyzer = DirectionalCausalAnalyzer(G)

    # Call with only the Goal 3 arguments (no edge_telemetry, no service_telemetry)
    result = analyzer.step(
        {"A": 5.0, "B": 0.0},
        {"A": {"Normal": 0.1, "Degrading": 0.0, "Critical": 0.9},
         "B": {"Normal": 1.0, "Degrading": 0.0, "Critical": 0.0}},
        # deliberately omitting edge_telemetry and service_telemetry
    )

    assert result["root_cause"] == "A", f"Expected A, got {result['root_cause']}"
    assert result["scores"]["A"]["classification"] == "ROOT_CAUSE"

    # Tick 2: B becomes victim
    result2 = analyzer.step(
        {"A": 6.0, "B": 4.0},
        {"A": {"Normal": 0.0, "Degrading": 0.0, "Critical": 1.0},
         "B": {"Normal": 0.2, "Degrading": 0.0, "Critical": 0.8}},
    )
    assert result2["root_cause"] == "A", f"Expected A on tick 2, got {result2['root_cause']}"
    assert result2["scores"]["B"]["classification"] == "PROPAGATED_VICTIM"

    _pass(name)


# ---------------------------------------------------------------------------
# TEST 12 — Goal 2 discovered graph consumed without hardcoded edges
# ---------------------------------------------------------------------------
def test_12_goal2_graph_no_hardcoded_edges():
    name = "TEST 12 (Goal 2 graph used without hardcoded edges)"

    graph_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/experiments/discovered_service_graph.json")
    )

    if not os.path.exists(graph_file):
        print(f"{name} ... SKIPPED (No discovered_service_graph.json)")
        return

    with open(graph_file, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data.get("nodes", []):
        G.add_node(node)
    for edge in data.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["destination"],
            request_count=edge.get("request_count", 0),
        )

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    assert num_nodes == 8,  f"Expected 8 nodes, got {num_nodes}"
    assert num_edges == 7,  f"Expected 7 edges, got {num_edges}"

    # Build edge telemetry from graph edges (no hardcoded list of edges)
    edge_tel = {}
    for src, dst, data_attr in G.edges(data=True):
        erate = 0.5 if src == "ts-train-service" and dst == "ts-route-service" else 0.0
        edge_tel[(src, dst)] = EdgeTelemetry(
            source=src,
            destination=dst,
            request_rate=1.0,
            request_count=data_attr.get("request_count", 0.0),
            error_rate=erate,
            latency_p95=0.0,
            latency_available=False,
        )

    analyzer = DirectionalCausalAnalyzer(G)

    # Inject fault in ts-train-service (it has a known downstream: ts-route-service)
    # Tick 1: train-service becomes critical
    res = analyzer.step(
        {"ts-train-service": 5.0},
        {"ts-train-service": {"Normal": 0.0, "Degrading": 0.1, "Critical": 0.9}},
        edge_telemetry=edge_tel,
    )
    assert res["root_cause"] == "ts-train-service", \
        f"Expected ts-train-service, got {res['root_cause']}"

    # Tick 2: downstream ts-route-service degrades
    res2 = analyzer.step(
        {"ts-train-service": 5.0, "ts-route-service": 4.0},
        {
            "ts-train-service": {"Normal": 0.0, "Degrading": 0.1, "Critical": 0.9},
            "ts-route-service":  {"Normal": 0.2, "Degrading": 0.0, "Critical": 0.8},
        },
        edge_telemetry=edge_tel,
    )
    assert res2["root_cause"] == "ts-train-service", \
        f"Expected ts-train-service on tick 2, got {res2['root_cause']}"
    assert res2["scores"]["ts-route-service"]["classification"] == "PROPAGATED_VICTIM", \
        f"Expected ts-route-service=PROPAGATED_VICTIM"

    _pass(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PREFACE-DBN - GOAL 4 Telemetry Tests")
    print("=" * 60)
    print()

    test_01_healthy_service_telemetry()
    test_02_high_cpu_service()
    test_03_high_memory_service()
    test_04_high_error_rate_service()
    test_05_high_latency_dependency()
    test_06_high_request_rate_dependency()
    test_07_upstream_fails_first()
    test_08_downstream_fails_first()
    test_09_missing_latency_no_crash()
    test_10_zero_traffic_no_division_by_zero()
    test_11_goal3_backward_compatibility()
    test_12_goal2_graph_no_hardcoded_edges()

    print()
    print("=" * 60)
    print("ALL GOAL 4 TESTS PASSED.")
    print("=" * 60)
