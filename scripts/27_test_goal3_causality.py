import sys
import os
import networkx as nx

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.causal_rca import DirectionalCausalAnalyzer
from src.ddn_core_phase3 import DynamicDecisionNetworkPhase3
import json

def test_independent_anomaly():
    """
    TEST 1:
    Independent anomaly
    A healthy, B anomalous.
    Expected: B should be preferred root cause.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)
    
    analyzer = DirectionalCausalAnalyzer(G)
    
    # Tick 1: B becomes anomalous
    anomaly_signals = {"A": 0.0, "B": 5.0}
    posteriors = {
        "A": {"Normal": 0.9, "Degrading": 0.1, "Critical": 0.0},
        "B": {"Normal": 0.1, "Degrading": 0.1, "Critical": 0.8}
    }
    
    res = analyzer.step(anomaly_signals, posteriors)
    assert res["root_cause"] == "B", f"Expected B, got {res['root_cause']}"
    assert res["scores"]["B"]["classification"] == "ROOT_CAUSE"
    print("TEST 1 (Independent Anomaly) PASSED")


def test_upstream_failure_propagation():
    """
    TEST 2:
    Upstream failure propagation
    A anomalous first, then B anomalous later.
    Expected: A should be preferred root cause.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)
    
    analyzer = DirectionalCausalAnalyzer(G)
    
    # Tick 1: A becomes anomalous
    res = analyzer.step(
        {"A": 5.0, "B": 0.0},
        {"A": {"Normal": 0.1, "Critical": 0.9}, "B": {"Normal": 0.9, "Critical": 0.1}}
    )
    assert res["root_cause"] == "A"
    
    # Tick 2: B becomes anomalous due to A
    res = analyzer.step(
        {"A": 6.0, "B": 4.0},
        {"A": {"Normal": 0.0, "Critical": 1.0}, "B": {"Normal": 0.2, "Critical": 0.8}}
    )
    
    assert res["root_cause"] == "A", f"Expected A, got {res['root_cause']}"
    assert res["scores"]["B"]["classification"] == "PROPAGATED_VICTIM"
    print("TEST 2 (Upstream failure propagation) PASSED")


def test_three_level_propagation():
    """
    TEST 3:
    Three-level propagation
    A -> B -> C
    A fails first, then B, then C
    Expected: A remains root cause.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)
    G.add_edge("B", "C", request_count=100)
    
    analyzer = DirectionalCausalAnalyzer(G)
    
    # Tick 1: A fails
    analyzer.step(
        {"A": 5.0, "B": 0.0, "C": 0.0},
        {"A": {"Critical": 0.9}, "B": {"Critical": 0.0}, "C": {"Critical": 0.0}}
    )
    
    # Tick 2: B fails
    analyzer.step(
        {"A": 6.0, "B": 4.0, "C": 0.0},
        {"A": {"Critical": 0.9}, "B": {"Critical": 0.8}, "C": {"Critical": 0.0}}
    )
    
    # Tick 3: C fails, and B is highly anomalous
    res = analyzer.step(
        {"A": 6.0, "B": 7.0, "C": 5.0},
        {"A": {"Critical": 0.9}, "B": {"Critical": 0.9}, "C": {"Critical": 0.8}}
    )
    
    assert res["root_cause"] == "A", f"Expected A, got {res['root_cause']}"
    assert res["scores"]["B"]["classification"] == "PROPAGATED_VICTIM"
    assert res["scores"]["C"]["classification"] == "PROPAGATED_VICTIM"
    print("TEST 3 (Three-level propagation) PASSED")


def test_downstream_first_anomaly():
    """
    TEST 4:
    Downstream-first anomaly
    C anomalous first, A healthy.
    Expected: C should NOT be incorrectly attributed to A.
    """
    G = nx.DiGraph()
    G.add_edge("A", "C", request_count=100)
    
    analyzer = DirectionalCausalAnalyzer(G)
    
    # Tick 1: C fails
    res = analyzer.step(
        {"A": 0.0, "C": 5.0},
        {"A": {"Critical": 0.0}, "C": {"Critical": 0.9}}
    )
    
    assert res["root_cause"] == "C", f"Expected C, got {res['root_cause']}"
    assert res["scores"]["C"]["classification"] == "ROOT_CAUSE"
    assert res["scores"]["A"]["classification"] == "NORMAL"
    print("TEST 4 (Downstream-first anomaly) PASSED")


def test_recovery():
    """
    TEST 5:
    Recovery
    A recovers, then B/C recover.
    Expected: causal/victim scores decrease appropriately.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", request_count=100)
    
    analyzer = DirectionalCausalAnalyzer(G)
    
    # Tick 1: Both failing (A failed first)
    analyzer.history["A"]["degradation_start_tick"] = 1
    analyzer.current_tick = 1
    res = analyzer.step(
        {"A": 5.0, "B": 5.0},
        {"A": {"Critical": 0.9}, "B": {"Critical": 0.9}}
    )
    assert res["root_cause"] == "A"
    
    # Tick 2: A recovers, B is still slightly anomalous
    res = analyzer.step(
        {"A": 0.0, "B": 3.0},
        {"A": {"Critical": 0.0, "Normal": 1.0}, "B": {"Critical": 0.5, "Normal": 0.5}}
    )
    
    # A is no longer critical, B should be seen as independent or recovering
    assert res["scores"]["A"]["classification"] == "NORMAL"
    print("TEST 5 (Recovery) PASSED")

def test_existing_goal2_graph():
    """
    TEST 6:
    Load discovered_service_graph.json and verify RCA works.
    """
    graph_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/experiments/discovered_service_graph.json'))
    if not os.path.exists(graph_file):
        print("TEST 6 SKIPPED (No discovered_service_graph.json found)")
        return
        
    with open(graph_file, "r") as f:
        data = json.load(f)
        
    G = nx.DiGraph()
    for node in data.get("nodes", []):
        G.add_node(node)
    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["destination"], request_count=edge.get("request_count", 0))
        
    analyzer = DirectionalCausalAnalyzer(G)
    
    # Inject fault in ts-train-service
    res = analyzer.step(
        {"ts-train-service": 5.0},
        {"ts-train-service": {"Critical": 0.9}}
    )
    assert res["root_cause"] == "ts-train-service"
    
    # Downstream ts-route-service degrades
    res = analyzer.step(
        {"ts-train-service": 5.0, "ts-route-service": 4.0},
        {"ts-train-service": {"Critical": 0.9}, "ts-route-service": {"Critical": 0.8}}
    )
    assert res["root_cause"] == "ts-train-service"
    assert res["scores"]["ts-route-service"]["classification"] == "PROPAGATED_VICTIM"
    print("TEST 6 (Goal 2 graph integration) PASSED")

if __name__ == "__main__":
    print("Running Goal 3 Causality Tests...\n")
    test_independent_anomaly()
    test_upstream_failure_propagation()
    test_three_level_propagation()
    test_downstream_first_anomaly()
    test_recovery()
    test_existing_goal2_graph()
    print("\nALL TESTS PASSED.")
