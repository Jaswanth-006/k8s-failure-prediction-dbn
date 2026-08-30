import json
import os
import requests
import networkx as nx

PROMETHEUS_URL = "http://localhost:9090"

SERVICES = {
    "ts-ui-dashboard",
    "ts-user-service",
    "ts-train-service",
    "ts-route-service",
    "ts-order-service",
    "ts-payment-service",
    "ts-inventory-service",
    "ts-station-service",
}

OUTPUT_FILE = "data/experiments/discovered_service_graph.json"


def query_prometheus(query: str):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=5,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(
            f"Prometheus query failed: {data}"
        )

    return data["data"]["result"]


def discover_service_graph():

    query = """
    sum by (source_workload, destination_workload) (
        istio_requests_total{reporter="source"}
    )
    """

    results = query_prometheus(query)

    graph = nx.DiGraph()

    # Add all known services as nodes.
    for service in SERVICES:
        graph.add_node(service)

    for item in results:

        metric = item.get("metric", {})

        source = metric.get("source_workload")
        destination = metric.get("destination_workload")

        if not source or not destination:
            continue

        # Ignore services outside our TrainTicket topology.
        if source not in SERVICES:
            continue

        if destination not in SERVICES:
            continue

        # Ignore self-calls.
        if source == destination:
            continue

        request_count = float(item["value"][1])

        graph.add_edge(
            source,
            destination,
            request_count=request_count,
        )

    return graph


def save_graph(graph):

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    data = {
        "nodes": list(graph.nodes()),
        "edges": [
            {
                "source": source,
                "destination": destination,
                "request_count": graph[source][destination][
                    "request_count"
                ],
            }
            for source, destination in graph.edges()
        ],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    return data


def main():

    print("=" * 70)
    print("PREFACE-DBN — GOAL 2")
    print("Automatic Microservice Dependency Discovery")
    print("=" * 70)

    print("\nQuerying Prometheus...")

    graph = discover_service_graph()

    print("\nDiscovered dependency edges:")
    print("-" * 70)

    for source, destination in graph.edges():

        request_count = graph[source][destination][
            "request_count"
        ]

        print(
            f"{source:25s} -> "
            f"{destination:25s} "
            f"(requests={request_count:.0f})"
        )

    print("\nGraph summary:")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\nNetworkX edges:")

    for source, destination in graph.edges():

        print(
            f"  {source} -> {destination}"
        )

    save_graph(graph)

    print("\nGraph saved to:")
    print(f"  {OUTPUT_FILE}")

    print("\nSUCCESS: dependency graph discovered automatically.")


if __name__ == "__main__":
    main()