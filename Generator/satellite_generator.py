"""
Satellite network graph builder.

Based on intermodalTransportationNetworksCG graph_generator.py architecture.
Converts Instance into networkx graph for column generation.

Reference: README Section 5.4 graph building

Uses string node IDs to avoid TempNode identity issues with networkx.
"""

import networkx as nx
from typing import Dict

from .instance_generator import Instance, TempNode, TempArc


def build_graph(instance: Instance) -> nx.DiGraph:
    """Build networkx graph from Instance.

    Reference: intermodalTransportationNetworksCG graph_generator.build_graph

    Uses string node IDs derived from TempNode attributes to ensure
    consistent node identification across the graph and instance.sources/sinks.

    Args:
        instance: Instance object with temp_network

    Returns:
        networkx DiGraph with edge attributes:
        - weight: arc cost
        - cap: arc capacity
        - arc_type: type of arc
        - energy_consumption: energy used by arc
        - start_node: original TempNode (for reference)
        - end_node: original TempNode (for reference)
    """
    G = nx.DiGraph()

    # Create node key mapping
    node_map = {}  # TempNode -> string key

    def get_node_key(node: TempNode) -> str:
        """Get or create string key for a TempNode."""
        if node not in node_map:
            node_map[node] = f"{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}"
        return node_map[node]

    # Add all nodes from arcs
    for arc in instance.temp_network.arcs:
        start_key = get_node_key(arc.start)
        end_key = get_node_key(arc.end)

        G.add_edge(
            start_key,
            end_key,
            weight=arc.cost,
            cap=arc.capacity,
            arc_type=arc.arc_type,
            energy_consumption=arc.energy_consumption,
            start_node=arc.start,
            end_node=arc.end
        )

    # Also add any standalone nodes (source/sink nodes that have no arcs)
    for node in instance.temp_network.nodes:
        node_key = get_node_key(node)
        if node_key not in G.nodes():
            G.add_node(node_key, temp_node=node)

    # Build reverse mapping: string key -> TempNode
    G.node_map = node_map

    return G


def get_temp_node(graph: nx.DiGraph, node_key: str) -> TempNode:
    """Get the original TempNode from a graph node key."""
    return graph.nodes[node_key].get('temp_node')


def build_heuristic(instance: Instance) -> Dict:
    """Build heuristic dictionary for A* search.

    Reference: README Section 3.5 acceleration mechanism 1 (A*启发式函数)

    Heuristic: estimated minimum delay from state node to any sink.
    Since sink connects to virtual_gs -> ground stations, the heuristic
    is the minimum possible downlink delay (altitude / orbital_speed).

    Args:
        instance: Instance object

    Returns:
        Dict: node_key -> {target_key -> estimated_delay}
    """
    heuristic = {}

    # Get sink and ground station node keys
    sink_keys = set()
    gs_keys = set()
    for node in instance.temp_network.nodes:
        key = f"{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}"
        if node.node_type == 'sink':
            sink_keys.add(key)
        elif node.node_type == 'ground':
            gs_keys.add(key)
        elif node.node_type == 'virtual_ground':
            sink_keys.add(key)

    def get_node_key(node):
        return f"{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}"

    # For each state node, estimate delay to sinks
    for node in instance.temp_network.nodes:
        if node.node_type != 'state':
            continue

        node_key = get_node_key(node)
        sat_alt = node.coordinate[2] if len(node.coordinate) == 3 else 500.0
        # Minimum possible delay: time to downlink to nearest ground station
        # This is admissible since any path must eventually go through a GS
        min_delay = sat_alt / 7.0 / 60.0  # minutes (altitude / 7km/s / 60s)

        # Add same heuristic for all sinks and ground stations
        heuristic[node_key] = {}
        for sink_key in sink_keys:
            heuristic[node_key][sink_key] = min_delay
        for gs_key in gs_keys:
            heuristic[node_key][gs_key] = min_delay

    return heuristic


if __name__ == '__main__':
    # Test graph building
    from .instance_generator import Instance, TimeWindow, TLEParser, DEFAULT_GS_CONFIG

    parser = TLEParser()
    sats = parser._get_sample_satellites()

    time_window = TimeWindow(start=0.0, end=60.0)

    instance = Instance(
        time_window=time_window,
        rs_count=5,
        cs_count=15,
        gs_config=DEFAULT_GS_CONFIG,
        tle_data=sats[:5],
        subset=0.1
    )

    instance.build_instance(seed=42)

    G = build_graph(instance)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Check source nodes
    print("\nSource nodes:")
    for task_id, src in list(instance.sources.items())[:3]:
        src_key = f"{src.satellite}_{src.time_slot}_{src.node_type}_{src.route}"
        print(f"  {task_id}: {src_key} (in graph: {src_key in G.nodes()})")

    heuristic = build_heuristic(instance)
    print(f"\nHeuristic entries: {len(heuristic)}")