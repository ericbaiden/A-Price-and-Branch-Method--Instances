"""Optional CSR implementation of the same exact A*/Dijkstra searches.

Build native/shortest_path.cpp as native/shortest_path.dll (MSVC /O2 /LD).
The Python implementation remains the portable fallback. A full pass always
uses unperturbed weights, and unexpected native errors propagate to the caller.
"""
import ctypes
from pathlib import Path
import numpy as np
import networkx as nx

LIBRARY = Path(__file__).resolve().parent / 'native/shortest_path.dll'


class NativePricing:
    def __init__(self, graph, sinks):
        self.graph = graph
        self.nodes = list(graph)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        edges, offsets = [], [0]
        for u in self.nodes:
            edges.extend((u, v) for v in graph[u])
            offsets.append(len(edges))
        self.edges = edges
        self.offset = np.asarray(offsets, dtype=np.int64)
        self.next = np.asarray([self.index[v] for _, v in edges], dtype=np.int32)
        self.base = np.asarray([graph.edges[edge]['weight'] for edge in edges], dtype=np.float64)
        self.imaging = np.asarray([graph.edges[edge]['arc_type'] == 'imaging_convergence_arc' for edge in edges])
        self.heuristics = {}
        for sink in set(sinks):
            dist = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), sink, weight='weight')
            self.heuristics[sink] = np.asarray([dist.get(node, 0.0) for node in self.nodes], dtype=np.float64)
        self.library = ctypes.CDLL(str(LIBRARY))
        self.search = self.library.shortest_path
        self.search.restype = ctypes.c_int
        self.search.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.weights = {}

    def prepare(self, duals, volumes):
        prices = np.asarray([duals.get(edge, 0.0) for edge in self.edges])
        self.weights = {volume: self.base - prices * np.where(self.imaging, 1.0, volume)
                        for volume in set(volumes)}

    def path(self, source, sink, volume, dijkstra, cutoff):
        route = np.empty(len(self.nodes), dtype=np.int32)
        cost = ctypes.c_double()
        explored = ctypes.c_int64()
        hptr = None if dijkstra else self.heuristics[sink].ctypes.data
        length = self.search(len(self.nodes), self.offset.ctypes.data, self.next.ctypes.data,
            self.weights[volume].ctypes.data, hptr, self.index[source], self.index[sink],
            cutoff, route.ctypes.data, ctypes.byref(cost), ctypes.byref(explored))
        if length < 0:
            raise RuntimeError(f'Native shortest-path failure {length}')
        if length == 0:
            raise nx.NetworkXNoPath(f'No improving path from {source} to {sink}')
        return cost.value, [self.nodes[i] for i in route[:length]], explored.value
