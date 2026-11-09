"""Frozen instance store: build each scheduling instance once, then always load it.

Why this exists
---------------
`run_alns_benchmark.build_instance` caches the immutable physical contact layer in a
module-level `PHYSICAL_CACHE` seeded by *whichever instance is requested first*. Two
campaigns that request the same instances in a different order therefore get different
physical layers, and hence different instances under the same (seed, count). That is
not reproducible, and it silently mixed two instance families inside one result table.

The store removes the question entirely: every instance is generated once, hashed, and
written to `dataset/instances/`. Campaign runners load from the store instead of
regenerating. The recorded hash is the same `instance_hash` the runners use, so a
stored instance can be checked against any existing result record.

What is stored
--------------
Only what the solver and the heuristics actually read: the canonical graph and the task
layer (id, volume, priority, imaging window, and the source/sink node descriptors).
`ContinuousRelaxation` recomputes the A* heuristic from the graph itself, so the
heuristic is deliberately not stored. Node `coordinate` is not stored either: it is
consumed only during graph construction, and every lookup afterwards goes through the
(satellite, time_slot, node_type, route) key.

Layout
------
    dataset/instances/tasks_0100_seed_00.npz
    dataset/instances/manifest.json      # per-instance hash + provenance

Usage
-----
    from instance_store import save_instance, load_instance
    save_instance(instance, graph, tasks=100, seed=0, instance_sha256=sha)
    inst = load_instance(tasks=100, seed=0)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
STORE = ROOT / 'dataset' / 'instances'

# The horizon the main study uses. Instances built at any other horizon go into a
# `h<hours>` subdirectory so the two can never be confused.
HORIZON_DEFAULT = 3.0

@dataclass
class StoreInstance:
    """The subset of an Instance that the solver, GRASP and ALNS actually use.

    The remaining attributes mirror `Generator.Instance` so the campaign runners and
    the solvers can treat the two interchangeably. Only `tasks`, `sources`, `sinks`,
    `sat_params` and `graph` carry data; the rest are inert placeholders. In
    particular `heuristic` is empty on purpose: `ContinuousRelaxation` recomputes the
    A* delivery-distance approximation from the graph in its constructor.
    """
    tasks: List[object]
    sources: Dict[str, object]          # task_id -> TempNode
    sinks: Dict[str, object]            # task_id -> TempNode
    sat_params: Dict[str, object]
    graph: object                       # networkx.DiGraph with .node_map
    tasks_count: int
    seed: int
    instance_sha256: str
    node_map: Dict = field(default_factory=dict)
    heuristic: Dict = field(default_factory=dict)
    heuristic_dist: Dict = field(default_factory=dict)
    task_visibility: Dict = field(default_factory=dict)
    incidence_matrix: Dict = field(default_factory=dict)
    inter_sat_visibility: Dict = field(default_factory=dict)
    temp_network: object = None
    task_data_volumes: object = None
    rs_count: int = 136
    cs_count: int = 1321
    horizon_hours: float = 3.0

    @property
    def subset(self) -> float:
        """Fraction of the 200-task base subset, as the generator defines it."""
        return self.tasks_count / 200


def _path(tasks: int, seed: int, horizon_hours: float = HORIZON_DEFAULT) -> Path:
    """Store path for one instance.

    Instances from different planning horizons live in separate directories: a
    horizon changes the whole space-time graph, so a file from one horizon is not
    interchangeable with a file from another.
    """
    base = STORE if horizon_hours == HORIZON_DEFAULT else STORE / f'h{horizon_hours:g}'
    return base / f'tasks_{tasks:04d}_seed_{seed:02d}.npz'


def _key_parts(node):
    """The four fields that make up a node's key, from a TempNode or a string."""
    if isinstance(node, str):
        return None
    return (node.satellite, int(node.time_slot), node.node_type, node.route)


def save_instance(instance, graph, tasks: int, seed: int, instance_sha256: str,
                  horizon_hours: float = HORIZON_DEFAULT) -> Path:
    """Write one instance to the store and refresh its manifest entry."""
    dest = _path(tasks, seed, horizon_hours)
    dest.parent.mkdir(parents=True, exist_ok=True)

    node_keys = sorted(graph.nodes)
    index = {k: i for i, k in enumerate(node_keys)}

    # The arc-type vocabulary is derived from this graph and written alongside it, so
    # a type the code has never seen cannot be silently dropped during encoding.
    vocab = sorted({d.get('arc_type', '') for u in node_keys for d in graph[u].values()})
    code = {a: i for i, a in enumerate(vocab)}

    src, dst, weight, cap, arc = [], [], [], [], []
    for u in node_keys:
        for v, d in graph[u].items():
            src.append(index[u])
            dst.append(index[v])
            weight.append(float(d['weight']))
            cap.append(float(d['cap']))
            arc.append(code[d.get('arc_type', '')])

    ordered = sorted(instance.tasks, key=lambda t: t.task_id)
    task_ids = [t.task_id for t in ordered]
    src_nodes = [_key_parts(instance.sources[t.task_id]) for t in ordered]
    snk_nodes = [_key_parts(instance.sinks[t.task_id]) for t in ordered]
    for i, (s, k) in enumerate(zip(src_nodes, snk_nodes)):
        if s is None or k is None:
            raise TypeError(f'Expected TempNode endpoints for task {task_ids[i]}')

    np.savez_compressed(
        _path(tasks, seed, horizon_hours),
        node_keys=np.array(node_keys, dtype=object),
        edge_src=np.asarray(src, dtype=np.int64),
        edge_dst=np.asarray(dst, dtype=np.int64),
        edge_weight=np.asarray(weight, dtype=np.float64),
        edge_cap=np.asarray(cap, dtype=np.float64),
        edge_arc=np.asarray(arc, dtype=np.int16),
        arc_vocab=np.array(vocab, dtype=object),
        task_ids=np.array(task_ids, dtype=object),
        task_lat=np.asarray([t.target_lat for t in ordered], dtype=np.float64),
        task_lon=np.asarray([t.target_lon for t in ordered], dtype=np.float64),
        task_volume=np.asarray([t.data_volume for t in ordered], dtype=np.float64),
        task_priority=np.asarray([t.priority for t in ordered], dtype=np.float64),
        task_release=np.asarray([t.release_slot for t in ordered], dtype=np.int64),
        task_deadline=np.asarray([t.deadline_slot for t in ordered], dtype=np.int64),
        task_duration=np.asarray([t.duration for t in ordered], dtype=np.float64),
        task_src=np.array([json.dumps(s) for s in src_nodes], dtype=object),
        task_snk=np.array([json.dumps(k) for k in snk_nodes], dtype=object),
        sat_ids=np.array(sorted(instance.sat_params), dtype=object),
        meta=np.array(json.dumps({'tasks': tasks, 'seed': seed,
                                  'instance_sha256': instance_sha256,
                                  'nodes': len(node_keys), 'edges': len(src)})),
    )
    _update_manifest(tasks, seed, instance_sha256, len(node_keys), len(src),
                     horizon_hours)
    return _path(tasks, seed, horizon_hours)


def _update_manifest(tasks, seed, sha, nodes, edges,
                     horizon_hours=HORIZON_DEFAULT):
    base = STORE if horizon_hours == HORIZON_DEFAULT else STORE / f'h{horizon_hours:g}'
    base.mkdir(parents=True, exist_ok=True)
    path = base / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    manifest[f'{tasks:04d}_{seed:02d}'] = {
        'tasks': tasks, 'seed': seed, 'instance_sha256': sha,
        'graph_nodes': nodes, 'graph_edges': edges,
        'horizon_hours': horizon_hours,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')


def has_instance(tasks: int, seed: int, horizon_hours: float = HORIZON_DEFAULT) -> bool:
    return _path(tasks, seed, horizon_hours).is_file()


def load_instance(tasks: int, seed: int,
                  horizon_hours: float = HORIZON_DEFAULT) -> StoreInstance:
    """Rebuild the stored instance. Deterministic: reads only the frozen file."""
    import networkx as nx
    from Generator.instance_generator import TempNode, SatelliteParams, Task

    path = _path(tasks, seed, horizon_hours)
    if not path.is_file():
        raise FileNotFoundError(f'Instance not in store: {path}')

    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    node_keys = [str(k) for k in z['node_keys']]
    arc_vocab = [str(a) for a in z['arc_vocab']]

    graph = nx.DiGraph()
    graph.add_nodes_from(node_keys)
    src, dst = z['edge_src'], z['edge_dst']
    weight, cap, arc = z['edge_weight'], z['edge_cap'], z['edge_arc']
    for i in range(len(src)):
        graph.add_edge(node_keys[src[i]], node_keys[dst[i]],
                       weight=float(weight[i]), cap=float(cap[i]),
                       arc_type=arc_vocab[int(arc[i])])
    graph.node_map = {}

    def rebuild(blob):
        sat, slot, ntype, route = json.loads(str(blob))
        return TempNode(satellite=sat, time_slot=int(slot), node_type=ntype,
                        route=route, coordinate=(0.0, 0.0, 0.0))

    task_ids = [str(t) for t in z['task_ids']]
    store_tasks = [Task(task_id=task_ids[i],
                        target_lat=float(z['task_lat'][i]),
                        target_lon=float(z['task_lon'][i]),
                        data_volume=float(z['task_volume'][i]),
                        priority=float(z['task_priority'][i]),
                        release_slot=int(z['task_release'][i]),
                        deadline_slot=int(z['task_deadline'][i]),
                        duration=float(z['task_duration'][i]))
                   for i in range(len(task_ids))]
    sources = {task_ids[i]: rebuild(z['task_src'][i]) for i in range(len(task_ids))}
    sinks = {task_ids[i]: rebuild(z['task_snk'][i]) for i in range(len(task_ids))}
    sat_params = {str(s): SatelliteParams(sat_id=str(s),
                                          sat_type='RS' if str(s).startswith('RS') else 'CS')
                  for s in z['sat_ids']}

    return StoreInstance(tasks=store_tasks, sources=sources, sinks=sinks,
                         sat_params=sat_params, graph=graph,
                         tasks_count=meta['tasks'], seed=meta['seed'],
                         instance_sha256=meta['instance_sha256'])


def verify_all(verbose: bool = False) -> int:
    """Check every stored instance against its manifest hash. Returns failures."""
    from run_alns_benchmark import instance_hash
    manifest = json.loads((STORE / 'manifest.json').read_text(encoding='utf-8'))
    bad = 0
    for key, rec in sorted(manifest.items()):
        inst = load_instance(rec['tasks'], rec['seed'])
        got = instance_hash(inst, inst.graph)
        ok = got == rec['instance_sha256']
        if not ok:
            bad += 1
        if verbose or not ok:
            print(f'  {"OK  " if ok else "FAIL"} {key}: {rec["instance_sha256"][:16]}'
                  f'{"  recomputed=" + got[:16] if not ok else ""}')
    print(f'  verified {len(manifest)} instances, {bad} mismatches')
    return bad


if __name__ == '__main__':
    raise SystemExit(1 if verify_all(verbose=True) else 0)
