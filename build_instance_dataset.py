#!/usr/bin/env python3
"""Build the frozen instance dataset, once, in a deterministic order.

The physical contact layer is seeded from one fixed call, `ex.build_instance(0,
136, 1321, 1.0)`, so it no longer depends on which instance happens to be requested
first. Every instance is then generated from that template, hashed, and written to
`dataset/instances/` by `instance_store`.

Re-running this script is a no-op for instances already present, so an interrupted
build can simply be restarted.

The stored hash uses the same `instance_hash` the campaign runners record, which lets
any existing result file be traced back to the instance it was computed on.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'Solver'))
sys.path.insert(2, str(ROOT / 'Generator'))

if 'GRB_LICENSE_FILE' not in os.environ:
    for lic in ('D:/software/win64/bin/gurobi.lic',
                'D:/software/Gurobi/win64/bin/gurobi.lic'):
        if Path(lic).is_file():
            os.environ['GRB_LICENSE_FILE'] = lic
            break

import copy

import experiments as ex
from Generator.instance_generator import TempNetwork
import run_alns_benchmark as bench
from instance_store import save_instance, has_instance

# The one call that defines the physical contact layer for every instance.
TEMPLATE_SEED = 0
TEMPLATE_COUNT = 200        # subset = 1.0

TASK_COUNTS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
SEEDS = tuple(range(10))

PHYSICAL = {}


def build_template():
    """Generate the physical layer exactly once, from a fixed call."""
    instance = ex.build_instance(TEMPLATE_SEED, 136, 1321, TEMPLATE_COUNT / 200)
    tpl = copy.copy(instance)
    tpl.temp_network = TempNetwork()
    excluded = {'source', 'sink', 'waiting'}
    tpl.temp_network.nodes = {n for n in instance.temp_network.nodes
                              if n.node_type not in excluded}
    tpl.temp_network.arcs = {a for a in instance.temp_network.arcs
                             if a.start.node_type not in excluded
                             and a.end.node_type not in excluded}
    tpl.tasks, tpl.sources, tpl.sinks = [], {}, {}
    tpl.task_visibility, tpl.heuristic, tpl.heuristic_dist = {}, {}, {}
    tpl.incidence_matrix, tpl.inter_sat_visibility = {}, {}
    PHYSICAL['full'] = tpl
    print(f'template built from (seed={TEMPLATE_SEED}, count={TEMPLATE_COUNT})', flush=True)


def make_instance(count: int, seed: int):
    """Task layer for (count, seed) on the fixed physical template."""
    tpl = PHYSICAL['full']
    inst = copy.copy(tpl)
    inst.subset = count / 200
    inst.task_data_volumes = None
    inst.temp_network = TempNetwork()
    inst.temp_network.nodes = set(tpl.temp_network.nodes)
    inst.temp_network.arcs = set(tpl.temp_network.arcs)
    inst.tasks, inst.sources, inst.sinks = [], {}, {}
    inst.task_visibility, inst.heuristic, inst.heuristic_dist = {}, {}, {}
    inst.generate_tasks(seed)
    inst.set_arc_capacities()
    inst.build_heuristic_dist()
    inst.get_heuristic_fast()
    inst.incidence_matrix, inst.inter_sat_visibility = {}, {}
    return inst


def main():
    build_template()
    todo = [(n, s) for n in TASK_COUNTS for s in SEEDS if not has_instance(n, s)]
    done = len(TASK_COUNTS) * len(SEEDS) - len(todo)
    print(f'{done} instances already stored; {len(todo)} to build', flush=True)

    for i, (count, seed) in enumerate(todo, 1):
        t0 = time.perf_counter()
        inst = make_instance(count, seed)
        graph = bench.canonical_graph(inst)
        if len(inst.tasks) != count:
            raise RuntimeError(f'Requested {count} tasks, generated {len(inst.tasks)}')
        sha = bench.instance_hash(inst, graph)
        path = save_instance(inst, graph, count, seed, sha)
        print(f'[{i}/{len(todo)}] tasks={count} seed={seed} '
              f'nodes={len(graph)} edges={graph.number_of_edges()} '
              f'sha={sha[:12]} {time.perf_counter()-t0:.1f}s', flush=True)

    print('DATASET COMPLETE', flush=True)


if __name__ == '__main__':
    main()
