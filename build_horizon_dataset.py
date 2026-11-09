#!/usr/bin/env python3
"""Build and store instances for an extended planning horizon.

The physical contact layer is orbit-determined and independent of the task count, so
it is built once per horizon, written to `dataset/physical/`, and reused by every task
count and seed afterwards. Only the task layer is regenerated per instance.

Usage
-----
    python build_horizon_dataset.py --hours 24 --tasks 1000 2000 5000 --seeds 0 1 2

Each instance is written through `instance_store`, which keys the file by horizon, so
a 24-hour instance can never be mistaken for a three-hour one.
"""
from __future__ import annotations

import argparse
import copy
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

import experiments as ex
import physical_cache
from Generator.instance_generator import TempNetwork, TimeWindow
import Generator.instance_generator as generator
import run_alns_benchmark as bench
from instance_store import save_instance, has_instance

RS, CS = 136, 1321


def install_expanded_generator():
    """Raise the candidate-target budget for large task counts.

    `generate_tasks` samples target coordinates and keeps the visible ones, with a
    fixed 5000-attempt budget. That is enough up to about 2000 tasks; beyond it the
    generator silently returns fewer tasks than requested (2709 of 5000 at a 24-hour
    horizon). The manuscript's 2000--10000 experiments already use the relaxed budget
    below, so applying it here keeps the long-horizon suite comparable with them.
    """
    from run_scalability_m200 import expanded_generator_source
    import textwrap
    ns = dict(generator.__dict__)
    exec(compile(textwrap.dedent(expanded_generator_source()),
                 '<expanded_generate_tasks>', 'exec'), ns)
    generator.Instance.generate_tasks = ns['generate_tasks']


def build_template(hours: float):
    """Physical contact layer for one horizon, cached on disk across processes."""
    cached = physical_cache.load(hours, RS, CS)
    if cached is not None:
        print(f'template loaded from disk ({hours:g} h)', flush=True)
        return cached

    print(f'building physical template for {hours:g} h...', flush=True)
    ex.TIME_WINDOW = TimeWindow(start=0.0, end=hours * 60.0)
    t0 = time.perf_counter()
    # The subset only sets task count; the contact layer is what we want here.
    inst = ex.build_instance(0, RS, CS, 1.0)

    tpl = copy.copy(inst)
    tpl.temp_network = TempNetwork()
    excluded = {'source', 'sink', 'waiting'}
    tpl.temp_network.nodes = {n for n in inst.temp_network.nodes
                              if n.node_type not in excluded}
    tpl.temp_network.arcs = {a for a in inst.temp_network.arcs
                             if a.start.node_type not in excluded
                             and a.end.node_type not in excluded}
    tpl.tasks, tpl.sources, tpl.sinks = [], {}, {}
    tpl.task_visibility, tpl.heuristic, tpl.heuristic_dist = {}, {}, {}
    tpl.incidence_matrix, tpl.inter_sat_visibility = {}, {}

    path = physical_cache.save(tpl, hours, RS, CS)
    print(f'template built in {time.perf_counter()-t0:.0f}s -> {path.name}', flush=True)
    return tpl


def make_instance(template, hours: float, count: int, seed: int):
    """Task layer for (count, seed) on the stored physical template."""
    import run_alns_benchmark_m600 as campaign   # sets the generator path it expects
    inst = copy.copy(template)
    inst.subset = count / 200
    inst.task_data_volumes = None
    inst.temp_network = TempNetwork()
    inst.temp_network.nodes = set(template.temp_network.nodes)
    inst.temp_network.arcs = set(template.temp_network.arcs)
    inst.tasks, inst.sources, inst.sinks = [], {}, {}
    inst.task_visibility, inst.heuristic, inst.heuristic_dist = {}, {}, {}
    inst.generate_tasks(seed)
    inst.set_arc_capacities()
    inst.build_heuristic_dist()
    inst.get_heuristic_fast()
    inst.incidence_matrix, inst.inter_sat_visibility = {}, {}
    return inst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hours', type=float, required=True)
    ap.add_argument('--tasks', type=int, nargs='+', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()

    install_expanded_generator()
    template = build_template(args.hours)
    for count in args.tasks:
        for seed in args.seeds:
            if has_instance(count, seed, args.hours):
                print(f'  skip {count}/{seed} (stored)', flush=True)
                continue
            t0 = time.perf_counter()
            inst = make_instance(template, args.hours, count, seed)
            graph = bench.canonical_graph(inst)
            if len(inst.tasks) != count:
                raise RuntimeError(f'Requested {count}, generated {len(inst.tasks)}')
            sha = bench.instance_hash(inst, graph)
            save_instance(inst, graph, count, seed, sha, args.hours)
            print(f'  {count} tasks seed {seed}: {len(graph):,} nodes '
                  f'{graph.number_of_edges():,} arcs, {time.perf_counter()-t0:.0f}s',
                  flush=True)

    # Release the template's memory before the caller reuses the process.
    template.temp_network.nodes.clear()
    template.temp_network.arcs.clear()
    print('HORIZON DATASET COMPLETE', flush=True)


if __name__ == '__main__':
    main()
