"""Fresh matched M=600 P&B / 20-start GRASP / 60-second ALNS experiments.

Run from this directory. Results include schedules, independent audits, source
hashes, graph hashes, environment, and convergence/operator records. A frozen
protocol prevents silent mixing of outputs after code or configuration changes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if 'GRB_LICENSE_FILE' not in os.environ:
    for license_location in ('D:/software/win64/bin/gurobi.lic',
                             'D:/software/Gurobi/win64/bin/gurobi.lic'):
        if Path(license_location).is_file():
            os.environ['GRB_LICENSE_FILE'] = license_location
            break

import gurobipy as gp
import networkx as nx
import numpy as np
import psutil

import experiments as ex
from Generator.instance_generator import TempNetwork
from Generator.satellite_generator import build_graph
from Solver.alns import run_alns, run_randomized_construction
from Solver.column_generation import ContinuousRelaxation, find_frac_flows, solve_last_iter_int, compute_true_objective
from alns_validation import audit_schedule

PHYSICAL_CACHE = {}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def source_hashes():
    paths = [Path(__file__), ROOT / 'experiments.py', ROOT / 'alns_validation.py']
    paths += sorted((ROOT / 'Solver').glob('*.py'))
    paths += sorted((ROOT / 'Generator').glob('*.py'))
    paths += sorted((ROOT / 'Solver/native').glob('*.cpp'))
    paths += sorted((ROOT / 'Solver/native').glob('*.dll'))
    paths += sorted((ROOT / 'data/tle').glob('*.tle'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_protocol(args):
    hashes = source_hashes()
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        'version': 1, 'penalty': 600, 'task_counts': args.tasks, 'instance_seeds': args.seeds,
        'rs_count': 136, 'cs_count': 1321, 'alns_random_seed': '10000 + instance_seed',
        'alns_time_limit': args.time_limit, 'alns_max_iterations': args.max_iterations,
        'grasp_restarts': 20, 'grasp_random_seed': '20000 + instance_seed',
        'destroy_operators': ['random', 'cost', 'bottleneck', 'conflict'],
        'repair_operators': ['randomized_priority_RCL_0.3', 'noisy_savings_0.8_1.2',
                             'regret2_sampled_shortlist_12_alternative_acquisition'],
        'removal_fraction': [0.05, 0.25], 'weight_segment': 25, 'reaction': 0.2,
        'operator_rewards': {'global_best': 8, 'current_improvement': 4, 'accepted': 1, 'rejected': 0},
        'annealing': 'T0=0.01*z_initial/log(2); T(t)=T0*0.001**(elapsed/time_limit)',
        'alns_budget': 'time_limit bounds the adaptive search only; residual-router setup, unconstrained path estimates and the initial construction are timed separately and reported as preprocessing_time and construction_time. GRASP is likewise bounded by its 20 restarts, not a clock, so neither heuristic is charged for bootstrapping.',
        'timing': 'solve time includes all algorithm setup, initialization, search, certification and integer recovery; input graph construction and independent validation excluded',
        'time_limit_check': 'between shortest-path calls; operator computation and one call may overrun slightly',
        'instances': 'original TLE generator; unit-volume tasks and common VGS; physical contacts cached, all task layers regenerated from seed',
        'comparators': 'fresh P&B A*+filter k=1 with exact full pricing certification; fresh 20-start residual GRASP using same exact shortest-path primitive as ALNS',
        'source_fingerprint': fingerprint, 'source_sha256': hashes,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'networkx': nx.__version__, 'numpy': np.__version__,
                        'gurobi': list(gp.gurobi.version()), 'ram_bytes': psutil.virtual_memory().total,
                        'cpu': platform.processor(), 'cpu_model': 'Intel Core Ultra 9 290HX Plus',
                        'logical_cpus': os.cpu_count(),
                        'execution': 'one case and one algorithm at a time; heuristics single-threaded; Gurobi Threads=8'},
    }


def build_instance(seed, count, task_data_volumes=None):
    # Shared physical contact data are immutable. No task-dependent arcs survive
    # into the template. The main generator still produces every task layer.
    if not PHYSICAL_CACHE:
        instance = ex.build_instance(seed, 136, 1321, count / 200)

        def physical_template(src):
            """Strip the task-dependent layers, keeping only immutable contacts."""
            tpl = copy.copy(src)
            tpl.temp_network = TempNetwork()
            excluded = {'source', 'sink', 'waiting'}
            tpl.temp_network.nodes = {n for n in src.temp_network.nodes
                                      if n.node_type not in excluded}
            tpl.temp_network.arcs = {a for a in src.temp_network.arcs
                                     if a.start.node_type not in excluded
                                     and a.end.node_type not in excluded}
            tpl.tasks, tpl.sources, tpl.sinks = [], {}, {}
            tpl.task_visibility, tpl.heuristic, tpl.heuristic_dist = {}, {}, {}
            tpl.incidence_matrix, tpl.inter_sat_visibility = {}, {}
            return tpl

        # ex.build_instance has already generated the task layer with the default
        # unit volume, so a volume override has to be applied by regenerating the
        # tasks on the cached physical template. The default path (None) keeps the
        # original behaviour byte-for-byte, which verify_volume_option.py checks
        # against the frozen hash b207ec99...
        template = physical_template(instance)
        template.subset = count / 200
        if task_data_volumes is not None:
            template.task_data_volumes = list(task_data_volumes)
        else:
            template.task_data_volumes = None
        template.generate_tasks(seed)
        template.set_arc_capacities()
        template.build_heuristic_dist()
        template.get_heuristic_fast()
        template.incidence_matrix, template.inter_sat_visibility = {}, {}
        PHYSICAL_CACHE['full'] = template
        # Hand back the freshly generated task layer. The object returned by
        # ex.build_instance still carries the default unit-volume tasks.
        instance = template
    else:
        instance = copy.copy(PHYSICAL_CACHE['full'])
        instance.subset = count / 200
        instance.task_data_volumes = list(task_data_volumes) if task_data_volumes is not None else None
        instance.temp_network = TempNetwork()
        instance.temp_network.nodes = set(PHYSICAL_CACHE['full'].temp_network.nodes)
        instance.temp_network.arcs = set(PHYSICAL_CACHE['full'].temp_network.arcs)
        instance.tasks, instance.sources, instance.sinks = [], {}, {}
        instance.task_visibility, instance.heuristic, instance.heuristic_dist = {}, {}, {}
        instance.generate_tasks(seed)
        instance.set_arc_capacities()
        instance.build_heuristic_dist()
        instance.get_heuristic_fast()
    instance.incidence_matrix, instance.inter_sat_visibility = {}, {}
    return instance


def canonical_graph(instance):
    old = build_graph(instance)
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(old.nodes))
    for u in sorted(old.nodes):
        for v in sorted(old[u]):
            d = old.edges[u, v]
            graph.add_edge(u, v, weight=d['weight'], cap=d['cap'], arc_type=d['arc_type'])
    graph.node_map = old.node_map
    old.clear()
    return graph


def instance_hash(instance, graph):
    digest = hashlib.sha256()
    for u in graph:
        digest.update((u + '\n').encode())
        for v, d in graph[u].items():
            digest.update(json.dumps([u, v, float(d['weight']), float(d['cap']), d['arc_type']],
                                     separators=(',', ':')).encode())
    for t in sorted(instance.tasks, key=lambda t: t.task_id):
        digest.update(json.dumps([t.task_id, float(t.data_volume), float(t.priority),
                                 ex._get_node_key(graph, instance.sources[t.task_id]),
                                 ex._get_node_key(graph, instance.sinks[t.task_id])]).encode())
    return digest.hexdigest()


def run_pb(instance, graph):
    start = time.perf_counter()
    cr = ContinuousRelaxation(graph, instance.tasks, instance.sources, instance.sinks,
                              600, instance.heuristic, instance.sat_params, k_alt=1)
    setup = time.perf_counter() - start
    cr.build_restricted_masterproblem()
    # solve() already ends with a full-pricing certification pass and records it in
    # cr.certification. A second identical pass would re-price every task to confirm
    # a result that was just established, so the certified state is read from there.
    cert = cr.solve(filter_on=True, dijkstra=False, max_iterations=200)
    if not cert.get('certified_no_new_columns'):
        raise RuntimeError('Column generation ended without a certificate')
    passes = cert['passes']
    lp_obj = float(cr.rmp.ObjVal)
    lower = lp_obj - 1e-6 * len(instance.tasks)
    certification_time = cert['time']
    integer_start = time.perf_counter()
    fractional = len(find_frac_flows(cr))
    if fractional:
        cr, value, _ = solve_last_iter_int(cr)
    else:
        value = compute_true_objective(cr)
    integer_time = time.perf_counter() - integer_start
    schedule = [{'task': flow.trip.task_id, 'nodes': list(flow.path[1])}
                for flow, var in cr.x.items() if var.X > 0.5]
    rejected = [t.task_id for t, var in cr.dummy_vars.items() if var.X > 0.5]
    result = {'obj': float(value), 'lp_obj': lp_obj, 'lower_bound': lower,
              'solve_time': time.perf_counter() - start, 'preprocessing_time': setup,
              'certification_time': certification_time, 'integer_recovery_time': integer_time,
              'certified_no_negative_columns': True, 'certification_passes': passes,
              'fractional_flows': fractional, 'completed': len(schedule), 'rejected': len(rejected),
              'schedule': schedule, 'rejected_tasks': rejected,
              'num_columns': len(cr.x), 'iterations': cr.num_it,
              'gap_pct': 100 * (value - lower) / lower if lower > 0 else None}
    cr.rmp.dispose()
    return result


def run_case(count, seed, args, protocol):
    path = args.output / f'tasks_{count:04d}_seed_{seed:02d}.json'
    if path.is_file():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached.get('source_fingerprint') != protocol['source_fingerprint']:
            raise RuntimeError(f'Fingerprint mismatch: {path}')
        if cached.get('status') == 'complete':
            print(f'CACHED tasks={count} seed={seed}', flush=True)
            return
    start = time.perf_counter()
    print(f'START tasks={count} seed={seed}', flush=True)
    instance = build_instance(seed, count)
    graph = canonical_graph(instance)
    build_time = time.perf_counter() - start
    if len(instance.tasks) != count:
        raise RuntimeError('Requested task count does not match generated instance')
    record = {'status': 'running', 'started_at': datetime.now().isoformat(),
              'tasks': count, 'seed': seed, 'penalty': 600,
              'source_fingerprint': protocol['source_fingerprint'],
              'instance_sha256': instance_hash(instance, graph),
              'graph_nodes': len(graph), 'graph_edges': graph.number_of_edges(),
              'build_time': build_time, 'algorithms': {}}
    write_json(path, record)
    methods = [('P&B', lambda: run_pb(instance, graph)),
               ('GRASP', lambda: run_randomized_construction(instance, graph, penalty=600,
                                                            seed=20000 + seed, restarts=20)),
               ('ALNS', lambda: run_alns(instance, graph, penalty=600, seed=10000 + seed,
                                        time_limit=args.time_limit, max_iterations=args.max_iterations,
                                        stall_iterations=args.stall_iterations))]
    for name, method in methods:
        result = method()
        audit_start = time.perf_counter()
        result['audit'] = audit_schedule(instance, graph, result, 600)
        result['validation_time'] = time.perf_counter() - audit_start
        if not result['audit']['feasible']:
            raise RuntimeError(f'{name} failed independent validation: {result["audit"]}')
        record['algorithms'][name] = result
        write_json(path, record)
        print(f'  {name}: obj={result["obj"]:.1f} completed={result["completed"]} '
              f'time={result["solve_time"]:.3f}s iterations={result.get("iterations", "-")}', flush=True)
        gc.collect()
    lower = record['algorithms']['P&B']['lower_bound']
    for result in record['algorithms'].values():
        if result['obj'] < lower - 1e-5:
            raise RuntimeError('Heuristic objective below certified bound')
        result['gap_to_certified_bound_pct'] = 100 * (result['obj'] - lower) / lower
    record.update(status='complete', finished_at=datetime.now().isoformat(),
                  total_time=time.perf_counter() - start,
                  peak_rss_bytes=psutil.Process().memory_info().peak_wset)
    write_json(path, record)
    print(f'DONE tasks={count} seed={seed} total={record["total_time"]:.1f}s', flush=True)
    graph.clear()
    del graph, instance
    gc.collect()


def parse_list(value):
    return [int(i) for i in value.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=parse_list, default=[100, 500, 1000])
    parser.add_argument('--seeds', type=parse_list, default=list(range(10)))
    parser.add_argument('--time-limit', type=float, default=60)
    parser.add_argument('--stall-iterations', type=int, default=None,
                        help='End the search after this many consecutive iterations '
                             'without an incumbent improvement (None disables).')
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/alns_m600')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(args)
    frozen = args.output / 'protocol.json'
    if frozen.is_file():
        previous = json.loads(frozen.read_text(encoding='utf-8'))
        if previous != protocol:
            raise RuntimeError('Frozen protocol changed: use a separate output directory')
    else:
        write_json(frozen, protocol)
    print(f'PROTOCOL {protocol["source_fingerprint"]} output={args.output}', flush=True)
    errors = []
    for count in args.tasks:
        for seed in args.seeds:
            try:
                run_case(count, seed, args, protocol)
            except Exception:
                detail = traceback.format_exc()
                errors.append({'tasks': count, 'seed': seed, 'traceback': detail})
                print(detail, flush=True)
                write_json(args.output / 'errors.json', errors)
                # Never continue a long campaign after a correctness/runtime failure.
                raise
    print('ALL CASES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
