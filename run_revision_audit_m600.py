"""Reproducible EJOR audit campaign at M=600.

The revised manuscript states that every algorithm comparison uses M=600, but
the consolidated audit (results/revision_audit_m200) was run at M=200. This is
the same frozen driver with the penalty raised and a separate output tree, so
the main and validation blocks come from one protocol instead of two.

Run: python run_revision_audit_m600.py --block main --tasks 100-1000 --seeds 0-9
Output: results/revision_audit_m600/{main,validation,ablation,sensitivity}/
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
# Use the installed local license when the caller has not selected one.
for license_path in (Path('D:/software/win64/bin/gurobi.lic'), Path('D:/software/Gurobi/win64/bin/gurobi.lic')):
    if license_path.is_file() and 'GRB_LICENSE_FILE' not in os.environ:
        os.environ['GRB_LICENSE_FILE'] = str(license_path)
        break

import gurobipy as gp
import networkx as nx
import numpy as np
import experiments as ex
from Generator.satellite_generator import build_graph
from Generator.instance_generator import TempNetwork
from Solver.column_generation import (ContinuousRelaxation, CAPACITY_ARC_TYPES,
    resource_coefficient, find_frac_flows, solve_last_iter_int, compute_true_objective)
from Solver.branch_and_price import solve_branch_and_price

OUT = ROOT / 'results/revision_audit_m600'
PB = 'CG+A* (filter,k=1)'
NF = 'CG+A* (no-filter,k=1)'
DJ = 'CG+Dijkstra (filter,k=1)'
_INSTANCE_CACHE = None
_PHYSICAL_CACHE = {}


def fingerprint():
    digest = hashlib.sha256()
    paths = [ROOT / 'experiments.py', Path(__file__).resolve()]
    paths += sorted((ROOT / 'Solver').glob('*.py')) + sorted((ROOT / 'Generator').glob('*.py'))
    paths += sorted((ROOT / 'data/tle').glob('*.tle'))
    paths += sorted((ROOT / 'Solver/native').glob('*.cpp'))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_audit_instance(seed, rs, cs, tasks):
    # Full-constellation satellites and contacts are independent of the task RNG.
    # Reduced constellations sample satellites, so their key also includes seed.
    cache_key = rs, cs, seed if rs < 136 or cs < 1321 else None
    template = _PHYSICAL_CACHE.get(cache_key)
    if template is None:
        instance = ex.build_instance(seed, rs, cs, tasks / 200)
        template = copy.copy(instance)
        template.temp_network = TempNetwork()
        excluded = {'source', 'sink', 'waiting'}
        template.temp_network.nodes = {n for n in instance.temp_network.nodes if n.node_type not in excluded}
        template.temp_network.arcs = {a for a in instance.temp_network.arcs
            if a.start.node_type not in excluded and a.end.node_type not in excluded}
        template.tasks, template.sources, template.sinks = [], {}, {}
        template.task_visibility, template.heuristic, template.heuristic_dist = {}, {}, {}
        template.incidence_matrix, template.inter_sat_visibility = {}, {}
        _PHYSICAL_CACHE[cache_key] = template
    else:
        instance = copy.copy(template)
        instance.subset = tasks / 200
        instance.temp_network = TempNetwork()
        instance.temp_network.nodes = set(template.temp_network.nodes)
        instance.temp_network.arcs = set(template.temp_network.arcs)
        instance.tasks, instance.sources, instance.sinks = [], {}, {}
        instance.task_visibility, instance.heuristic, instance.heuristic_dist = {}, {}, {}
        instance.generate_tasks(seed)
        instance.set_arc_capacities()
        instance.build_heuristic_dist()
        instance.get_heuristic_fast()
    # These construction-only dictionaries are not consumed by either solver.
    instance.incidence_matrix = {}
    instance.inter_sat_visibility = {}
    return instance


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def audit_schedule(instance, graph, selected, rejected, penalty, objective):
    loads = {}
    seen = set()
    valid = True
    cost = 0.0
    task_map = {t.task_id: t for t in instance.tasks}
    for task_id, nodes in selected:
        t = task_map[task_id]
        valid &= task_id not in seen
        seen.add(task_id)
        valid &= nodes[0] == ex._get_node_key(graph, instance.sources[task_id])
        valid &= nodes[-1] == ex._get_node_key(graph, instance.sinks[task_id])
        valid &= len(nodes) == len(set(nodes))
        for u, v in zip(nodes, nodes[1:]):
            data = graph.edges[u, v]
            cost += data['weight']
            if data['arc_type'] in CAPACITY_ARC_TYPES:
                loads[u, v] = loads.get((u, v), 0.0) + resource_coefficient(t, data)
    valid &= seen.isdisjoint(rejected) and len(seen) + len(rejected) == len(instance.tasks)
    excess = max((load - graph.edges[edge]['cap'] for edge, load in loads.items()), default=0.0)
    residual = abs(cost + penalty * len(rejected) - objective)
    return {'feasible': bool(valid and excess <= 1e-6 and residual <= 1e-5),
            'max_capacity_excess': max(0.0, excess), 'objective_residual': residual,
            'completed_path_cost': cost, 'rejection_cost': penalty * len(rejected)}


def run_pb(instance, graph, penalty, label=PB):
    start = time.perf_counter()
    cr = ContinuousRelaxation(graph, instance.tasks, instance.sources, instance.sinks,
        penalty, instance.heuristic, instance.sat_params, k_alt=1)
    cr.build_restricted_masterproblem()
    cert = cr.solve(filter_on=label != NF, dijkstra=label == DJ, max_iterations=200)
    lp_obj = float(cr.rmp.ObjVal)
    frac = len(find_frac_flows(cr))
    ip_start = time.perf_counter()
    nodes = 0
    if frac:
        cr, objective, nodes = solve_last_iter_int(cr)
    else:
        objective = compute_true_objective(cr)
    integer_time = time.perf_counter() - ip_start
    selected = [(flow.trip.task_id, flow.path[1]) for flow, var in cr.x.items() if var.X > 0.5]
    rejected = [task.task_id for task, var in cr.dummy_vars.items() if var.X > 0.5]
    audit = audit_schedule(instance, graph, selected, rejected, penalty, objective)
    if not audit['feasible']:
        raise RuntimeError(f'Infeasible integer recovery: {audit}')
    lower = cr.certified_lower_bound if cert['certified_no_new_columns'] else None
    result = {'label': label, 'obj': float(objective), 'int_obj': float(objective),
        'lp_obj': lp_obj, 'lower_bound': lower,
        'root_lp_gap_pct': 100 * (objective - lower) / lower if lower else None,
        'raw_rmp_gap_pct': 100 * (objective - lp_obj) / lp_obj,
        'root_solve_time': cr.root_solve_time, 'certification': cert,
        'integer_recovery_time': integer_time, 'solve_time': time.perf_counter() - start,
        'completed': len(selected), 'rejected': len(rejected), 'fractional_flows': frac,
        'bb_nodes': nodes, 'num_iters': cr.num_it, 'num_columns': sum(map(len, cr.flows.values())),
        'total_pricing_calls': cr.total_pricing_calls, 'audit': audit,
        'schedule': [{'task': task_id, 'nodes': path} for task_id, path in selected],
        'rejected_tasks': rejected}
    cr.rmp.dispose()
    return result


def run_construction(instance, graph, penalty, seed, restarts=1):
    """Priority Greedy or randomized insertion, with exact base-cost A* routing."""
    start = time.perf_counter()
    rng = random.Random(seed)
    best = None
    sink = ex._get_node_key(graph, next(iter(instance.sinks.values())))
    distances = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), sink, weight='weight')
    for restart in range(restarts):
        remaining = sorted(instance.tasks, key=lambda t: (-t.priority, t.task_id))
        order = []
        while remaining:
            index = rng.randrange(max(1, int(0.3 * len(remaining)))) if restarts > 1 else 0
            order.append(remaining.pop(index))
        loads = {}
        selected, rejected = [], []
        total = 0.0
        for task in order:
            source = ex._get_node_key(graph, instance.sources[task.task_id])
            def weight(u, v, data):
                if restarts > 1 and data['arc_type'] in CAPACITY_ARC_TYPES:
                    if loads.get((u, v), 0.0) + resource_coefficient(task, data) > data['cap'] + 1e-9:
                        return None
                return data['weight']
            try:
                path = nx.astar_path(graph, source, sink, heuristic=lambda u, v: distances.get(u, 0.0), weight=weight)
            except nx.NetworkXNoPath:
                path = None
            if path is not None:
                edges = list(zip(path, path[1:]))
                cost = sum(graph.edges[edge]['weight'] for edge in edges)
                feasible = all(loads.get(edge, 0.0) + resource_coefficient(task, graph.edges[edge]) <= graph.edges[edge]['cap'] + 1e-9
                    for edge in edges if graph.edges[edge]['arc_type'] in CAPACITY_ARC_TYPES)
            if path is None or not feasible or cost >= penalty:
                rejected.append(task.task_id)
                total += penalty
            else:
                selected.append((task.task_id, path))
                total += cost
                for edge in edges:
                    if graph.edges[edge]['arc_type'] in CAPACITY_ARC_TYPES:
                        loads[edge] = loads.get(edge, 0.0) + resource_coefficient(task, graph.edges[edge])
        if best is None or (total, -len(selected)) < (best[0], -len(best[1])):
            best = total, selected, rejected
    total, selected, rejected = best
    audit = audit_schedule(instance, graph, selected, rejected, penalty, total)
    if not audit['feasible']:
        raise RuntimeError(str(audit))
    return {'obj': total, 'completed': len(selected), 'rejected': len(rejected),
        'solve_time': time.perf_counter() - start, 'restarts': restarts, 'audit': audit}


def run_arc_mip(instance, graph, penalty, time_limit=3600):
    """Independent binary arc model; discard only source/sink-unreachable arcs."""
    started = time.perf_counter()
    model = gp.Model('audit_arc_ip')
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0
    model.Params.Threads = 8
    x, y, resource_vars = {}, {}, {}
    reverse = graph.reverse(copy=False)
    sink = ex._get_node_key(graph, next(iter(instance.sinks.values())))
    can_deliver = nx.descendants(reverse, sink) | {sink}
    for task in instance.tasks:
        source = ex._get_node_key(graph, instance.sources[task.task_id])
        reachable = (nx.descendants(graph, source) | {source}) & can_deliver
        y[task] = model.addVar(vtype=gp.GRB.BINARY, obj=penalty)
        incoming, outgoing = {}, {}
        for u in reachable:
            for v, data in graph[u].items():
                if v not in reachable:
                    continue
                var = model.addVar(vtype=gp.GRB.BINARY, obj=data['weight'])
                x[task, u, v] = var
                outgoing.setdefault(u, []).append(var)
                incoming.setdefault(v, []).append(var)
                if data['arc_type'] in CAPACITY_ARC_TYPES:
                    resource_vars.setdefault((u, v), []).append((resource_coefficient(task, data), var))
        for node in reachable | {source, sink}:
            rhs = (1 - y[task]) if node == source else (y[task] - 1 if node == sink else 0)
            model.addConstr(gp.quicksum(outgoing.get(node, [])) - gp.quicksum(incoming.get(node, [])) == rhs)
    for edge, variables in resource_vars.items():
        model.addConstr(gp.LinExpr([c for c, _ in variables], [v for _, v in variables]) <= graph.edges[edge]['cap'])
    model.optimize()
    result = {'status': int(model.Status), 'obj': float(model.ObjVal) if model.SolCount else None,
        'lower_bound': float(model.ObjBound), 'completed': sum(var.X < .5 for var in y.values()) if model.SolCount else None,
        'solve_time': time.perf_counter() - started, 'num_variables': model.NumVars}
    model.dispose()
    return result


def run_case(block, tasks, seed, penalty, fp):
    global _INSTANCE_CACHE
    path = OUT / block / f'tasks_{tasks:04d}_seed_{seed:02d}'
    if block == 'sensitivity':
        path = path.with_name(path.name + f'_m_{int(penalty):04d}')
    path = path.with_suffix('.json')
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get('status') == 'complete' and cached.get('fingerprint') == fp:
            print(f'CACHED {block} tasks={tasks} seed={seed} M={penalty:g}', flush=True)
            return
    started_at = datetime.now().isoformat(timespec='seconds')
    start = time.perf_counter()
    log = path.with_suffix('.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'START {block} tasks={tasks} seed={seed} M={penalty:g}', flush=True)
    rs, cs = (10, 50) if block == 'validation' else (136, 1321)
    with log.open('w', encoding='utf-8', buffering=1) as handle, redirect_stdout(handle):
        cache_key = tasks, seed, rs, cs
        reused = _INSTANCE_CACHE is not None and _INSTANCE_CACHE[0] == cache_key
        if reused:
            _, inst, graph = _INSTANCE_CACHE
            build_time = 0.0
        else:
            _INSTANCE_CACHE = None
            gc.collect()
            random.seed(seed)
            np.random.seed(seed)
            inst = build_audit_instance(seed, rs, cs, tasks)
            graph = build_graph(inst)
            build_time = time.perf_counter() - start
            _INSTANCE_CACHE = cache_key, inst, graph
        if len(inst.tasks) != tasks:
            raise RuntimeError(f'Requested {tasks} tasks, generated {len(inst.tasks)}')
        algorithms = {PB: run_pb(inst, graph, penalty)}
        if block in ('main', 'validation'):
            algorithms['Greedy'] = run_construction(inst, graph, penalty, seed)
        if block == 'validation' or (block == 'main' and tasks in (100, 500, 1000)):
            algorithms['GRASP'] = run_construction(inst, graph, penalty, seed, restarts=20)
        if block == 'ablation':
            algorithms[NF] = run_pb(inst, graph, penalty, NF)
            algorithms[DJ] = run_pb(inst, graph, penalty, DJ)
        if block == 'validation' and tasks == 50:
            algorithms['Full 0-1 ILP'] = run_arc_mip(inst, graph, penalty)
        if block == 'validation':
            algorithms['Branch-and-price'] = solve_branch_and_price(inst, graph, penalty=penalty)
        record = {'status': 'complete', 'block': block, 'tasks': tasks, 'seed': seed,
            'penalty': penalty, 'fingerprint': fp, 'started_at': started_at,
            'finished_at': datetime.now().isoformat(timespec='seconds'), 'rs_count': rs,
            'cs_count': cs, 'graph_nodes': len(graph), 'graph_edges': graph.number_of_edges(),
            'build_time': build_time, 'algorithms': algorithms, 'elapsed': time.perf_counter() - start,
            'graph_reused': reused,
            'environment': {'python': platform.python_version(), 'gurobi': gp.gurobi.version(),
                'platform': platform.platform(), 'processor': platform.processor(),
                'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'unset')},
            'model': 'unit-task, common VGS, equality cover, unperturbed pricing, complete exact pass'}
        record['pricing_implementation'] = 'native CSR A*/Dijkstra, same arc order and weights' if algorithms[PB] else 'python'
        write_json(path, record)
    pb = algorithms[PB]
    print(f'DONE {block} tasks={tasks} seed={seed} obj={pb["obj"]:.0f} completed={pb["completed"]} '
        f'certified={pb["certification"]["certified_no_new_columns"]} time={record["elapsed"]:.1f}s', flush=True)


def parse_range(spec):
    result = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            result.extend(range(lo, hi + 1, 100 if hi > 99 else 1))
        else:
            result.append(int(part))
    return list(dict.fromkeys(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', choices=['main', 'validation', 'ablation', 'sensitivity'], required=True)
    parser.add_argument('--tasks', default='100-1000')
    parser.add_argument('--seeds', default='0-9')
    parser.add_argument('--penalties', default='600')
    args = parser.parse_args()
    fp = fingerprint()
    failed = 0
    for tasks in parse_range(args.tasks):
        for seed in parse_range(args.seeds):
            for penalty in parse_range(args.penalties):
                try:
                    run_case(args.block, tasks, seed, float(penalty), fp)
                except Exception:
                    failed += 1
                    traceback.print_exc()
    sys.exit(bool(failed))


if __name__ == '__main__':
    main()
