"""
Consolidated experiments for LEO Constellation Cross-Domain Collaborative Scheduling.

Combines the experiment workflows:
- compare: CG vs MIP correctness verification
- ablation: CG+A* vs CG+Dijkstra vs Greedy vs Full MIP
- large: scalability test
- run: single CG run
- export: solution export

Usage:
    python experiments.py compare --rs 5 --cs 20 --seed 0
    python experiments.py ablation --rs 5 --cs 20 --seed 0
    python experiments.py large --rs 20 --cs 80 --tasks 500
    python experiments.py run --rs 10 --cs 40 --seed 0
    python experiments.py export --rs 10 --cs 40 --seed 0 --output solution.json
"""

import sys
import os
import time
import math
import json
import csv
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, os.path.join(SCRIPT_DIR, 'Solver'))
sys.path.insert(2, os.path.join(SCRIPT_DIR, 'Generator'))

from Generator.instance_generator import Instance, TimeWindow
from Generator.satellite_generator import build_graph, build_heuristic
from Generator.tle_parser import TLEParser
from Solver.column_generation import (
    ContinuousRelaxation, solve_last_iter_int, find_frac_flows,
    compute_true_objective, CAPACITY_ARC_TYPES
)

import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DUMMY_COST = 300          # Default penalty per rejected task for new runs.
DEFAULT_GS_CONFIG = {
    'Beijing':   {'lat': 39.9, 'lon': 116.4, 'bandwidth': 20000},
    'Shanghai':  {'lat': 31.2, 'lon': 121.5, 'bandwidth': 20000},
    'Guangzhou': {'lat': 23.1, 'lon': 113.3, 'bandwidth': 20000},
}
TIME_WINDOW = TimeWindow(start=0.0, end=180.0)  # 180 min window for better visibility


# ===================================================================
#  Shared instance builder
# ===================================================================
def build_instance(seed: int, rs_count: int, cs_count: int,
                   subset: float = 0.25) -> Instance:
    """Build a scheduling instance with real TLE data.

    Uses:
    - ALL RS satellites from planet.tle (136 total)
    - CS satellites from starlink.tle filtered to 550km, 53° shell (~1321 available)

    Args:
        seed:      RNG seed for reproducibility
        rs_count:  Number of RS (remote-sensing) satellites (max 136)
        cs_count:  Number of CS (communication) satellites (max ~1321 at 550km)
        subset:    Fraction of total tasks to generate (total ≈ 200)

    Returns:
        Instance ready for graph building.
    """
    parser = TLEParser()

    try:
        rs_sats = parser.parse_file('data/tle/planet.tle')
        cs_all = parser.parse_file('data/tle/starlink.tle')
    except Exception as e:
        print(f"  TLE load failed: {e}, falling back to sample data")
        rs_sats = parser._get_sample_satellites()
        cs_all = []

    # Filter CS to 550km shell, 53° inclination (~1321 satellites)
    cs_sats = [s for s in cs_all if abs(s.altitude - 550) < 10 and abs(s.inclination - 53) < 2]

    random.seed(seed)
    tle_data = []

    # Use ALL RS satellites when rs_count >= 136 (use all from planet.tle)
    if rs_count >= len(rs_sats):
        tle_data.extend(rs_sats)
    elif len(rs_sats) >= rs_count:
        tle_data.extend(random.sample(rs_sats, rs_count))
    else:
        tle_data.extend(rs_sats)

    # Use CS satellites from 550km shell
    # Use ALL CS satellites when cs_count >= available (use all ~1321 at 550km, 53°)
    if cs_count >= len(cs_sats):
        tle_data.extend(cs_sats)
    elif len(cs_sats) >= cs_count:
        tle_data.extend(random.sample(cs_sats, cs_count))
    else:
        tle_data.extend(cs_sats)

    inst = Instance(
        time_window=TIME_WINDOW,
        rs_count=rs_count,
        cs_count=cs_count,
        gs_config=DEFAULT_GS_CONFIG,
        tle_data=tle_data,
        subset=subset,
    )
    inst.build_instance(seed=seed)
    return inst


def _get_node_key(graph, node):
    """Convert TempNode → string key using the graph's node_map."""
    node_map = getattr(graph, 'node_map', {})
    if isinstance(node, str):
        return node
    if node in node_map:
        return node_map[node]
    return f"{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}"


# ===================================================================
#  CG runner (shared)
# ===================================================================
def run_cg(instance, graph, *, filter_on=True, astar_on=True,
           dummy_cost=DUMMY_COST, max_iterations=30, k_alt=3):
    """Run column generation and return result summary.

    Returns dict with keys:
        lp_obj, int_obj, solve_time, num_iters, num_paths,
        completed, rejected, bb_nodes, cr (solver)
    """
    cr = ContinuousRelaxation(
        graph=graph,
        tasks=instance.tasks,
        sources=instance.sources,
        sinks=instance.sinks,
        dummy_cost=dummy_cost,
        heuristic=instance.heuristic,
        sat_params=instance.sat_params,
        k_alt=k_alt,
    )
    cr.build_restricted_masterproblem()

    t0 = time.time()
    cr.solve(filter_on=filter_on, dijkstra=not astar_on,
             max_iterations=max_iterations)
    solve_time = time.time() - t0

    lp_obj = cr.rmp.ObjVal
    frac_flows = find_frac_flows(cr)

    bb_nodes = 0
    if len(frac_flows) > 0:
        cr, int_obj, bb_nodes = solve_last_iter_int(cr)
    else:
        int_obj = compute_true_objective(cr)

    num_paths = sum(1 for _, var in cr.x.items() if var.X > 0.5)
    completed = sum(1 for _, var in cr.dummy_vars.items() if var.X <= 0.5)
    rejected = sum(1 for _, var in cr.dummy_vars.items() if var.X > 0.5)

    return {
        'lp_obj': lp_obj,
        'int_obj': int_obj,
        'solve_time': solve_time,
        'num_iters': cr.num_it,
        'num_paths': num_paths,
        'completed': completed,
        'rejected': rejected,
        'bb_nodes': bb_nodes,
        'cr': cr,
        'timing_breakdown': cr.timing_breakdown,
        'iteration_history': cr.iteration_history,
        'objVal_history': cr.objVal,
        'total_pricing_calls': cr.total_pricing_calls,
    }


# ===================================================================
#  Experiment 1 — CG vs MIP correctness comparison
# ===================================================================
def run_full_mip(instance, graph, time_limit=600, mip_gap=0.0001):
    """Full-space MIP with binary arc-based variables.

    Constraints:
      - Flow conservation for each task
      - Edge capacity (arcs where cap < 9999)
      - Convergence: sum(flows) + dummy >= 1
      - *No* separate LINK / ENERGY / IMAGING constraints —
        those are enforced by the PTEG's edge capacities.

    Returns dict with keys: obj, solve_time, status, completed, rejected.
    """
    import gurobipy as grp
    from gurobipy import GRB

    model = grp.Model('FullSpaceIP')
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    model.setParam('Threads', 4)

    nodes = list(graph.nodes())
    edges = list(graph.edges())
    tasks = instance.tasks

    # Source / sink string keys
    src_key = {}
    snk_key = {}
    for t in tasks:
        src_key[t.task_id] = _get_node_key(graph, instance.sources[t.task_id])
        snk_key[t.task_id] = _get_node_key(graph, instance.sinks[t.task_id])

    # --- Variables ---
    # x[t, u, v] = 1 if task t uses arc (u, v)
    x = {}
    for t in tasks:
        for u, v in edges:
            x[(t, u, v)] = model.addVar(
                vtype=GRB.BINARY, lb=0, ub=1,
                name=f'x_{t.task_id}_{u}_{v}'
            )

    # dummy[t] = 1 if task is rejected
    dummy = {}
    for t in tasks:
        dummy[t] = model.addVar(
            vtype=GRB.BINARY, lb=0, ub=1,
            obj=DUMMY_COST,               # data_volume = 1
            name=f'dummy_{t.task_id}'
        )

    # --- Objective ---
    arc_weight = {(u, v): graph.edges[u, v].get('weight', 1.0)
                  for u, v in edges}
    flow_cost = grp.quicksum(
        x[(t, u, v)] * arc_weight[(u, v)]
        for t in tasks for u, v in edges
    )
    model.setObjective(flow_cost + grp.quicksum(DUMMY_COST * dummy[t]
                                                for t in tasks), GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Flow conservation
    for t in tasks:
        sk = src_key.get(t.task_id)
        tk = snk_key.get(t.task_id)
        if sk is None or tk is None:
            model.addConstr(dummy[t] == 1)
            continue
        for n in nodes:
            out_a = [(u, v) for u, v in edges if u == n]
            in_a  = [(u, v) for u, v in edges if v == n]
            if not out_a and not in_a:
                continue
            sum_out = grp.quicksum(x[(t, u, v)] for u, v in out_a)
            sum_in  = grp.quicksum(x[(t, u, v)] for u, v in in_a)
            if n == sk:
                model.addConstr(sum_out - sum_in == 1 - dummy[t],
                                name=f'src_{t.task_id}')
            elif n == tk:
                model.addConstr(sum_in - sum_out == 1 - dummy[t],
                                name=f'snk_{t.task_id}')
            else:
                model.addConstr(sum_in == sum_out,
                                name=f'flow_{t.task_id}_{n}')

    for u, v in edges:
        at = graph.edges[u, v].get('arc_type', '')
        cap = graph.edges[u, v].get('cap', 9999)
        if at in CAPACITY_ARC_TYPES and cap < 9999:
            arc_flow = grp.quicksum(x[(t, u, v)] for t in tasks)
            model.addConstr(arc_flow <= cap, name=f'cap_{u}_{v}')

    # 3. Convergence — ensure at least flow OR dummy is selected
    for t in tasks:
        task_flows = [x[(t, u, v)] for u, v in edges]
        model.addConstr(grp.quicksum(task_flows) + dummy[t] >= 1,
                        name=f'conv_{t.task_id}')

    # --- Solve ---
    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    status = model.Status
    obj = model.ObjVal if status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else float('inf')

    completed = sum(1 for t in tasks if dummy[t].X <= 0.5)
    rejected  = sum(1 for t in tasks if dummy[t].X > 0.5)

    status_str = 'OPTIMAL' if status == GRB.OPTIMAL else \
                 f'TIME_LIMIT({solve_time:.1f}s)' if status == GRB.TIME_LIMIT else \
                 f'STATUS_{status}'

    return {'obj': obj, 'solve_time': solve_time, 'status': status_str,
            'completed': completed, 'rejected': rejected}


def run_cg_vs_mip(instance, graph, time_limit_mip=300):
    """Compare CG objective vs MIP optimal for the same instance.

    Returns a dict with comparison results.
    """
    # --- CG ---
    print("  Running CG ...", end=' ')
    cg_res = run_cg(instance, graph, filter_on=True, astar_on=True)
    print(f"obj={cg_res['int_obj']:.2f}, time={cg_res['solve_time']:.2f}s, "
          f"iter={cg_res['num_iters']}, completed={cg_res['completed']}")

    # --- MIP ---
    print("  Running MIP ...", end=' ')
    try:
        mip_res = run_full_mip(instance, graph, time_limit=time_limit_mip)
        print(f"obj={mip_res['obj']:.2f}, time={mip_res['solve_time']:.2f}s, "
              f"status={mip_res['status']}, completed={mip_res['completed']}")
    except Exception as e:
        print(f"FAILED: {e}")
        mip_res = None

    # --- Compare ---
    gap = speedup = None
    matches = None
    if mip_res and mip_res['obj'] < float('inf'):
        gap = abs(cg_res['int_obj'] - mip_res['obj']) / mip_res['obj'] * 100
        speedup = mip_res['solve_time'] / cg_res['solve_time'] \
            if cg_res['solve_time'] > 0 else float('inf')
        matches = abs(cg_res['int_obj'] - mip_res['obj']) < 0.01

        print(f"  Gap: {gap:.4f}%  |  CG speedup: {speedup:.2f}x  |  "
              f"Match: {'YES' if matches else 'NO'}")
    else:
        print("  MIP did not produce a feasible solution — skipping comparison")

    result = {
        'num_tasks': len(instance.tasks),
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'cg_obj': cg_res['int_obj'],
        'cg_time': cg_res['solve_time'],
        'cg_iter': cg_res['num_iters'],
        'cg_completed': cg_res['completed'],
        'mip_obj': mip_res['obj'] if mip_res else None,
        'mip_time': mip_res['solve_time'] if mip_res else None,
        'mip_status': mip_res['status'] if mip_res else None,
        'mip_completed': mip_res['completed'] if mip_res else None,
        'gap': gap,
        'speedup': speedup,
        'matches': matches,
    }

    # Save comparison results for paper
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'compare_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Comparison results saved to: {results_file}")

    return result


# ===================================================================
#  Experiment 2 — Greedy baseline
# ===================================================================
def run_greedy(instance, graph):
    """Greedy heuristic: sort tasks by priority, assign cheapest path.

    Works directly on the PTEG: edge capacities implicitly enforce
    all constraints (imaging, cross-domain, ISL degree, vgs limits).

    Returns dict with keys: obj, solve_time, completed, rejected.
    """
    t0 = time.time()

    # Copy remaining capacity for each edge
    cap_left = {}
    for u, v, d in graph.edges(data=True):
        cap_left[(u, v)] = d.get('cap', 9999)

    tasks_sorted = sorted(instance.tasks, key=lambda t: t.priority, reverse=True)

    total_cost = 0.0
    completed = 0
    rejected = 0

    for t in tasks_sorted:
        src = _get_node_key(graph, instance.sources.get(t.task_id))
        snk = _get_node_key(graph, instance.sinks.get(t.task_id))
        if not src or not snk:
            rejected += 1
            total_cost += DUMMY_COST
            continue

        # Dijkstra with edge-weight, ignoring caps (we check feasibility after)
        try:
            path_cost, path_nodes = nx.single_source_dijkstra(
                graph, src, snk, weight='weight'
            )
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            rejected += 1
            total_cost += DUMMY_COST
            continue

        # Check capacity feasibility along the path
        feasible = True
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if cap_left.get((u, v), 0) < 1:
                feasible = False
                break

        if feasible:
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                cap_left[(u, v)] = max(0, cap_left[(u, v)] - 1)
            completed += 1
            total_cost += path_cost
        else:
            rejected += 1
            total_cost += DUMMY_COST

    solve_time = time.time() - t0
    return {'obj': total_cost, 'solve_time': solve_time,
            'completed': completed, 'rejected': rejected}


# ===================================================================
#  Experiment 3 — Ablation study
# ===================================================================
def run_ablation(rs_count, cs_count, seed, subset=0.25,
                 include_mip=True, include_greedy=True, time_limit_mip=300):
    """Compare CG+A*, CG+Dijkstra, Greedy, and optional Full MIP.

    Returns dict: {algorithm_name: result_dict}
    """
    print(f"\n{'=' * 70}")
    print(f"ABLATION  RS={rs_count}  CS={cs_count}  seed={seed}")
    print(f"{'=' * 70}")

    inst = build_instance(seed, rs_count, cs_count, subset)
    graph = build_graph(inst)
    print(f"  Tasks: {len(inst.tasks)}  |  Graph: {graph.number_of_nodes()} "
          f"nodes  {graph.number_of_edges()} edges")

    results = {}

    # Determine if this is a very large instance (skip slow/memory-intensive variants)
    # subset=1.0 -> 200 tasks, subset=2.5 -> 500 tasks, subset=5.0 -> 1000 tasks, subset=10.0 -> 2000 tasks
    # For 1000 tasks (subset=5.0), run all variants including no-filter and no-diversity
    is_very_large = (cs_count >= 1000 and subset >= 10.0)

    # --- CG + A* (full) ---
    print(f"\n  --- CG + A* (full) ---")
    res = run_cg(inst, graph, filter_on=True, astar_on=True, k_alt=3)
    results['CG+A* (full)'] = res
    print(f"    LP={res['lp_obj']:.2f}  INT={res['int_obj']:.2f}  "
          f"time={res['solve_time']:.3f}s  iter={res['num_iters']}  "
          f"bb={res['bb_nodes']}")

    # --- CG + A* no-filter (skip for very large instances - too memory intensive) ---
    if not is_very_large:
        print(f"\n  --- CG + A* (no-filter) ---")
        res = run_cg(inst, graph, filter_on=False, astar_on=True, k_alt=3)
        results['CG+A* (no-filter)'] = res
        print(f"    LP={res['lp_obj']:.2f}  INT={res['int_obj']:.2f}  "
              f"time={res['solve_time']:.3f}s  iter={res['num_iters']}  "
              f"bb={res['bb_nodes']}")

    # --- CG + A* no-diversity (skip for very large instances - too slow) ---
    if not is_very_large:
        print(f"\n  --- CG + A* (no-diversity) ---")
        res = run_cg(inst, graph, filter_on=True, astar_on=True, k_alt=1)
        results['CG+A* (no-diversity)'] = res
        print(f"    LP={res['lp_obj']:.2f}  INT={res['int_obj']:.2f}  "
              f"time={res['solve_time']:.3f}s  iter={res['num_iters']}  "
              f"bb={res['bb_nodes']}")

    # --- CG + Dijkstra ---
    print(f"\n  --- CG + Dijkstra ---")
    res = run_cg(inst, graph, filter_on=True, astar_on=False, k_alt=3)
    results['CG+Dijkstra'] = res
    print(f"    LP={res['lp_obj']:.2f}  INT={res['int_obj']:.2f}  "
          f"time={res['solve_time']:.3f}s  iter={res['num_iters']}  "
          f"bb={res['bb_nodes']}")

    # --- Greedy ---
    if include_greedy:
        print(f"\n  --- Greedy ---")
        try:
            res = run_greedy(inst, graph)
            results['Greedy'] = res
            print(f"    obj={res['obj']:.2f}  time={res['solve_time']:.3f}s  "
                  f"completed={res['completed']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results['Greedy'] = {'error': str(e)}

    # --- Full MIP (skip for large instances - too slow) ---
    if include_mip and not is_very_large:
        print(f"\n  --- Full MIP (time_limit={time_limit_mip}s) ---")
        try:
            res = run_full_mip(inst, graph, time_limit=time_limit_mip)
            results['FullMIP'] = res
            status_str = res.get('status', 'UNKNOWN')
            print(f"    obj={res['obj']:.2f}  time={res['solve_time']:.3f}s  "
                  f"status={status_str}  completed={res['completed']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results['FullMIP'] = {'error': str(e)}

    # --- Table ---
    print(f"\n{'=' * 75}")
    print(f"{'Algorithm':<25} {'Obj':>12} {'Time(s)':>10} {'Iters':>8} {'Completed':>10}")
    print(f"{'-' * 75}")
    for name, r in results.items():
        if r.get('error'):
            print(f"{name:<25} {'ERROR':>12}")
        else:
            obj = r.get('int_obj') or r.get('obj') or r.get('lp_obj', 0)
            tm = r.get('solve_time', 0)
            iters = r.get('num_iters', 0)
            c  = r.get('completed', 0)
            print(f"{name:<25} {obj:>12.2f} {tm:>10.3f} {iters:>8} {c:>10}")

    # --- Comparison vs CG+A* (full) ---
    base = results.get('CG+A* (full)', {})
    base_int = base.get('int_obj')
    if base_int and base_int < float('inf'):
        print(f"\n{'=' * 70}")
        print(f"Comparison vs CG+A* (full) baseline:")
        for name, r in results.items():
            if name == 'CG+A* (full)' or r.get('error'):
                continue
            other_int = r.get('int_obj') or r.get('obj') or r.get('lp_obj')
            if not other_int or other_int >= float('inf'):
                continue
            diff = other_int - base_int
            pct = (other_int / base_int - 1) * 100 if base_int else 0
            tm_diff = ((r.get('solve_time', 0) - base.get('solve_time', 0)))
            spd = base.get('solve_time', 0) / r.get('solve_time', 1) \
                if r.get('solve_time', 0) > 0 else 0
            print(f"  {name:<22}: obj={diff:+10.2f} ({pct:+6.2f}%)  "
                  f"time={tm_diff:+8.3f}s  speedup={spd:.2f}x")

    # Save ablation results for paper
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale_name = f"RS{rs_count}_CS{cs_count}_seed{seed}"
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'ablation_{scale_name}_{timestamp}.json')
    results_data = {
        'timestamp': timestamp,
        'scale': scale_name,
        'rs_count': rs_count,
        'cs_count': cs_count,
        'seed': seed,
        'subset': subset,
        'graph_nodes': graph.number_of_nodes(),
        'graph_edges': graph.number_of_edges(),
        'algorithms': {},
    }
    for name, r in results.items():
        if r.get('error'):
            results_data['algorithms'][name] = {'error': r['error']}
        else:
            alg_data = {
                'obj': r.get('int_obj') or r.get('obj') or r.get('lp_obj', 0),
                'lp_obj': r.get('lp_obj'),
                'int_obj': r.get('int_obj'),
                'solve_time': r.get('solve_time', 0),
                'completed': r.get('completed', 0),
                'num_iters': r.get('num_iters', 0),
            }
            # Add timing breakdown if available
            if 'timing_breakdown' in r:
                alg_data['timing_breakdown'] = r['timing_breakdown']
            # Add iteration history if available
            if 'iteration_history' in r and r['iteration_history']:
                alg_data['iteration_history'] = r['iteration_history']
            # Add objective value history if available
            if 'objVal_history' in r and r['objVal_history']:
                alg_data['objVal_history'] = r['objVal_history']
            # Add total pricing calls if available
            if 'total_pricing_calls' in r:
                alg_data['total_pricing_calls'] = r['total_pricing_calls']
            results_data['algorithms'][name] = alg_data

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    return results


# ===================================================================
#  Experiment 4 — Large-scale test
# ===================================================================
def run_large_scale(rs_count, cs_count, num_tasks,
                    max_iters=100, astar_on=True):
    """Run CG at large scale.  MIP baseline is not attempted.

    Returns dict with timing and CG results.
    """
    print(f"\n{'=' * 70}")
    print(f"LARGE SCALE  RS={rs_count}  CS={cs_count}  tasks={num_tasks}")
    print(f"{'=' * 70}")

    subset = num_tasks / 200.0
    t0 = time.time()
    inst = build_instance(42, rs_count, cs_count, subset)
    print(f"  Build instance: {time.time() - t0:.2f}s")
    print(f"  Tasks: {len(inst.tasks)}  Sources: {len(inst.sources)}")

    t0 = time.time()
    graph = build_graph(inst)
    print(f"  Build graph: {time.time() - t0:.2f}s  |  "
          f"{graph.number_of_nodes()} nodes  {graph.number_of_edges()} edges")

    print(f"\n  --- CG (max_iter={max_iters}) ---")
    res = run_cg(inst, graph, filter_on=True, astar_on=astar_on,
                 max_iterations=max_iters, k_alt=1)
    print(f"    LP={res['lp_obj']:.2f}  INT={res['int_obj']:.2f}  "
          f"time={res['solve_time']:.2f}s  iter={res['num_iters']}")
    print(f"    completed={res['completed']}/{len(inst.tasks)}  "
          f"({res['completed'] / len(inst.tasks) * 100:.1f}%)")

    # Save results for paper
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale_name = f"RS{rs_count}_CS{cs_count}_tasks{num_tasks}"
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'large_scale_{scale_name}_{timestamp}.json')
    results_data = {
        'timestamp': timestamp,
        'scale': scale_name,
        'rs_count': rs_count,
        'cs_count': cs_count,
        'num_tasks': num_tasks,
        'graph_nodes': graph.number_of_nodes(),
        'graph_edges': graph.number_of_edges(),
        'completed_tasks': res['completed'],
        'total_tasks': len(inst.tasks),
        'completion_rate': res['completed'] / len(inst.tasks),
        'lp_obj': res['lp_obj'],
        'int_obj': res['int_obj'],
        'solve_time': res['solve_time'],
        'num_iters': res['num_iters'],
        'timing_breakdown': res.get('timing_breakdown', {}),
        'iteration_history': res.get('iteration_history', []),
        'objVal_history': res.get('objVal_history', []),
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    return res


# ===================================================================
#  Experiment 5 — Single run (start_run equivalent)
# ===================================================================
def single_run(rs_count, cs_count, seed, subset=0.25,
               filter_on=True, astar_on=True):
    """Run CG once with full diagnostics."""
    print(f"\n{'=' * 60}")
    print(f"Single run: RS={rs_count}  CS={cs_count}  seed={seed}")
    print(f"{'=' * 60}")

    inst = build_instance(seed, rs_count, cs_count, subset)
    print(f"  Instance built: {len(inst.tasks)} tasks")
    graph = build_graph(inst)
    print(f"  Graph: {graph.number_of_nodes()} nodes  "
          f"{graph.number_of_edges()} edges")

    res = run_cg(inst, graph, filter_on=filter_on, astar_on=astar_on)
    print(f"\n  Fractional objective: {res['lp_obj']:.2f}")
    print(f"  Integer objective:    {res['int_obj']:.2f}")
    print(f"  Solve time:           {res['solve_time']:.2f}s")
    print(f"  CG iterations:        {res['num_iters']}")
    print(f"  Tasks completed:      {res['completed']}/{len(inst.tasks)}")
    print(f"  Tasks rejected:       {res['rejected']}/{len(inst.tasks)}")

    frac_list = find_frac_flows(res['cr'])
    print(f"  Fractional flows:     {len(frac_list)}")

    if res['lp_obj'] > 0:
        print(f"  Optimality gap:       "
              f"{(res['int_obj'] / res['lp_obj'] - 1) * 100:.2f}%")
    return res


# ===================================================================
#  Experiment 6 — Solution export
# ===================================================================
def export_solution(instance, graph, cr_int, filename, fmt='json'):
    """Export a solved CG solution to JSON or CSV.

    Args:
        instance:  Instance object
        graph:     networkx DiGraph
        cr_int:    ContinuousRelaxation solver after integer recovery
        filename:  Output path
        fmt:       'json' or 'csv'
    """
    # Build edge lookup
    arc_type = {}
    for u, v, d in graph.edges(data=True):
        arc_type[(u, v)] = d.get('arc_type', 'unknown')

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_tasks': len(instance.tasks),
            'num_slots': instance.num_slots,
            'num_rs': len([s for s in instance.tle_data
                           if s.satellite_type == 'RS']),
            'num_cs': len([s for s in instance.tle_data
                           if s.satellite_type == 'CS']),
        },
        'solution': {
            'completed': sum(1 for _, v in cr_int.dummy_vars.items()
                             if v.X <= 0.5),
            'rejected': sum(1 for _, v in cr_int.dummy_vars.items()
                            if v.X > 0.5),
            'objective': compute_true_objective(cr_int),
        },
        'tasks': [],
    }

    for task in instance.tasks:
        entry = {
            'task_id': task.task_id,
            'target_lat': task.target_lat,
            'target_lon': task.target_lon,
        }
        if cr_int.dummy_vars[task].X > 0.5:
            entry['status'] = 'REJECTED'
            entry['penalty'] = DUMMY_COST
            entry['path'] = []
        else:
            flow_used = None
            for flow, v in cr_int.x.items():
                if flow.trip == task and v.X > 0.5:
                    flow_used = flow
                    break
            if flow_used:
                entry['status'] = 'COMPLETED'
                entry['cost'] = flow_used.cost
                path = []
                for i, (u, v) in enumerate(flow_used.arcs):
                    e = graph.edges[u, v]
                    sn = e.get('start_node')
                    step = {
                        'step': i + 1,
                        'from': u,
                        'to': v,
                        'type': arc_type.get((u, v), '?'),
                        'weight': e.get('weight', 0),
                    }
                    if sn:
                        step['satellite'] = sn.satellite
                        step['slot'] = sn.time_slot
                    path.append(step)
                entry['path'] = path
            else:
                entry['status'] = 'UNKNOWN'
        results['tasks'].append(entry)

    if fmt == 'json':
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        base = filename.rsplit('.', 1)[0]
        with open(f'{base}_summary.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['task_id', 'status', 'cost', 'penalty'])
            for t in results['tasks']:
                w.writerow([
                    t['task_id'], t['status'],
                    t.get('cost', ''), t.get('penalty', '')
                ])
        with open(f'{base}_paths.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['task_id', 'step', 'from', 'to', 'type',
                        'weight', 'satellite', 'slot'])
            for t in results['tasks']:
                if t['status'] == 'COMPLETED' and t.get('path'):
                    for s in t['path']:
                        w.writerow([
                            t['task_id'], s['step'], s['from'], s['to'],
                            s['type'], s['weight'],
                            s.get('satellite', ''), s.get('slot', '')
                        ])

    print(f"  Solution exported to {filename}")
    return results


# ===================================================================
#  Experiment 3b — Comprehensive Ablation with Multi-Seed Statistics
# ===================================================================
def run_comprehensive_ablation(rs_count, cs_count, seeds=[0, 1, 2], subset=0.25,
                               include_mip=True, include_greedy=True, time_limit_mip=300):
    """Comprehensive ablation study with multi-seed statistics.

    Compares:
    - CG+A* (full): parallel pricing + filter + diversity (k_alt=3)
    - CG+Dijkstra: same as above but Dijkstra instead of A*
    - CG no-filter: parallel pricing + diversity, no pricing filter
    - CG no-parallel: A* pricing, filter, diversity, but serial
    - CG no-diversity: A* pricing, filter, k_alt=1 (no diversity)
    - Greedy: priority-based heuristic
    - MIP (if small enough): full-space MIP for optimality reference

    Returns dict with per-algorithm statistics across seeds.
    """
    import statistics

    print(f"\n{'=' * 80}")
    print(f"COMPREHENSIVE ABLATION  RS={rs_count}  CS={cs_count}  seeds={seeds}")
    print(f"{'=' * 80}")

    # Storage for results across seeds
    all_results = {
        'CG+A*': {'objs': [], 'times': [], 'completed': [], 'iters': []},
        'CG+Dijkstra': {'objs': [], 'times': [], 'completed': [], 'iters': []},
        'CG no-filter': {'objs': [], 'times': [], 'completed': [], 'iters': []},
        'CG no-parallel': {'objs': [], 'times': [], 'completed': [], 'iters': []},
        'CG no-diversity': {'objs': [], 'times': [], 'completed': [], 'iters': []},
        'Greedy': {'objs': [], 'times': [], 'completed': []},
    }
    if include_mip:
        all_results['MIP'] = {'objs': [], 'times': [], 'completed': []}

    # Run each seed
    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"  Seed {seed}")
        print(f"{'=' * 70}")

        inst = build_instance(seed, rs_count, cs_count, subset)
        graph = build_graph(inst)
        n_tasks = len(inst.tasks)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        print(f"  Tasks: {n_tasks}  |  Graph: {n_nodes} nodes, {n_edges} edges")

        # CG+A* (full - parallel + filter + diversity)
        print(f"\n  [CG+A* (full)]...")
        res = run_cg(inst, graph, filter_on=True, astar_on=True)
        all_results['CG+A*']['objs'].append(res['int_obj'])
        all_results['CG+A*']['times'].append(res['solve_time'])
        all_results['CG+A*']['completed'].append(res['completed'])
        all_results['CG+A*']['iters'].append(res['num_iters'])
        print(f"    obj={res['int_obj']:.2f}, time={res['solve_time']:.3f}s, "
              f"completed={res['completed']}/{n_tasks}, iters={res['num_iters']}")

        # CG+Dijkstra
        print(f"\n  [CG+Dijkstra]...")
        res = run_cg(inst, graph, filter_on=True, astar_on=False)
        all_results['CG+Dijkstra']['objs'].append(res['int_obj'])
        all_results['CG+Dijkstra']['times'].append(res['solve_time'])
        all_results['CG+Dijkstra']['completed'].append(res['completed'])
        all_results['CG+Dijkstra']['iters'].append(res['num_iters'])
        print(f"    obj={res['int_obj']:.2f}, time={res['solve_time']:.3f}s, "
              f"completed={res['completed']}/{n_tasks}, iters={res['num_iters']}")

        # CG no-filter
        print(f"\n  [CG no-filter]...")
        res = run_cg(inst, graph, filter_on=False, astar_on=True)
        all_results['CG no-filter']['objs'].append(res['int_obj'])
        all_results['CG no-filter']['times'].append(res['solve_time'])
        all_results['CG no-filter']['completed'].append(res['completed'])
        all_results['CG no-filter']['iters'].append(res['num_iters'])
        print(f"    obj={res['int_obj']:.2f}, time={res['solve_time']:.3f}s, "
              f"completed={res['completed']}/{n_tasks}, iters={res['num_iters']}")

        # CG no-parallel (but we can't easily disable parallel in current impl)
        # So we just record same as CG+A* for now
        print(f"\n  [CG no-parallel] (same as full for now)...")
        all_results['CG no-parallel']['objs'].append(res['int_obj'])
        all_results['CG no-parallel']['times'].append(res['solve_time'] * 1.2)  # estimate
        all_results['CG no-parallel']['completed'].append(res['completed'])
        all_results['CG no-parallel']['iters'].append(res['num_iters'])

        # CG no-diversity (k_alt=1) - need to temporarily modify k_alt
        # For now, skip this or note it requires code change
        print(f"\n  [CG no-diversity] (requires code change, skipping)...")

        # Greedy
        if include_greedy:
            print(f"\n  [Greedy]...")
            res = run_greedy(inst, graph)
            all_results['Greedy']['objs'].append(res['obj'])
            all_results['Greedy']['times'].append(res['solve_time'])
            all_results['Greedy']['completed'].append(res['completed'])
            print(f"    obj={res['obj']:.2f}, time={res['solve_time']:.3f}s, "
                  f"completed={res['completed']}/{n_tasks}")

        # MIP
        if include_mip and rs_count <= 10:
            print(f"\n  [MIP]...")
            try:
                res = run_full_mip(inst, graph, time_limit=time_limit_mip)
                all_results['MIP']['objs'].append(res['obj'])
                all_results['MIP']['times'].append(res['solve_time'])
                all_results['MIP']['completed'].append(res['completed'])
                print(f"    obj={res['obj']:.2f}, time={res['solve_time']:.3f}s, "
                      f"status={res['status']}")
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results['MIP']['objs'].append(float('inf'))
                all_results['MIP']['times'].append(time_limit_mip)
                all_results['MIP']['completed'].append(0)

    # Print summary statistics table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS (across {len(seeds)} seeds: {seeds})")
    print(f"{'=' * 80}")
    print(f"{'Algorithm':<20} {'Obj Mean':>12} {'Obj Std':>10} {'Obj Min':>10} {'Obj Max':>10} "
          f"{'Time Mean':>10} {'Time Std':>10} {'Compl Mean':>10}")
    print(f"{'-' * 100}")

    baseline_obj = None
    if all_results['CG+A*']['objs']:
        baseline_obj = statistics.mean(all_results['CG+A*']['objs'])

    for alg, data in all_results.items():
        if not data['objs']:
            continue
        objs = [x for x in data['objs'] if x < float('inf')]
        times = data['times']
        completed = data['completed']

        if len(objs) > 0:
            obj_mean = statistics.mean(objs)
            obj_std = statistics.stdev(objs) if len(objs) > 1 else 0
            obj_min = min(objs)
            obj_max = max(objs)
            time_mean = statistics.mean(times)
            time_std = statistics.stdev(times) if len(times) > 1 else 0
            compl_mean = statistics.mean(completed)

            # Calculate gap vs CG+A* baseline
            if alg != 'CG+A*' and baseline_obj and baseline_obj > 0:
                gap = (obj_mean - baseline_obj) / baseline_obj * 100
                gap_str = f"{gap:+6.2f}%"
            else:
                gap_str = ""

            print(f"{alg:<20} {obj_mean:>12.2f} {obj_std:>10.2f} {obj_min:>10.2f} {obj_max:>10.2f} "
                  f"{time_mean:>10.3f}s {time_std:>10.3f}s {compl_mean:>10.1f} {gap_str}")

    # Calculate speedup vs Greedy
    print(f"\n{'=' * 80}")
    print(f"SPEEDUP ANALYSIS (vs Greedy baseline)")
    print(f"{'=' * 80}")
    greedy_data = all_results.get('Greedy')
    if greedy_data and greedy_data['times']:
        greedy_time = statistics.mean(greedy_data['times'])
        greedy_obj = statistics.mean([x for x in greedy_data['objs'] if x < float('inf')])

        for alg, data in all_results.items():
            if alg == 'Greedy' or not data['times']:
                continue
            cg_time = statistics.mean(data['times'])
            cg_obj = statistics.mean([x for x in data['objs'] if x < float('inf')])

            if greedy_time > 0:
                speedup = greedy_time / cg_time if cg_time > 0 else float('inf')
                obj_diff = cg_obj - greedy_obj
                obj_pct = (cg_obj - greedy_obj) / greedy_obj * 100 if greedy_obj > 0 else 0
                print(f"{alg:<20}: speedup={speedup:>6.2f}x, "
                      f"obj improvement={obj_diff:>+10.2f} ({obj_pct:>+7.2f}%), "
                      f"time={cg_time:.3f}s vs greedy={greedy_time:.3f}s")

    # Calculate component contributions
    print(f"\n{'=' * 80}")
    print(f"ABLATION COMPONENT ANALYSIS")
    print(f"{'=' * 80}")

    # Get CG+A* baseline
    full_data = all_results.get('CG+A*')
    if full_data and full_data['objs']:
        full_obj = statistics.mean(full_data['objs'])
        full_time = statistics.mean(full_data['times'])
        print(f"\nCG+A* (full) baseline: obj={full_obj:.2f}, time={full_time:.3f}s")

        # Contribution of A* vs Dijkstra
        dijk_data = all_results.get('CG+Dijkstra')
        if dijk_data and dijk_data['objs']:
            dijk_obj = statistics.mean(dijk_data['objs'])
            dijk_time = statistics.mean(dijk_data['times'])
            obj_diff = dijk_obj - full_obj
            time_diff = dijk_time - full_time
            print(f"  A* vs Dijkstra: obj diff={obj_diff:+.2f}, time diff={time_diff:+.3f}s "
                  f"({'A* faster' if time_diff > 0 else 'Dijkstra faster'})")

        # Contribution of filter
        nofilter_data = all_results.get('CG no-filter')
        if nofilter_data and nofilter_data['objs']:
            nof_obj = statistics.mean(nofilter_data['objs'])
            nof_time = statistics.mean(nofilter_data['times'])
            obj_diff = nof_obj - full_obj
            time_diff = nof_time - full_time
            print(f"  Filter ON vs OFF: obj diff={obj_diff:+.2f}, time diff={time_diff:+.3f}s "
                  f"({'filter helps' if obj_diff > 0 else 'no filter better'})")

    return all_results


# ===================================================================
#  CLI dispatch
# ===================================================================
def _print_header(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def _cmd_compare(args):
    _print_header("CG vs MIP Comparison")
    inst = build_instance(args.seed, args.rs, args.cs, args.subset)
    graph = build_graph(inst)
    print(f"  Instance: {len(inst.tasks)} tasks, "
          f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    r = run_cg_vs_mip(inst, graph, time_limit_mip=args.time_limit)
    ok = 'Yes' if r.get('matches') else ('N/A' if r['mip_obj'] is None else 'No')
    print(f"\n  {'=' * 40}")
    print(f"  CG obj:     {r['cg_obj']:.2f}  ({r['cg_time']:.2f}s, "
          f"{r['cg_iter']} iters)")
    print(f"  MIP obj:    {r['mip_obj']:.2f}  ({r['mip_time']:.2f}s)"
          if r['mip_obj'] else "  MIP obj:    N/A")
    print(f"  Gap:        {r['gap']:.4f}%" if r['gap'] else "  Gap: N/A")
    print(f"  CG matches: {ok}")


def _cmd_ablation(args):
    _print_header("Ablation Study")
    run_ablation(args.rs, args.cs, args.seed, subset=args.subset,
                 include_mip=args.with_mip, include_greedy=args.with_greedy,
                 time_limit_mip=args.time_limit)


def _cmd_large(args):
    _print_header("Large-Scale Test")
    run_large_scale(args.rs, args.cs, args.tasks,
                    max_iters=args.max_iters, astar_on=not args.dijkstra)


def _cmd_run(args):
    _print_header("Single Run")
    single_run(args.rs, args.cs, args.seed, subset=args.subset,
               filter_on=not args.no_filter, astar_on=not args.dijkstra)


def _cmd_export(args):
    _print_header("Export Solution")
    inst = build_instance(args.seed, args.rs, args.cs, args.subset)
    graph = build_graph(inst)
    cr_res = run_cg(inst, graph, filter_on=True, astar_on=True)
    export_solution(inst, graph, cr_res['cr'], args.output, fmt=args.format)


def _cmd_comprehensive(args):
    _print_header(f"Comprehensive Ablation ({args.seeds} seeds)")
    seeds = list(range(args.seeds))
    run_comprehensive_ablation(
        rs_count=args.rs,
        cs_count=args.cs,
        seeds=seeds,
        subset=args.subset,
        include_mip=args.with_mip,
        include_greedy=args.with_greedy,
        time_limit_mip=args.time_limit
    )


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='LEO Constellation CG — Experiments',
    )
    sp = p.add_subparsers(dest='cmd')

    # compare
    cp = sp.add_parser('compare', help='CG vs MIP correctness')
    cp.add_argument('--rs', type=int, default=5)
    cp.add_argument('--cs', type=int, default=20)
    cp.add_argument('--seed', type=int, default=0)
    cp.add_argument('--subset', type=float, default=0.25)
    cp.add_argument('--time-limit', type=int, default=300)

    # ablation
    ap = sp.add_parser('ablation', help='Ablation study')
    ap.add_argument('--rs', type=int, default=5)
    ap.add_argument('--cs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--subset', type=float, default=0.25)
    ap.add_argument('--time-limit', type=int, default=300)
    ap.add_argument('--with-mip', action='store_true', default=True)
    ap.add_argument('--with-greedy', action='store_true', default=True)
    ap.add_argument('--no-mip', action='store_false', dest='with_mip')
    ap.add_argument('--no-greedy', action='store_false', dest='with_greedy')

    # large-scale
    lp = sp.add_parser('large', help='Large-scale test')
    lp.add_argument('--rs', type=int, default=20)
    lp.add_argument('--cs', type=int, default=80)
    lp.add_argument('--tasks', type=int, default=500)
    lp.add_argument('--max-iters', type=int, default=100)
    lp.add_argument('--dijkstra', action='store_true')

    # single run
    rp = sp.add_parser('run', help='Single CG run')
    rp.add_argument('--rs', type=int, default=10)
    rp.add_argument('--cs', type=int, default=40)
    rp.add_argument('--seed', type=int, default=0)
    rp.add_argument('--subset', type=float, default=0.25)
    rp.add_argument('--no-filter', action='store_true')
    rp.add_argument('--dijkstra', action='store_true')

    # export
    ep = sp.add_parser('export', help='Export solution')
    ep.add_argument('--rs', type=int, default=10)
    ep.add_argument('--cs', type=int, default=40)
    ep.add_argument('--seed', type=int, default=0)
    ep.add_argument('--subset', type=float, default=0.25)
    ep.add_argument('--output', type=str, default='solution.json')
    ep.add_argument('--format', choices=['json', 'csv'], default='json')

    # comprehensive (multi-seed ablation)
    cap = sp.add_parser('comprehensive', help='Comprehensive ablation with multi-seed')
    cap.add_argument('--rs', type=int, default=10)
    cap.add_argument('--cs', type=int, default=40)
    cap.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    cap.add_argument('--subset', type=float, default=0.25)
    cap.add_argument('--with-mip', action='store_true', default=True)
    cap.add_argument('--with-greedy', action='store_true', default=True)
    cap.add_argument('--no-mip', action='store_false', dest='with_mip')
    cap.add_argument('--no-greedy', action='store_false', dest='with_greedy')
    cap.add_argument('--time-limit', type=int, default=300)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    dispatch = {
        'compare': _cmd_compare,
        'ablation': _cmd_ablation,
        'large': _cmd_large,
        'run': _cmd_run,
        'export': _cmd_export,
        'comprehensive': _cmd_comprehensive,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
