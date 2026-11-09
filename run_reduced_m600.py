"""Rerun the reduced-constellation baseline at M=600.

The manuscript states "all algorithm comparisons use M=600" (Section 5.1), but
the current reduced-constellation table (5.2.1, from results/ejor/grasp) uses
M=300. This script reruns it at M=600 to unify the penalty across the paper.

Runs Greedy, GRASP, and P&B on the 10-RS/50-CS constellation at 50/100 tasks
× 10 seeds (20 instances). Does not rerun the full ILP, which is too slow at
100 tasks and is not an algorithm comparison (it is an exactness check).
"""
import json
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
from Generator.satellite_generator import build_graph
from Solver.grasp_validation import run_grasp

PENALTY = 600
OUT = ROOT / 'results/reduced_m600'


def run_greedy_reduced(instance, graph, penalty):
    """Greedy on the reduced constellation, same as the full-constellation one."""
    import networkx as nx
    start = time.perf_counter()
    cap_left = {}
    for u, v, d in graph.edges(data=True):
        cap_left[(u, v)] = d.get('cap', 9999)

    schedule, rejected = [], []
    total_cost, completed = 0.0, 0

    for t in sorted(instance.tasks, key=lambda t: t.priority, reverse=True):
        src = ex._get_node_key(graph, instance.sources.get(t.task_id))
        snk = ex._get_node_key(graph, instance.sinks.get(t.task_id))
        path_cost = path_nodes = None
        if src and snk:
            try:
                path_cost, path_nodes = nx.single_source_dijkstra(
                    graph, src, snk, weight='weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                path_nodes = None

        feasible = bool(path_nodes) and all(
            cap_left.get((u, v), 0) >= 1 for u, v in zip(path_nodes, path_nodes[1:]))

        if feasible:
            for u, v in zip(path_nodes, path_nodes[1:]):
                cap_left[(u, v)] = max(0, cap_left[(u, v)] - 1)
            completed += 1
            total_cost += path_cost
            schedule.append({'task': t.task_id, 'nodes': list(path_nodes)})
        else:
            rejected.append(t.task_id)
            total_cost += penalty

    return {'obj': total_cost, 'solve_time': time.perf_counter() - start,
            'completed': completed, 'rejected': len(rejected),
            'schedule': schedule, 'rejected_tasks': rejected}


def run_pb_reduced(instance, graph, penalty):
    """P&B on the reduced constellation."""
    from Solver.column_generation import ContinuousRelaxation
    result = ContinuousRelaxation(instance, graph, penalty).solve()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Set experiments.DUMMY_COST so that if anything calls it the penalty is right.
    ex.DUMMY_COST = PENALTY

    results = []
    for tasks in (50, 100):
        subset = tasks / 200.0
        for seed in range(10):
            t0 = time.perf_counter()
            inst = ex.build_instance(seed, 10, 50, subset)
            graph = build_graph(inst)
            build_time = time.perf_counter() - t0

            # Three algorithms in the same order as the full constellation.
            greedy_result = run_greedy_reduced(inst, graph, PENALTY)
            grasp_result = run_grasp(inst, graph, restarts=20, seed=seed,
                                     time_budget=600, rejection_penalty=PENALTY)
            pb_result = run_pb_reduced(inst, graph, PENALTY)

            record = {
                'tasks': tasks, 'seed': seed, 'penalty': PENALTY,
                'graph_nodes': len(graph), 'graph_edges': graph.number_of_edges(),
                'build_time': build_time,
                'algorithms': {
                    'Greedy': greedy_result,
                    'GRASP': grasp_result,
                    'P&B': pb_result,
                }
            }
            path = OUT / f'tasks_{tasks:04d}_seed_{seed:02d}.json'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2)
            print(f'DONE tasks={tasks} seed={seed} '
                  f'Greedy={greedy_result["obj"]:.0f}/{greedy_result["completed"]} '
                  f'GRASP={grasp_result["obj"]:.0f}/{grasp_result["completed"]} '
                  f'P&B={pb_result["obj"]:.0f}/{pb_result["completed"]} '
                  f'total={time.perf_counter()-t0:.1f}s', flush=True)
            results.append(record)

    print(f'ALL COMPLETE: {len(results)} instances', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
