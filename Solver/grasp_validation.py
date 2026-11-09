"""GRASP-style randomized insertion heuristic for CTSP-LEO (Reviewer #3, comment 5).

Problem-specific heuristic:
  - randomized (RCL) task ordering biased by priority;
  - residual-capacity-aware routing: each insertion runs a shortest-path
    search on the residual graph in which saturated capacity arcs are removed,
    so a task is re-routed around bottlenecks instead of being rejected;
  - multi-start: the best schedule over `restarts` independent runs is kept.

Run on the reduced 10-RS/50-CS validation constellation (50- and 100-task
scales, 10 seeds each) and compared against Greedy, P&B, and the full ILP.
"""

import sys, os, json, time, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(1, os.path.join(REPO_DIR, 'Solver'))
sys.path.insert(2, os.path.join(REPO_DIR, 'Generator'))

import experiments as ex
import networkx as nx
from Generator.satellite_generator import build_graph
from Solver.column_generation import CAPACITY_ARC_TYPES


def run_grasp(instance, graph, restarts=20, seed=0, rcl_frac=0.3,
              time_budget=None, rejection_penalty=200):
    """Multi-start randomized insertion heuristic with residual routing."""
    rng = random.Random(seed)
    tasks = list(instance.tasks)
    base_key = {t.task_id: ex._get_node_key(graph, instance.sources[t.task_id])
                for t in tasks}
    snk_key = {t.task_id: ex._get_node_key(graph, instance.sinks[t.task_id])
               for t in tasks}

    best = None  # (obj, completed, rejected)
    best_paths = None
    t0 = time.time()
    for r in range(restarts):
        if time_budget and (time.time() - t0) > time_budget:
            break
        # priority-biased RCL ordering
        remaining = sorted(tasks, key=lambda t: t.priority, reverse=True)
        order = []
        while remaining:
            k = max(1, int(rcl_frac * len(remaining)))
            pick = rng.randrange(min(k, len(remaining)))
            order.append(remaining.pop(pick))

        # residual graph copy: saturated arcs are removed as they fill up
        residual = graph.copy()
        cap_left = {}
        for u, v, d in residual.edges(data=True):
            cap_left[(u, v)] = d.get('cap', 9999)

        total_cost = 0.0
        completed = 0
        rejected = 0
        selected_paths = []
        for t in order:
            src = base_key.get(t.task_id)
            snk = snk_key.get(t.task_id)
            if not src or not snk or src not in residual or snk not in residual:
                rejected += 1
                total_cost += rejection_penalty
                continue
            try:
                path_cost, path_nodes = nx.single_source_dijkstra(
                    residual, src, snk, weight='weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                rejected += 1
                total_cost += rejection_penalty
                continue
            if path_cost >= rejection_penalty:
                rejected += 1
                total_cost += rejection_penalty
                continue
            # commit: consume one unit on each capacitated arc
            arcs = list(zip(path_nodes[:-1], path_nodes[1:]))
            for (u, v) in arcs:
                cap_left[(u, v)] = cap_left.get((u, v), 9999) - 1
            for (u, v) in arcs:
                if (u, v) in cap_left and cap_left[(u, v)] <= 0:
                    if residual.has_edge(u, v):
                        residual.remove_edge(u, v)
            completed += 1
            total_cost += path_cost
            selected_paths.append(
                (t.task_id, t.data_volume, path_cost, path_nodes)
            )

        if (best is None or total_cost < best[0] or
                (total_cost == best[0] and completed > best[1])):
            best = (total_cost, completed, rejected)
            best_paths = selected_paths

    capacity_load = {}
    recomputed_path_cost = 0.0
    endpoints_valid = True
    for task_id, volume, path_cost, path_nodes in best_paths:
        recomputed_path_cost += sum(
            graph.edges[u, v].get('weight', 1.0)
            for u, v in zip(path_nodes[:-1], path_nodes[1:])
        )
        endpoints_valid = endpoints_valid and (
            path_nodes[0] == base_key[task_id]
            and path_nodes[-1] == snk_key[task_id]
        )
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            if graph.edges[u, v].get('arc_type') in CAPACITY_ARC_TYPES:
                capacity_load[(u, v)] = capacity_load.get((u, v), 0.0) + volume

    max_capacity_excess = max(
        (
            load - graph.edges[u, v].get('cap', 9999)
            for (u, v), load in capacity_load.items()
        ),
        default=0.0,
    )
    recomputed_obj = recomputed_path_cost + best[2] * rejection_penalty
    audit = {
        'endpoints_valid': endpoints_valid,
        'max_capacity_excess': max(0.0, max_capacity_excess),
        'objective_residual': abs(recomputed_obj - best[0]),
        'feasible': (
            endpoints_valid
            and max_capacity_excess <= 1e-9
            and abs(recomputed_obj - best[0]) <= 1e-9
        ),
    }

    return {'obj': best[0], 'completed': best[1], 'rejected': best[2],
            'restarts_used': r + 1,
            'solve_time': time.time() - t0,
            'audit': audit}


def main():
    out_dir = os.path.join(REPO_DIR, 'results', 'ejor', 'grasp')
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for tasks in (50, 100):
        subset = tasks / 200.0
        for seed in range(10):
            t0 = time.time()
            inst = ex.build_instance(seed, 10, 50, subset)
            graph = build_graph(inst)
            build_t = time.time() - t0
            n_tasks = len(inst.tasks)

            grasp = run_grasp(inst, graph, restarts=20, seed=seed,
                              time_budget=600, rejection_penalty=200)

            # P&B reference from the certified validation suite
            pb_ref = None
            ref_file = os.path.join(REPO_DIR, 'results', 'ejor',
                                    'validation',
                                    f'tasks_{tasks:04d}_seed_{seed:02d}.json')
            if os.path.exists(ref_file):
                d = json.load(open(ref_file))
                alg = d['algorithms']['CG+A* (filter,k=1)']
                pb_ref = {'obj': alg['int_obj'], 'completed': alg['completed'],
                          'lp_obj': alg['lp_obj']}
                greedy_alg = d['algorithms']['Greedy']
                greedy_ref = {
                    'obj': greedy_alg['obj'],
                    'completed': greedy_alg['completed'],
                }
            else:
                greedy_ref = None
            ilp_ref = None
            if tasks == 50 and pb_ref is not None:
                # 50-task validation suite contains the full ILP comparison
                d = json.load(open(ref_file))
                r = d['algorithms'].get('Full 0-1 ILP', {})
                if r.get('obj') is not None:
                    ilp_ref = {'obj': r['obj']}

            rec = {'tasks': n_tasks, 'seed': seed,
                   'graph_nodes': graph.number_of_nodes(),
                   'graph_edges': graph.number_of_edges(),
                   'build_time': build_t,
                   'greedy': greedy_ref,
                   'grasp': grasp, 'pb': pb_ref, 'ilp': ilp_ref}
            results.append(rec)
            print(f"tasks={n_tasks} seed={seed} "
                  f"greedy={greedy_ref['obj'] if greedy_ref else '-'} "
                  f"grasp={grasp['obj']:.0f}({grasp['completed']}) "
                  f"pb={pb_ref['obj'] if pb_ref else '-'} "
                  f"[{time.time()-t0:.1f}s]", flush=True)
            with open(os.path.join(out_dir, 'grasp_validation.json'), 'w') as f:
                json.dump(results, f, indent=2)
    print('DONE')


if __name__ == '__main__':
    main()
