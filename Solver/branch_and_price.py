"""Exact branch-and-price with explicit treatment of forbidden path columns.

Branching on a fractional task-path variable partitions the node into fixing
that path and forbidding it. Fixing consumes its capacities and removes its
task. Forbidden paths are excluded by exact Yen simple-path pricing. This is
an exact reference implementation, not an accelerated large-instance solver.
"""
from __future__ import annotations

import heapq
import math
import time
import networkx as nx

from Solver.column_generation import (ContinuousRelaxation, Flow, CAPACITY_ARC_TYPES,
    resource_coefficient, find_frac_flows)


class NodeRelaxation(ContinuousRelaxation):
    def __init__(self, *args, forbidden=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden = forbidden or {}

    def pricing_solver(self, task, dijkstra=False, penalized_arcs=None, perturbation_seed=0):
        banned = self.forbidden.get(task.task_id, set())
        if not banned:
            return super().pricing_solver(task, dijkstra=dijkstra)
        def weight(u, v, data):
            price = self.shadow_cap.get((u, v), 0.0)
            if price > 1e-9:
                raise RuntimeError(f'Invalid positive dual for an upper capacity bound: {price}')
            # A <= capacity row has a nonpositive minimization dual. Roundoff
            # around zero must not make a zero-cost connector negative.
            return data['weight'] - min(price, 0.0) * resource_coefficient(task, data)
        try:
            candidates = nx.shortest_simple_paths(self.graph, self.sources[task.task_id],
                self.sinks[task.task_id], weight=weight)
            for nodes in candidates:
                cost = sum(weight(u, v, self.graph.edges[u, v]) for u, v in zip(nodes, nodes[1:]))
                if cost - self.shadow_conv[task] >= -1e-6:
                    return None
                if tuple(nodes) not in banned:
                    return Flow((cost, nodes), task, list(zip(nodes, nodes[1:])), cost - self.shadow_conv[task])
        except nx.NetworkXNoPath:
            return None
        return None


def solve_branch_and_price(instance, graph, penalty=300, time_limit=3600):
    """Return incumbent and global bound; exactness requires a closed tree."""
    start = time.perf_counter()
    tasks_by_id = {task.task_id: task for task in instance.tasks}
    incumbent = penalty * len(instance.tasks)
    best_schedule = []
    counter = 0
    queue = [(0.0, counter, {}, {})]  # inherited bound, id, fixed paths, forbidden paths
    visited = 0
    while queue and time.perf_counter() - start < time_limit:
        inherited_bound, _, fixed, forbidden = heapq.heappop(queue)
        if inherited_bound >= incumbent - 1e-5:
            continue
        visited += 1
        residual = graph.copy()
        fixed_cost = 0.0
        feasible = True
        for task_id, nodes in fixed.items():
            task = tasks_by_id[task_id]
            for u, v in zip(nodes, nodes[1:]):
                data = residual.edges[u, v]
                fixed_cost += data['weight']
                if data['arc_type'] in CAPACITY_ARC_TYPES:
                    data['cap'] -= resource_coefficient(task, data)
                    feasible &= data['cap'] >= -1e-9
        if not feasible:
            continue
        remaining = [task for task in instance.tasks if task.task_id not in fixed]
        if not remaining:
            if fixed_cost < incumbent:
                incumbent = fixed_cost
                best_schedule = list(fixed.items())
            continue
        cr = NodeRelaxation(residual, remaining, instance.sources, instance.sinks,
            penalty, {}, instance.sat_params, k_alt=1, forbidden=forbidden)
        cr.build_restricted_masterproblem()
        cert = cr.solve(max_iterations=200)
        if not cert['certified_no_new_columns']:
            cr.rmp.dispose()
            heapq.heappush(queue, (inherited_bound, counter, fixed, forbidden))
            break
        bound = fixed_cost + cr.certified_lower_bound
        if bound >= incumbent - 1e-5:
            cr.rmp.dispose()
            continue
        fractional = find_frac_flows(cr)
        if not fractional:
            objective = fixed_cost + cr.rmp.ObjVal
            if objective < incumbent:
                incumbent = objective
                best_schedule = list(fixed.items()) + [(flow.trip.task_id, flow.path[1])
                    for flow, var in cr.x.items() if var.X > .5]
        else:
            flow, value = min(fractional, key=lambda item: abs(item[1] - .5))
            task_id, path = flow.trip.task_id, tuple(flow.path[1])
            down = {key: set(val) for key, val in forbidden.items()}
            down.setdefault(task_id, set()).add(path)
            up = {**fixed, task_id: path}
            counter += 1
            heapq.heappush(queue, (bound, counter, fixed, down))
            counter += 1
            heapq.heappush(queue, (bound, counter, up, forbidden))
        cr.rmp.dispose()
    closed = not queue
    lower = incumbent if closed else min(incumbent, min(node[0] for node in queue))
    return {'obj': incumbent, 'lower_bound': lower, 'tree_closed': closed, 'nodes': visited,
        'solve_time': time.perf_counter() - start, 'completed': len(best_schedule),
        'schedule': [{'task': task_id, 'nodes': list(path)} for task_id, path in best_schedule]}
