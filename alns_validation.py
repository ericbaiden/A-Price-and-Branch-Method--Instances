"""Schedule verification independent of ALNS residual state and path costs."""
from __future__ import annotations
import math
from collections import Counter


# Deliberately recompute units from graph attributes rather than solver arrays.
RESOURCE_TYPES = {'storage_arc', 'imaging_convergence_arc',
                  'cross_domain_convergence_arc', 'cross_domain_arc', 'isl_arc',
                  'downlink_arc', 'downlink_convergence_arc', 'vgs_arc'}


def _key(graph, node):
    if isinstance(node, str):
        return node
    mapping = getattr(graph, 'node_map', {})
    if node in mapping:
        return mapping[node]
    return f'{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}'


def audit_schedule(instance, graph, result, penalty):
    tasks = {t.task_id: t for t in instance.tasks}
    seen, loads, errors = set(), Counter(), []
    cost = 0.0
    for entry in result['schedule']:
        task_id, nodes = entry['task'], entry['nodes']
        if task_id not in tasks or task_id in seen:
            errors.append(f'Unknown/duplicate selected task: {task_id}')
            continue
        seen.add(task_id)
        if not nodes or nodes[0] != _key(graph, instance.sources[task_id]) or nodes[-1] != _key(graph, instance.sinks[task_id]):
            errors.append(f'Incorrect endpoints: {task_id}')
        if len(nodes) != len(set(nodes)):
            errors.append(f'Non-simple path: {task_id}')
        for u, v in zip(nodes, nodes[1:]):
            if not graph.has_edge(u, v):
                errors.append(f'Missing arc: {task_id}')
                continue
            d = graph.edges[u, v]
            cost += float(d['weight'])
            if d.get('arc_type') in RESOURCE_TYPES:
                loads[u, v] += 1.0 if d['arc_type'] == 'imaging_convergence_arc' else float(tasks[task_id].data_volume)
    rejected = result['rejected_tasks']
    if len(set(rejected)) != len(rejected) or seen.intersection(rejected) or seen.union(rejected) != set(tasks):
        errors.append('Selected/rejected tasks do not form a partition')
    excess = max([0.0] + [load - float(graph.edges[e]['cap']) for e, load in loads.items()])
    expected = cost + penalty * len(rejected)
    residual = abs(expected - result['obj'])
    feasible = (not errors and excess <= 1e-7 and math.isfinite(residual) and residual <= 1e-6
                and result['completed'] == len(seen) and result['rejected'] == len(rejected))
    return {'feasible': bool(feasible), 'errors': errors[:20],
            'max_capacity_excess': excess, 'objective_residual': residual,
            'completed_path_cost': cost, 'rejection_cost': penalty * len(rejected),
            'resources_used': len(loads)}
