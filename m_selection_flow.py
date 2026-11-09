"""Exact service ceiling and strict penalty threshold for unit common-sink flow.

Let F(k) be minimum physical path cost when exactly k tasks are served. The
integral min-cost flow value function F is convex. Thus its largest discrete
marginal cost is F(K)-F(K-1), where K is maximum feasible service. Every
optimum of F(k)+M*(n-k) serves K if M exceeds that last marginal. Equality can
tie with K-1. The smallest *integer*, strictly service-prioritizing penalty is
floor(F(K)-F(K-1))+1. This is instance-specific, not a universal economic M.

Rejections pass through an extra collector and a single bounded connector;
fixing rejection count remains an ordinary integral network-flow model.
"""
from collections import defaultdict
import math
import time

import gurobipy as gp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from alns_validation import audit_schedule
from gurobi_full_models_m200 import graph_arrays, configure, path_in_active_flow


class MSelectionFlow:
    def __init__(self, instance, graph, time_limit=600, threads=8, soft_mem_gb=8,
                 log_file=None):
        self.instance, self.graph = instance, graph
        start = time.perf_counter()
        a = self.a = graph_arrays(instance, graph)
        if any(abs(t.data_volume-1) > 1e-12 for t in a['tasks']):
            raise ValueError('Requires unit-volume tasks')
        if any(graph.in_degree(a['nodes'][s]) for s in a['sources']):
            raise ValueError('Requires sources without incoming arcs')
        n, m, nt = len(a['nodes']), len(a['edges']), len(a['tasks'])
        self.m, self.nt = m, nt
        tails = np.concatenate([a['tails'], a['sources'], [n]])
        heads = np.concatenate([a['heads'], np.full(nt, n), [a['sink']]])
        cols = np.tile(np.arange(m+nt+1), 2)
        incidence = coo_matrix((np.concatenate([np.ones(m+nt+1), -np.ones(m+nt+1)]),
                               (np.concatenate([tails, heads]), cols)),
                              shape=(n+1, m+nt+1)).tocsr()
        rhs = np.zeros(n+1)
        rhs[a['sources']], rhs[a['sink']] = 1, -nt
        physical_ub = np.where(a['resource'], np.minimum(nt, np.floor(a['caps']+1e-9)), nt)
        upper = np.concatenate([physical_ub, np.ones(nt), [nt]])
        self.physical_objective = np.concatenate([a['costs'], np.zeros(nt+1)])
        self.model = gp.Model('M_selection_exact_collector_network')
        configure(self.model, time_limit, threads, soft_mem_gb, log_file)
        self.x = self.model.addMVar(m+nt+1, lb=0, ub=upper)
        self.model.addMConstr(incidence, self.x, '=', rhs)
        self.model.update()
        self.build_time = time.perf_counter()-start
        self.solve_count = 0

    def close(self):
        self.model.dispose()

    def solve(self, mode, penalty=0, rejected_count=None):
        objective = self.physical_objective.copy()
        if mode == 'maximum_service':
            objective[:] = 0
            objective[-1] = 1
        elif mode == 'penalty':
            objective[-1] = penalty
        elif mode != 'fixed_service':
            raise ValueError(mode)
        lower = np.zeros(self.nt+1)
        upper = np.concatenate([np.ones(self.nt), [self.nt]])
        if rejected_count is not None:
            lower[-1] = upper[-1] = rejected_count
        self.x[self.m:].LB, self.x[self.m:].UB = lower, upper
        self.x.Obj = objective
        start = time.perf_counter()
        self.model.optimize()
        optimize_time = time.perf_counter()-start
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'Uncertified stage: {mode}, status={self.model.Status}')
        values = self.x.X
        fractional = float(np.max(np.abs(values-np.rint(values))))
        if fractional > 1e-5:
            raise RuntimeError(f'Nonintegral network optimum: {fractional}')
        start = time.perf_counter()
        a, m = self.a, self.m
        residual = np.rint(values[:m]).astype(np.int32)
        adjacency = defaultdict(list)
        for edge in np.flatnonzero(residual):
            adjacency[int(a['tails'][edge])].append((int(a['heads'][edge]), int(edge)))
        selected, rejected = [], []
        for i, task in enumerate(a['tasks']):
            if values[m+i] > .5:
                rejected.append(task.task_id)
            else:
                nodes, edges = path_in_active_flow(a['sources'][i], a['sink'], adjacency, residual)
                residual[edges] -= 1
                selected.append({'task': task.task_id, 'nodes': [a['nodes'][v] for v in nodes]})
        circulation_cost = float(np.dot(residual, a['costs']))
        if circulation_cost > 1e-6 and mode != 'maximum_service':
            raise RuntimeError('Positive-cost leftover circulation')
        # The first stage has no physical-cost objective: discard any residual
        # circulation before auditing the selected paths. Later stages minimize
        # actual cost and must have no positive-cost circulation.
        cost = float(np.dot(values[:m], a['costs']))-circulation_cost
        result = {'status': 'OPTIMAL', 'mode': mode, 'penalty': penalty,
                  'fixed_rejected_count': rejected_count,
                  'obj': cost+penalty*len(rejected),
                  'optimization_objective': float(self.model.ObjVal),
                  'completed': len(selected), 'rejected': len(rejected),
                  'physical_cost': cost, 'schedule': selected, 'rejected_tasks': rejected,
                  'discarded_circulation_cost': circulation_cost,
                  'fractional_residual': fractional, 'optimize_time': optimize_time,
                  'extraction_time': time.perf_counter()-start,
                  'gurobi_runtime': float(self.model.Runtime),
                  'warm_started': self.solve_count > 0,
                  'timing_scope': 'parameter exploration on a reused model; not algorithm benchmark timing'}
        audit_start = time.perf_counter()
        result['audit'] = audit_schedule(self.instance, self.graph, result, penalty)
        result['audit_time'] = time.perf_counter()-audit_start
        assert result['audit']['feasible'], result['audit']
        assert abs(values[-1]-len(rejected)) < 1e-5
        assert abs(float(np.dot(objective, values))-self.model.ObjVal) < 1e-5
        self.solve_count += 1
        return result

    def service_frontier_endpoint(self):
        ceiling = self.solve('maximum_service')
        r = ceiling['rejected']
        assert abs(ceiling['optimization_objective']-r) < 1e-5
        best = self.solve('fixed_service', rejected_count=r)
        previous = self.solve('fixed_service', rejected_count=r+1) if best['completed'] else None
        delta = best['physical_cost']-previous['physical_cost'] if previous else 0
        assert delta >= -1e-5
        if abs(delta-round(delta)) < 1e-7:
            delta = float(round(delta))
        minimum_integer = math.floor(delta)+1 if best['completed'] else 0
        proof = {'maximum_completed': best['completed'], 'minimum_rejected': r,
                 'minimum_cost_at_maximum_service': best['physical_cost'],
                 'minimum_cost_at_one_fewer_completed': previous['physical_cost'] if previous else None,
                 'critical_penalty': delta,
                 'smallest_strict_integer_penalty': minimum_integer,
                 'tie_at_critical_penalty': bool(previous),
                 'guarantee': 'M > critical_penalty forces every global optimum to serve the maximum feasible count; integer recommendation is instance-specific',
                 'model_build_time': self.build_time}
        return proof, {'maximum_service': ceiling, 'minimum_cost_maximum_service': best,
                       'minimum_cost_one_fewer_service': previous}

    def individual_path_diagnostic(self):
        a = self.a
        keep = (~a['resource']) | (np.floor(a['caps']+1e-9) >= 1)
        reverse = coo_matrix((a['costs'][keep], (a['heads'][keep], a['tails'][keep])),
                             shape=(len(a['nodes']), len(a['nodes']))).tocsr()
        costs = dijkstra(reverse, directed=True, indices=a['sink'])[a['sources']]
        finite = costs[np.isfinite(costs)]
        return {'individually_reachable': int(len(finite)),
                'individually_unreachable': int(np.sum(~np.isfinite(costs))),
                'minimum_path_costs_by_task': [float(c) if np.isfinite(c) else None for c in costs],
                'reachable_with_cost_below_200': int(np.sum(costs < 200-1e-9)),
                'reachable_with_cost_equal_200': int(np.sum(np.abs(costs-200) <= 1e-9)),
                'reachable_with_cost_above_200': int(np.sum(np.isfinite(costs) & (costs > 200+1e-9))),
                'finite_cost_quantiles_0_25_50_75_95_100': np.quantile(finite, [0,.25,.5,.75,.95,1]).tolist() if len(finite) else None,
                'scope': 'Individual unit-capacity paths without competition; lower costs do not determine globally marginal service costs'}
