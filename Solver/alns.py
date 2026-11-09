"""Independent, capacity-feasible ALNS on the complete task-routing graph.

No column pool, incumbent, or dual price from P&B is used. NativePricing is
only an exact shortest-path primitive; NetworkX provides a portable fallback.
The objective is base path cost plus M for each rejected task (not M*q).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import networkx as nx
import numpy as np

from Solver.column_generation import CAPACITY_ARC_TYPES
from Solver.native_pricing import NativePricing, LIBRARY


def node_key(graph, node):
    if isinstance(node, str):
        return node
    return getattr(graph, 'node_map', {}).get(
        node, f'{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}')


@dataclass(frozen=True)
class Route:
    cost: float
    nodes: tuple
    resources: tuple


class ResidualRouter:
    """Incremental residual capacities, including task-dependent data demand."""

    def __init__(self, instance, graph, penalty, *, native=True):
        self.graph, self.penalty = graph, float(penalty)
        self.tasks = {t.task_id: t for t in instance.tasks}
        self.ids = sorted(self.tasks)
        self.sources = {i: node_key(graph, instance.sources[i]) for i in self.ids}
        self.sinks = {i: node_key(graph, instance.sinks[i]) for i in self.ids}
        self.engine = NativePricing(graph, self.sinks.values()) if native and LIBRARY.is_file() else None
        self.edges = self.engine.edges if self.engine else list(graph.edges)
        self.edge_index = {e: k for k, e in enumerate(self.edges)}
        self.cap = np.asarray([
            float(graph.edges[e].get('cap', math.inf))
            if graph.edges[e].get('arc_type') in CAPACITY_ARC_TYPES else math.inf
            for e in self.edges])
        self.imaging = np.asarray([
            graph.edges[e].get('arc_type') == 'imaging_convergence_arc'
            for e in self.edges])
        self.demand = {q: np.where(self.imaging, 1.0, q)
                       for q in sorted({float(t.data_volume) for t in instance.tasks})}
        self.calls = 0
        self.reset()

    def reset(self):
        self.left = self.cap.copy()
        if self.engine:
            self.engine.weights = {q: self.engine.base.copy() for q in self.demand}
            for q, weights in self.engine.weights.items():
                weights[self.left + 1e-9 < self.demand[q]] = math.inf

    def _refresh(self, indices):
        if not self.engine or not len(indices):
            return
        indices = np.asarray(indices, dtype=np.intp)
        for q, weights in self.engine.weights.items():
            weights[indices] = np.where(self.left[indices] + 1e-9 >= self.demand[q][indices],
                                        self.engine.base[indices], math.inf)

    def route(self, task_id, *, blocked=()):
        self.calls += 1
        t = self.tasks[task_id]
        q = float(t.data_volume)
        source, sink = self.sources[task_id], self.sinks[task_id]
        blocked = tuple(blocked)
        if self.engine:
            weights = self.engine.weights[q]
            old = weights[list(blocked)].copy() if blocked else None
            if blocked:
                weights[list(blocked)] = math.inf
            try:
                cost, nodes, _ = self.engine.path(source, sink, q, False, self.penalty)
            except nx.NetworkXNoPath:
                return None
            finally:
                if blocked:
                    weights[list(blocked)] = old
        else:
            banned = set(blocked)

            def weight(u, v, d):
                k = self.edge_index[u, v]
                if k in banned or self.left[k] + 1e-9 < self.demand[q][k]:
                    return None
                return d['weight']

            try:
                cost, nodes = nx.single_source_dijkstra(
                    self.graph, source, sink, cutoff=self.penalty, weight=weight)
            except nx.NetworkXNoPath:
                return None
        if cost >= self.penalty - 1e-9:
            return None
        resources = tuple(self.edge_index[u, v] for u, v in zip(nodes, nodes[1:])
                          if math.isfinite(self.cap[self.edge_index[u, v]]))
        return Route(float(cost), tuple(nodes), resources)

    def commit(self, task_id, route):
        k = np.asarray(route.resources, dtype=np.intp)
        demand = self.demand[float(self.tasks[task_id].data_volume)][k]
        if np.any(self.left[k] + 1e-9 < demand):
            raise RuntimeError('Attempt to commit an infeasible route')
        self.left[k] -= demand
        self._refresh(k)

    def release(self, task_id, route):
        k = np.asarray(route.resources, dtype=np.intp)
        self.left[k] += self.demand[float(self.tasks[task_id].data_volume)][k]
        if np.any(self.left[k] > self.cap[k] + 1e-7):
            raise RuntimeError('Capacity release exceeds original capacity')
        self._refresh(k)

    def load(self, schedule):
        self.reset()
        for task_id, route in schedule.items():
            self.commit(task_id, route)


def objective(schedule, n, penalty):
    return sum(r.cost for r in schedule.values()) + penalty * (n - len(schedule))


def _order(router, candidates, rng, mode, savings):
    candidates = sorted(candidates)
    if mode == 'savings':
        return sorted(candidates, key=lambda i: -(savings[i] * rng.uniform(0.8, 1.2)))
    # Priority-biased randomized construction, as in the existing GRASP.
    remaining = sorted(candidates, key=lambda i: (-router.tasks[i].priority, i))
    order = []
    while remaining:
        order.append(remaining.pop(rng.randrange(max(1, int(0.3 * len(remaining))))))
    return order


def _repair(router, schedule, candidates, rng, mode, savings, deadline):
    if mode != 'regret2':
        for task_id in _order(router, candidates, rng, mode, savings):
            if time.perf_counter() >= deadline:
                break
            route = router.route(task_id)
            if route is not None:
                router.commit(task_id, route)
                schedule[task_id] = route
        return
    # Regret over the cheapest chain, a different acquisition, and rejection.
    # Recompute after each insertion so residual conflicts are respected.
    #
    # The shortlist is ranked by regret and drained in one pass. Inserting only the
    # single best candidate per round costs 2 routing calls per insertion while
    # re-sampling the shortlist every round, so on a 9k-task instance the call
    # count grows as O(remaining * shortlist) and the deadline is never reached
    # between calls. Draining the ranked shortlist keeps the call count linear in
    # the number of candidates, like the other two repair operators, while still
    # preferring the most regretful tasks first.
    remaining = set(candidates)
    while remaining and time.perf_counter() < deadline:
        # A sampled shortlist keeps regret repair affordable on the full graph.
        # Failed tasks are deferred to the next neighborhood, not permanently lost.
        shortlist = sorted(remaining)
        if len(shortlist) > 12:
            shortlist = sorted(rng.sample(shortlist, 12))
        infeasible = []
        scored = []
        for task_id in shortlist:
            if time.perf_counter() >= deadline:
                return
            first = router.route(task_id)
            if first is None:
                infeasible.append(task_id)
                continue
            imaging = [k for k in first.resources if router.imaging[k]]
            if not imaging:
                imaging = list(first.resources[:1])
            second = router.route(task_id, blocked=imaging) if imaging else None
            regret = (second.cost if second else router.penalty) - first.cost
            scored.append((regret, router.penalty - first.cost, rng.random(),
                           task_id, first))
        remaining.difference_update(infeasible)
        # Most regretful first; ties broken by the cheaper acquisition and then
        # randomly, so the operator stays randomized but deterministic given the rng.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, _, _, task_id, route in scored:
            if time.perf_counter() >= deadline:
                return
            # The route was priced against the residual state before any insertion
            # in this batch, so re-route before committing to stay capacity-feasible.
            fresh = router.route(task_id)
            if fresh is None:
                continue
            router.commit(task_id, fresh)
            schedule[task_id] = fresh
            remaining.discard(task_id)


def _destroy(router, schedule, rng, mode, count, unconstrained):
    ids = sorted(schedule)
    if mode == 'random':
        return rng.sample(ids, count)
    if mode == 'cost':
        # Expensive chains can be rerouted cheaply or replaced by rejected tasks.
        return sorted(ids, key=lambda i: -(schedule[i].cost * rng.uniform(0.8, 1.2)))[:count]
    if mode == 'bottleneck':
        def score(i):
            return sum((router.cap[k] - router.left[k]) / max(router.cap[k], 1e-9)
                       for k in schedule[i].resources) * rng.uniform(0.8, 1.2)
        return sorted(ids, key=lambda i: -score(i))[:count]
    # Free resources on a rejected task's unconstrained preferred chain.
    rejected = [i for i in router.ids if i not in schedule and unconstrained.get(i)]
    if not rejected:
        return rng.sample(ids, count)
    anchor = unconstrained[rng.choice(rejected)]
    resources = set(anchor.resources)
    ranked = sorted(ids, key=lambda i: -(len(resources.intersection(schedule[i].resources))
                                       + rng.random()))
    return ranked[:count]


def _roulette(names, weights, rng):
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]


def _update_weights(weights, scores, uses, reaction):
    for name in weights:
        if uses[name]:
            weights[name] = max(0.1, (1 - reaction) * weights[name]
                                + reaction * scores[name] / uses[name])
        scores[name], uses[name] = 0.0, 0


def _result(router, schedule, start, **extra):
    return {
        'obj': float(objective(schedule, len(router.ids), router.penalty)),
        'completed': len(schedule), 'rejected': len(router.ids) - len(schedule),
        'solve_time': time.perf_counter() - start,
        'routing_calls': router.calls,
        'routing_backend': 'native_exact_astar' if router.engine else 'networkx_dijkstra',
        'schedule': [{'task': i, 'nodes': list(schedule[i].nodes)} for i in sorted(schedule)],
        'rejected_tasks': [i for i in router.ids if i not in schedule],
        **extra,
    }


def run_randomized_construction(instance, graph, *, penalty=200, seed=0, restarts=20,
                                native=True):
    """Matched residual-routing, priority-randomized 20-start comparator."""
    start = time.perf_counter()
    router = ResidualRouter(instance, graph, penalty, native=native)
    preprocess = time.perf_counter() - start
    rng, best = random.Random(seed), {}
    for _ in range(restarts):
        router.reset()
        trial = {}
        _repair(router, trial, router.ids, rng, 'randomized_priority', {}, math.inf)
        if objective(trial, len(router.ids), penalty) < objective(best, len(router.ids), penalty):
            best = trial
    return _result(router, best, start, restarts_used=restarts, preprocessing_time=preprocess)


def run_alns(instance, graph, *, penalty=200, seed=0, time_limit=60.0,
             max_iterations=10000, segment=25, reaction=0.2, native=True,
             stall_iterations=None, stall_fraction=0.01):
    """Adaptive removal/repair, SA acceptance, and an independently retained best.

    `time_limit` bounds the search only. The residual-router setup, the
    unconstrained shortest-path estimates and the initial construction are timed
    separately and are not charged against it, so a large instance cannot lose its
    whole search budget to setup -- which is exactly what happens when the budget
    also covers construction. GRASP, the comparator, likewise spends whatever its
    20 restarts need and is not held to a search clock, so both heuristics are
    compared on solution quality rather than on how fast they can bootstrap.

    `stall_iterations` adds a convergence stop: the search ends after that many
    consecutive iterations without improving the incumbent, provided at least
    `stall_iterations` have run. Pass None to disable it and run the full budget.
    A converged run is reported with stop_reason='stalled', so a comparison can
    show that the budget was available and not exhausted.

    The budget is checked between shortest-path calls; validation and input graph
    construction are timed separately by the experiment driver.
    """
    if time_limit <= 0 or max_iterations < 0 or segment < 1 or not 0 < reaction <= 1:
        raise ValueError('Invalid ALNS stopping/adaptation parameters')
    start = time.perf_counter()
    rng = random.Random(seed)
    router = ResidualRouter(instance, graph, penalty, native=native)
    # These paths are estimates from the original graph, not an optimization oracle.
    unconstrained = {}
    for i in router.ids:
        unconstrained[i] = router.route(i)
    savings = {i: penalty - unconstrained[i].cost if unconstrained.get(i) else 0.0
               for i in router.ids}
    preprocess = time.perf_counter() - start
    construction_start = time.perf_counter()
    current = {}
    _repair(router, current, router.ids, rng, 'savings', savings, math.inf)
    construction = time.perf_counter() - construction_start
    search_start = time.perf_counter()
    deadline = search_start + time_limit
    current_obj = objective(current, len(router.ids), penalty)
    best, best_obj = dict(current), current_obj
    initial_obj, initial_completed = current_obj, len(current)
    trace = [{'time': time.perf_counter() - start, 'iteration': 0,
              'obj': float(best_obj), 'completed': len(best)}]
    destroys = ['random', 'cost', 'bottleneck', 'conflict']
    repairs = ['randomized_priority', 'savings', 'regret2']
    weights = [{n: 1.0 for n in names} for names in (destroys, repairs)]
    scores = [{n: 0.0 for n in names} for names in (destroys, repairs)]
    uses = [{n: 0 for n in names} for names in (destroys, repairs)]
    totals = [{n: 0 for n in names} for names in (destroys, repairs)]
    temp0 = max(1e-6, 0.01 * initial_obj / math.log(2))
    accepted, improved, iterations = 0, 0, 0
    since_improvement = 0
    stalled = False
    weight_history = []
    for iteration in range(1, max_iterations + 1):
        if time.perf_counter() >= deadline or not current:
            break
        iterations = iteration
        destroy = _roulette(destroys, weights[0], rng)
        repair = _roulette(repairs, weights[1], rng)
        count = min(len(current), max(1, int(rng.uniform(0.05, 0.25) * len(current))))
        removed = _destroy(router, current, rng, destroy, count, unconstrained)
        candidate = dict(current)
        for i in removed:
            router.release(i, candidate.pop(i))
        _repair(router, candidate, set(router.ids).difference(candidate), rng,
                repair, savings, deadline)
        value = objective(candidate, len(router.ids), penalty)
        elapsed_fraction = min(1.0, (time.perf_counter() - search_start) / time_limit)
        temperature = temp0 * (0.001 ** elapsed_fraction)
        take = value <= current_obj or rng.random() < math.exp(
            -min(700, max(0, value - current_obj) / temperature))
        reward = 0
        if value < best_obj - 1e-9:
            best, best_obj = dict(candidate), value
            improved += 1
            since_improvement = 0
            reward = 8
            trace.append({'time': time.perf_counter() - start, 'iteration': iteration,
                          'obj': float(value), 'completed': len(best)})
        elif value < current_obj - 1e-9:
            reward = 4
            since_improvement += 1
        elif take:
            reward = 1
            since_improvement += 1
        else:
            since_improvement += 1
        if take:
            current, current_obj = candidate, value
            accepted += 1
        else:
            # Restore every capacity after a rejected neighborhood, including tasks
            # newly inserted in repair. No alias to the best schedule is retained.
            router.load(current)
        for k, name in enumerate((destroy, repair)):
            scores[k][name] += reward
            uses[k][name] += 1
            totals[k][name] += 1
        if iteration % segment == 0:
            for k in range(2):
                _update_weights(weights[k], scores[k], uses[k], reaction)
            weight_history.append({'iteration': iteration,
                                   'destroy': dict(weights[0]), 'repair': dict(weights[1])})
        # Convergence stop, tested after the weight update so the reported
        # operator weights reflect what the search actually used. Two guards keep
        # it from firing during a transient: the search must have consumed a
        # meaningful share of its budget (`stall_fraction`), and the stall must
        # span `stall_iterations` consecutive iterations without an incumbent
        # improvement. The wall-clock guard matters because on small instances
        # iterations are cheap and an iteration count alone ends the run early.
        if (stall_iterations is not None and iteration >= stall_iterations
                and since_improvement >= stall_iterations
                and (time.perf_counter() - search_start) >= stall_fraction * time_limit):
            stalled = True
            break
    for k in range(2):
        _update_weights(weights[k], scores[k], uses[k], reaction)
    # Measured, not nominal: the deadline is only tested between shortest-path
    # calls, so one destroy/repair iteration on a large instance can run far past
    # it. Reporting `time_limit` here would hide that overrun from the accounting.
    search_time = time.perf_counter() - search_start
    if stalled:
        stop_reason = 'stalled'
    elif time.perf_counter() >= deadline:
        stop_reason = 'time_limit'
    else:
        stop_reason = 'iteration_limit'
    return _result(router, best, start, initial_obj=float(initial_obj),
                   initial_completed=initial_completed, preprocessing_time=preprocess,
                   construction_time=construction, search_time=search_time,
                   iterations=iterations, accepted=accepted, best_improvements=improved,
                   stop_reason=stop_reason,
                   destroy_uses=totals[0], repair_uses=totals[1],
                   destroy_weights=weights[0], repair_weights=weights[1],
                   weight_history=weight_history, best_trace=trace,
                   time_to_best=trace[-1]['time'])
