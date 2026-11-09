"""
Column Generation solver for LEO Constellation Cross-Domain Collaborative Scheduling.

Based on intermodalTransportationNetworksCG column_generation.py architecture.
Uses Gurobi as the LP/MIP solver.

Reference: README Section 3 (Algorithm Design) and Section 5.4 (Column Generation Solver)

Solver: Gurobi
"""

import math
import time
import gurobipy as grp
from gurobipy import GRB
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os

# Add parent directory for A_star and Dijkstra (they're copied locally)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import A_star and Dijkstra from local Solver
try:
    import Solver.A_star as A_star
    import Solver.Dijkstra as Dijkstra
    HAS_LOCAL_SOLVER = True
except ImportError:
    HAS_LOCAL_SOLVER = False


# Arc types whose capacity must be enforced consistently in the RMP/MIP.
CAPACITY_ARC_TYPES = {
    'storage_arc',
    'imaging_convergence_arc',
    'cross_domain_convergence_arc',
    'cross_domain_arc',
    'isl_arc',
    'downlink_arc',
    'downlink_convergence_arc',
    'vgs_arc',
}


def resource_coefficient(task, edge_data):
    """Acquisition counts on imaging connectors; GB on data resources."""
    return 1.0 if edge_data.get('arc_type') == 'imaging_convergence_arc' else task.data_volume


class Flow:
    """Represents a flow (path) for a task.

    Reference: intermodalTransportationNetworksCG Flow class
    """

    def __init__(self, path: Tuple, trip, arcs: List[Tuple], reduced_cost: float):
        self.path = path          # (cost, nodes)
        self.trip = trip          # Task object
        self.arcs = arcs          # List of (start_node, end_node)
        self.reduced_cost = reduced_cost
        self.cost = None

    def __eq__(self, other):
        return (self.path == other.path and
                self.trip == other.trip and
                self.arcs == other.arcs and
                self.cost == other.cost and
                self.reduced_cost == other.reduced_cost)

    def __hash__(self):
        # Note: reduced_cost is NOT included because it varies between pricing iterations
        return hash((self.path[0], tuple(self.path[1]), self.trip,
                    tuple(self.arcs), self.cost))


class ContinuousRelaxation:
    """Column generation solver with extended constraints.

    Reference: README Section 3.4 Algorithm 2: OneStageColumnGeneration

    Extended constraints (from README Section 2.3):
    - CAP: Edge capacity constraints (storage/bandwidth)
    - CAP: Edge capacity constraints (all arc types with explicit capacities)
    - CONV: Convergence constraints (sum(flows) + dummy >= 1)

    Dual variables:
    - shadow_conv[t]: task-path consistency (α_t)
    - shadow_cap[e]: edge capacity (β_e)
    - Note: All other constraints handled via virtual edge capacities
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        tasks: List,
        sources: Dict[str, object],
        sinks: Dict[str, object],
        dummy_cost: float,
        heuristic: Dict,
        sat_params: Dict = None,
        k_alt: int = 3
    ):
        """Initialize column generation solver.

        Args:
            graph: networkx DiGraph of time-expanded network
            tasks: List of Task objects
            sources: Dict mapping task_id -> source TempNode
            sinks: Dict mapping task_id -> sink TempNode
            dummy_cost: Cost for dummy (rejected) tasks
            heuristic: A* heuristic dictionary
            sat_params: Dict mapping sat_id -> SatelliteParams (for energy constraints)
            k_alt: Number of alternative paths per task for diversity (default: 3)
        """
        self.k_alt = k_alt  # Store for use in solve_pricing_problems
        # General inputs
        self.graph = graph
        self.tasks = tasks
        self.sat_params = sat_params if sat_params else {}

        # Convert TempNode sources/sinks to string keys
        self.sources = {}
        self.sinks = {}
        node_map = getattr(graph, 'node_map', {})
        for task_id, src_node in sources.items():
            key = f"{src_node.satellite}_{src_node.time_slot}_{src_node.node_type}_{src_node.route}"
            self.sources[task_id] = key
        for task_id, sink_node in sinks.items():
            key = f"{sink_node.satellite}_{sink_node.time_slot}_{sink_node.node_type}_{sink_node.route}"
            self.sinks[task_id] = key
        common_sink = next((node for node in graph if 'virtual_ground' in node), None)
        if common_sink is not None:
            self.sinks = {task_id: common_sink for task_id in self.sinks}
        self.dummy_cost = dummy_cost
        # One destination table per distinct sink, computed on the actual graph.
        # This also avoids sharing an unsafe altitude fallback with certification.
        self.heuristic = {node: {} for node in graph}
        for sink in set(self.sinks.values()):
            distances = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), sink, weight='weight')
            for node, distance in distances.items():
                self.heuristic[node][sink] = distance

        # RMP components
        self.dummy_vars: Dict = {}
        self.cap_constr: Dict = {}      # Edge capacity constraints
        self.conv_constr: Dict = {}      # Convergence upper-bound (dummy <= 1)
        self.conv_constr_lb: Dict = {}  # Convergence lower-bound (dummy >= 0)
        self.flow_vars: Dict = {}       # Flow variables per task
        # Note: All other constraints handled via virtual edge capacities

        self.x: Dict = {}                # Decision variables (flows)
        self.shadow_cap: Dict = {}       # Edge capacity duals (β_e)
        self.shadow_conv: Dict = {}      # Convergence constraint duals (α_t)

        for task in tasks:
            self.shadow_conv[task] = -math.inf

        self.rmp = None
        self.flows: Dict = {task: [] for task in tasks}

        # Algorithm state
        self.change = True
        self.lower_bound = [-math.inf]
        self.objVal = []
        self.rerun_trips: Set = set(self.tasks)  # Start with all tasks
        self.upper_bound = math.inf

        # Statistics
        self.num_rerun = []
        self.time_pp = []
        self.found_flows = []
        self.time_filter = []
        self.explored_nodes_pricing = 0
        self.num_pricing_problems = 0  # Total pricing problems solved
        self.num_it = -1
        self.solve_time = 0
        self.max_slot = 0

        # Timing breakdown (updated during solve)
        self.timing_breakdown = {
            'rmp_solving': 0.0,
            'pricing_filter': 0.0,
            'pricing_problems': 0.0,
            'update_duals': 0.0,
        }
        # Iteration history for convergence analysis
        self.iteration_history = []  # List of dicts with per-iteration data
        self.filter_history = []  # Filter decisions and screening statistics
        self._flow_arc_index = None  # Lazy index for the task-chain filter
        self.rmp_solve_count = 0
        # Total pricing problems solved (cumulative)
        self.total_pricing_calls = 0
        from Solver.native_pricing import NativePricing, LIBRARY
        self.native_pricing = NativePricing(graph, self.sinks.values()) if LIBRARY.is_file() else None

    def build_restricted_masterproblem(self):
        """Build RMP with all constraint types.

        Reference: README Section 5.4.1 ContinuousRelaxation extension
        Based on intermodalTransportationNetworksCG build_restricted_masterproblem
        """
        self.rmp = grp.Model('RMP')
        self.rmp.setParam('OutputFlag', 0)
        # Optimization: Enable presolve for faster LP solving
        self.rmp.setParam('Presolve', 2)
        # Optimization: Auto-select method (barrier often faster for large LPs)
        self.rmp.setParam('Method', -1)
        # Optimization: Use multiple threads
        self.rmp.setParam('Threads', 8)
        self.rmp.setParam('NumericFocus', 1)  # Better numerical stability

        # Dummy variables for rejected tasks
        # dummy = 1 means rejected (passenger not served)
        # NO ub=1 to allow dummy > 1 when multiple paths are considered
        # Reference: intermodalTransportationNetworksCG line 93
        self.dummy_vars = {
            task: self.rmp.addVar(lb=0,
                                  obj=self.dummy_cost,
                                  name=f'dummy_{task.task_id}')
            for task in self.tasks
        }

        # Constraint 1: Edge capacity constraints (CAP)
        # Reference: README Section 2.3 constraint (2)
        # All arcs with finite capacity participate in pricing:
        # - storage_arc: storage capacity limits
        # - imaging_convergence_arc: RS per-slot imaging limit
        # - cross_domain_convergence_arc: CS per-slot receiving limit
        # - cross_domain_arc: RS->CS transition capacity
        # - isl_arc: ISL bandwidth capacity (25 tasks per ISL per slot)
        # - downlink_arc: downlink bandwidth capacity
        # - downlink_convergence_arc: GS antenna capacity
        first_task = next(iter(self.tasks))
        zero_expr = 0 * self.dummy_vars[first_task]
        self.cap_constr = {}
        # A capacity row is introduced when the first column uses its resource.
        # Earlier columns cannot use an unseen resource, so its old coefficients
        # are all zero. Omitted zero-load rows do not change the full master.

        # Constraint 3: Convergence constraints (CONV)
        # Reference: intermodalTransportationNetworksCG line 100-102
        # Constraint: dummy >= 1
        # This forces each task to be served (dummy=1 means using the "not served" option costs dummy_cost)
        # Flows are added later via Column to also participate in this constraint
        self.conv_constr = {}
        for task in self.tasks:
            self.conv_constr[task] = self.rmp.addConstr(
                self.dummy_vars[task] == 1,
                name=f'conv_constr_{task.task_id}'
            )

        # Initialize shadow values
        self.shadow_cap = {key: 0.0 for key in self.cap_constr.keys()}
        # shadow_conv 保持在 __init__ 中初始化的 -math.inf，让cutoff机制正常工作
        # 参考版本也是 -math.inf，这样 pricing cutoff = -inf - 0.0001 = -inf，
        # 第一次迭代会找最短路径，之后根据实际最短路径成本更新
        # Note: All other constraints handled via virtual edge capacities

    def solve_rmp(self):
        """Solve RMP in each column generation iteration.

        Reference: intermodalTransportationNetworksCG solve_rmp
        """
        self.num_it += 1
        self.rmp_solve_count += 1
        self.rmp.reset()  # Important for Gurobi dual variables
        self.rmp.optimize()

        # Check status
        if self.rmp.Status != GRB.OPTIMAL:
            raise RuntimeError(f"RMP status is {self.rmp.Status}, not optimal")

    def _is_differentiable_arc(self, u, v) -> bool:
        """Check if arc type is suitable for route differentiation.

        Returns True for arcs whose alternative selection creates a meaningfully
        different path (different RS, different CS, different slot).
        """
        arc_type = self.graph.edges[u, v].get('arc_type', '')
        # Only penalize imaging_arc (which RS) and cross_domain_arc (which CS)
        return arc_type in ['imaging_arc', 'cross_domain_arc']

    def solve_pricing_problems(self, list_of_tasks: List, dijkstra: bool = False, num_workers: int = 8):
        """Solve pricing problems for a list of tasks.

        In the first pricing call (no existing columns), generates K diverse
        paths per task to provide column variety for the RMP.

        Args:
            list_of_tasks: List of tasks to solve pricing for
            dijkstra: Use Dijkstra instead of A*
            num_workers: Number of parallel workers for pricing (default: 8)
        """
        new_flows = []
        if self.native_pricing is not None:
            self.native_pricing.prepare(self.shadow_cap, [task.data_volume for task in list_of_tasks])

        # Use self.k_alt as base, adjust for very large scale
        num_tasks = len(list_of_tasks)
        use_diversity = self.k_alt > 1 and num_tasks <= 200

        if use_diversity:
            # Very large scale: use minimal diversity to save time
            k_alt = max(2, min(self.k_alt, 3))
        else:
            k_alt = 1

        # Use iteration number as perturbation seed for diverse path generation
        perturbation_seed = self.num_it + 1

        def solve_task_with_diversity(task):
            """Solve one pricing subproblem and optionally generate alternatives."""
            task_flows = []
            blocked_arcs = set()

            # First flow (best reduced cost path)
            flow = self.pricing_solver(task, dijkstra=dijkstra, perturbation_seed=perturbation_seed)
            if flow is not None:
                task_flows.append(flow)
                for u, v in flow.arcs:
                    if self._is_differentiable_arc(u, v):
                        blocked_arcs.add((u, v))
                    edge_data = self.graph.edges.get((u, v), {})
                    if edge_data.get('cap', 9999) == 1:
                        blocked_arcs.add((u, v))

            # Alternative flows for diversity (only if k_alt > 1)
            for k in range(1, k_alt):
                alt_flow = self.pricing_solver(
                    task, dijkstra=dijkstra, penalized_arcs=blocked_arcs,
                    perturbation_seed=perturbation_seed + k
                )
                if alt_flow is None:
                    break
                # Check for duplicates
                path_nodes = tuple(alt_flow.path[1])
                is_dup = False
                for existing in self.flows[task]:
                    if tuple(existing.path[1]) == path_nodes:
                        is_dup = True
                        break
                for nf in task_flows:
                    if nf is not alt_flow and tuple(nf.path[1]) == path_nodes:
                        is_dup = True
                        break
                if is_dup:
                    continue
                task_flows.append(alt_flow)
                for u, v in alt_flow.arcs:
                    edge_data = self.graph.edges.get((u, v), {})
                    if edge_data.get('cap', 9999) <= 10:
                        blocked_arcs.add((u, v))

            return task_flows

        if len(list_of_tasks) > 2:
            # Parallel pricing: use ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(solve_task_with_diversity, task): task for task in list_of_tasks}
                for future in futures:
                    new_flows.extend(future.result())
        else:
            for task in list_of_tasks:
                new_flows.extend(solve_task_with_diversity(task))

        self.add_new_flows(new_flows)
        self.calculate_LB(new_flows)

    def pricing_solver(self, task, dijkstra: bool = False, penalized_arcs: Set[Tuple] = None,
                       perturbation_seed: int = 0) -> Optional[Flow]:
        """Solve pricing problem for a single task.

        The pricing subproblem finds the minimum reduced cost path:
        rc = c_{t,p} - α_t - sum_e β_e * d_t * a_{etp}
             - sum_{(s,k)} (ρ_sk * b^{link}_{sktp} + ξ_sk * b^{img}_{sktp} + γ_sk * u_{sktp})

        When penalized_arcs is provided (for diverse path generation), an
        additional penalty is added to those arcs to force alternative routing.

        Args:
            perturbation_seed: Seed for random perturbation to explore diverse columns

        Returns:
            Flow object if negative reduced cost found, else None
        """
        PENALTY = 3.0  # Moderate penalty for arcs to avoid during diversity generation

        def new_cost(u, v, data):
            """Compute reduced cost for edge (u, v) for this specific task.

            Reduced cost formula:
            rc = c_{t,p} - α_t - Σ β_e * a_e - Σ ρ_sk * b^{link} - Σ ξ_sk * b^{img} - Σ γ_sk * u

            All dual variables should be considered to find the true minimum reduced cost path.
            """
            start_node = data.get('start_node')
            end_node = data.get('end_node')
            arc_type = data.get('arc_type', '')

            # Base cost
            base_cost = data['weight']

            # Dual adjustment for ALL capacity-constrained arcs (not just bottleneck cap=1)
            # ISL and downlink arcs have finite capacity (1000/20000) and participate in pricing
            key = (u, v)
            arc_cap = data.get('cap', 9999)
            if key in self.shadow_cap:
                base_cost -= self.shadow_cap[key] * resource_coefficient(task, data)

            # Penalty for arcs to avoid (diverse path generation)
            if penalized_arcs and key in penalized_arcs:
                base_cost += PENALTY

            return base_cost

        try:
            source = self.sources[task.task_id]
            target = self.sinks[task.task_id]
        except KeyError as exc:
            raise RuntimeError(f"Missing endpoint for task {task.task_id}") from exc

        # Solve shortest path
        # If shadow_conv is very low, don't use cutoff
        cutoff = self.shadow_conv[task] - 1e-6
        if cutoff < -1e18:
            cutoff = math.inf

        if self.native_pricing is not None and penalized_arcs is None:
            try:
                path = self.native_pricing.path(source, target, task.data_volume, dijkstra, cutoff)
            except nx.NetworkXNoPath:
                return None
        elif dijkstra:
            try:
                if HAS_LOCAL_SOLVER:
                    path = Dijkstra.single_source_dijkstra(
                        self.graph,
                        source=source,
                        target=target,
                        cutoff=cutoff,
                        weight=new_cost
                    )
                else:
                    path = nx.single_source_dijkstra(
                        self.graph,
                        source=source,
                        target=target,
                        cutoff=cutoff,
                        weight=new_cost
                    )
            except nx.NetworkXNoPath:
                return None
        else:
            try:
                if HAS_LOCAL_SOLVER:
                    path = A_star.path(
                        self.graph,
                        source=source,
                        target=target,
                        heuristic_dict=self.heuristic,
                        weight=new_cost,
                        cutoff=cutoff
                    )
                else:
                    path = nx.single_source_dijkstra(
                        self.graph,
                        source=source,
                        target=target,
                        cutoff=cutoff,
                        weight=new_cost
                    )
            except nx.NetworkXNoPath:
                return None

        path_arcs = list(zip(path[1], path[1][1:]))
        self.num_pricing_problems += 1
        self.explored_nodes_pricing += path[2] if len(path) > 2 else 0

        rc = path[0] - self.shadow_conv[task]
        if rc >= -1e-6:
            return None
        return Flow(path, task, path_arcs, rc)

    def add_new_flows(self, new_flows: List[Flow]):
        """Add new columns to RMP for discovered flows.
        Using Gurobi Column concept for constraint coefficients
        """
        self.found_flows.append(len(new_flows))

        # Optimization: Build flow-arc index for faster pricing filter
        self._flow_arc_index = None  # Lazy initialization

        for flow in new_flows:
            # Check for duplicates using path and arcs only (not cost, which may be None)
            # Two flows are the same if they have the same path nodes
            is_duplicate = False
            path_nodes = tuple(flow.path[1])  # The node list
            for existing_flow in self.flows[flow.trip]:
                existing_nodes = tuple(existing_flow.path[1])
                if path_nodes == existing_nodes and tuple(flow.arcs) == tuple(existing_flow.arcs):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Compute objective value (path cost)
            objective = 0
            for a in flow.arcs:
                objective += self.graph[a[0]][a[1]]['weight']

            flow.cost = objective

            # Build column with coefficients for all constraints
            c = grp.Column()

            # Add to capacity constraints with data_volume coefficient
            # All arcs with finite capacity participate in column
            for (u, v) in flow.arcs:
                edge_data = self.graph.edges[(u, v)]
                arc_type = edge_data['arc_type']

                # All capacity-constrained arcs
                if arc_type in CAPACITY_ARC_TYPES:
                    cap_key = (u, v)
                    if cap_key not in self.cap_constr:
                        self.cap_constr[cap_key] = self.rmp.addConstr(
                            grp.LinExpr() <= edge_data['cap'], name=f'cap_{u}_{v}')
                        self.shadow_cap[cap_key] = 0.0
                    c.addTerms(resource_coefficient(flow.trip, edge_data), self.cap_constr[cap_key])

            # Add to convergence constraint (coefficient 1.0 matches MIP formulation)
            # IMPORTANT: this must be OUTSIDE the for(u,v) loop (add once, not per arc)
            if flow.trip in self.conv_constr:
                c.addTerms(1.0, self.conv_constr[flow.trip])

# Add decision variable to RMP
            var_name = f'flow_{flow.trip.task_id}_{len(self.flows[flow.trip])}'
            var = self.rmp.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                obj=flow.cost,
                name=var_name,
                column=c
            )

            self.flows[flow.trip].append(flow)
            self.x[flow] = var

    def calculate_LB(self, new_flows: List[Flow]):
        """Calculate lower bound from pricing results."""
        # Only a completed exact full pass supports a global lower bound.
        self.lower_bound.append(-math.inf)

    def solve(self, filter_on: bool = True, dijkstra: bool = False, max_iterations: int = 30):
        """Stabilize the pool, then certify with exact pricing of every task."""
        started = time.perf_counter()
        self._solve_stabilization(filter_on, dijkstra, max_iterations)
        filtered_time = time.perf_counter() - started
        certification = {'certified_no_new_columns': False, 'passes': [],
                         'pricing_calls': 0, 'new_columns_total': 0, 'time': 0.0}
        for pass_idx in range(max_iterations):
            before = sum(map(len, self.flows.values()))
            cert_start = time.perf_counter()
            self.update_duals()
            self.rerun_trips = set(self.tasks)
            old_k = self.k_alt
            self.k_alt = 1
            try:
                self.solve_pricing_problems(sorted(self.tasks, key=lambda t: t.task_id), dijkstra=dijkstra)
            finally:
                self.k_alt = old_k
            self.total_pricing_calls += len(self.tasks)
            after = sum(map(len, self.flows.values()))
            found = self.found_flows[-1]
            self.solve_rmp()
            elapsed = time.perf_counter() - cert_start
            certification['time'] += elapsed
            certification['pricing_calls'] += len(self.tasks)
            certification['new_columns_total'] += after - before
            certification['passes'].append({'pass': pass_idx + 1, 'columns_before': before,
                'columns_after': after, 'new_columns': after - before, 'negative_paths': found, 'time': elapsed})
            if found == 0:
                certification['certified_no_new_columns'] = True
                # Each task's unexamined reduced cost is >= -1e-6. Subtracting
                # this total tolerance gives a conservative bound for the full LP.
                self.certified_lower_bound = self.rmp.ObjVal - len(self.tasks) * 1e-6
                self.lower_bound.append(self.certified_lower_bound)
                break
            if after == before:
                raise RuntimeError('Negative reduced-cost duplicate in certification; no certificate issued')
        self.certification = certification
        self.root_solve_time = filtered_time
        self.solve_time = time.perf_counter() - started
        return certification

    def _solve_stabilization(self, filter_on: bool = True, dijkstra: bool = False, max_iterations: int = 30):
        """Main column generation loop.

        Reference: README Section 3.4 Algorithm 2 (OneStageColumnGeneration)

        Optimization: Added timing statistics for performance analysis.
        """
        import time
        t = time.time()

        # On very small pools, filter bookkeeping and extra stabilization
        # rounds cost more than the shortest-path calls they would remove.
        # Use the same full-pricing schedule as the no-filter variant there;
        # screening is enabled for the medium/large instances where it pays.
        self.filter_enabled = bool(filter_on and len(self.tasks) > 200)
        filter_on = self.filter_enabled

        # Timing statistics
        time_rmp = 0
        time_filter = 0
        time_pricing = 0
        time_update_duals = 0

        # Reset iteration history
        self.iteration_history = []

        iteration = 0
        first_full_iteration_done = False
        while True:
            iteration += 1
            if iteration > max_iterations:
                break
            iter_record = {'iteration': iteration, 'time_rmp': 0, 'time_filter': 0,
                          'time_pricing': 0, 'time_update_duals': 0, 'flows': 0,
                          'rmp_solves': 0, 'pricing_calls': 0,
                          'bounded_arcs': 0, 'active_tasks': 0,
                          'filtered_tasks': 0, 'filter_reduction_pct': 0.0,
                          'forced_full_refresh': False}
            rmp_solve_start = self.rmp_solve_count
            if iteration % 10 == 1:
                print(f'  CG iteration {iteration}...')

            # Solve RMP
            t_start = time.time()
            self.solve_rmp()
            time_rmp += time.time() - t_start
            iter_record['time_rmp'] = time.time() - t_start
            self.objVal.append(self.rmp.objVal)
            if iteration <= 3 or iteration % 10 == 0:
                print(f'    RMP solved, objVal={self.rmp.objVal:.2f}')

            if not filter_on:
                # Always run pricing for all tasks (not just when change=True)
                # This ensures shadow_cap updates are propagated to new columns
                # Update duals FIRST so pricing uses current (not stale) dual values
                t_duals = time.time()
                self.update_duals()
                time_update_duals += time.time() - t_duals
                iter_record['time_update_duals'] = time.time() - t_duals
                t_pp = time.time()
                self.solve_pricing_problems(self.tasks, dijkstra=dijkstra)
                time_pricing += time.time() - t_pp
                iter_record['time_pricing'] = time.time() - t_pp
                self.num_rerun.append(len(self.tasks))
                self.total_pricing_calls += len(self.tasks)
                iter_record['pricing_calls'] = len(self.tasks)
                iter_record['active_tasks'] = len(self.tasks)
                first_full_iteration_done = True

                total_flows = sum(len(f) for f in self.flows.values())
                iter_record['flows'] = total_flows
                iter_record['obj'] = self.rmp.objVal
                iter_record['rmp_solves'] = self.rmp_solve_count - rmp_solve_start
                self.iteration_history.append(iter_record)
                # Only break after obj stabilizes AND no new flows were added
                if len(self.objVal) >= 2 and abs(self.objVal[-1] - self.objVal[-2]) < 1e-6:
                    if not self.change:
                        total = sum(len(f) for f in self.flows.values())
                        print(f'    Converged: obj stable at {self.rmp.objVal:.2f}, {total} flows')
                        break

            else:
                # Run pricing filter
                t_f = time.time()
                # Update duals FIRST so pricing uses current (not stale) dual values
                t_duals = time.time()
                self.update_duals()
                time_update_duals += time.time() - t_duals
                iter_record['time_update_duals'] = time.time() - t_duals
                # A periodic full refresh keeps the filtered pool from
                # drifting too far from the complete pricing closure.  It
                # reduces the number of expensive post-stabilization
                # certification passes while retaining filtered rounds in
                # between refreshes.
                force_full_refresh = iteration > 1 and iteration % 10 == 0
                if force_full_refresh:
                    self.rerun_trips = set(self.tasks)
                    filter_stats = {
                        'bounded_arcs': sum(1 for beta in self.shadow_cap.values()
                                            if abs(beta) > 1e-6),
                        'active_tasks': len(self.tasks),
                        'filtered_tasks': 0,
                        'filter_reduction_pct': 0.0,
                        'forced_full_refresh': True,
                    }
                    self.filter_history.append(filter_stats.copy())
                else:
                    filter_stats = self.pricing_filter()
                    filter_stats['forced_full_refresh'] = False
                time_filter += time.time() - t_f
                iter_record['time_filter'] = time.time() - t_f
                iter_record.update(filter_stats)

                total_flows = sum(len(f) for f in self.flows.values())

                # If no tasks to rerun, check convergence
                if len(self.rerun_trips) == 0:
                    if total_flows > 0:
                        # Have flows, check if optimal
                        if max(self.lower_bound) == self.rmp.objVal:
                            print(f'    Optimal: LB={max(self.lower_bound):.2f}, obj={self.rmp.objVal:.2f}')
                            break
                        # Not optimal but no bounded arcs - converged
                        print(f'    No bounded arcs, converged at {self.rmp.objVal:.2f}')
                        break
                    elif not first_full_iteration_done:
                        # First iteration with no flows and no bounded arcs - do all tasks
                        self.rerun_trips = set(self.tasks)
                        print(f'    First iteration, solving all tasks')
                    else:
                        # Already did full iteration but still no flows - converged
                        print(f'    No flows found, converged at {self.rmp.objVal:.2f}')
                        break

                # Only track rerun count when we actually do pricing
                self.num_rerun.append(len(self.rerun_trips))
                t_pp = time.time()

                # Solve pricing problems
                self.solve_pricing_problems(sorted(self.rerun_trips, key=lambda t: t.task_id), dijkstra=dijkstra)
                time_pricing += time.time() - t_pp
                iter_record['time_pricing'] = time.time() - t_pp
                self.total_pricing_calls += len(self.rerun_trips)
                iter_record['pricing_calls'] = len(self.rerun_trips)
                first_full_iteration_done = True

                # The next loop iteration solves the RMP once with the new
                # columns.  Solving it here as well duplicates the RMP work
                # that the unfiltered variant does not perform.
                total_flows = sum(len(f) for f in self.flows.values())
                iter_record['flows'] = total_flows
                iter_record['obj'] = self.rmp.objVal
                iter_record['rmp_solves'] = self.rmp_solve_count - rmp_solve_start
                self.iteration_history.append(iter_record)

                if iteration <= 3 or iteration % 10 == 0:
                    print(f'    Pricing complete (iter {iteration}), objVal={self.rmp.objVal:.2f}, flows={total_flows}')

                if iteration >= max_iterations:
                    print(f'    Max iterations reached ({iteration})')
                    break

                # Check convergence: objective stable for 3 consecutive iterations
                # (ignoring found_flows since duplicate columns don't improve objective)
                if len(self.objVal) >= 4:
                    recent_change = max(abs(self.objVal[i] - self.objVal[i-1])
                                      for i in range(max(1, len(self.objVal)-3), len(self.objVal)))
                    if recent_change < 1e-4:
                        print(f'    Converged: obj stable at {self.rmp.objVal:.2f}')
                        break

        # Final solve to get correct objective with all flows
        self.solve_rmp()
        self.solve_time = time.time() - t

        # Store timing breakdown in instance variables
        total_time = max(time_rmp + time_filter + time_pricing + time_update_duals, 1e-12)
        self.timing_breakdown = {
            'rmp_solving': time_rmp,
            'pricing_filter': time_filter,
            'pricing_problems': time_pricing,
            'update_duals': time_update_duals,
            'total': total_time,
        }

        # Print timing statistics
        print(f'\n  Timing breakdown:')
        print(f'    RMP solving:        {time_rmp:.2f}s ({time_rmp/total_time*100:.1f}%)')
        print(f'    Pricing filter:    {time_filter:.2f}s ({time_filter/total_time*100:.1f}%)')
        print(f'    Pricing problems:   {time_pricing:.2f}s ({time_pricing/total_time*100:.1f}%)')
        print(f'    Update duals:      {time_update_duals:.2f}s ({time_update_duals/total_time*100:.1f}%)')

    def update_duals(self):
        """Update dual variables for pricing.

        Reference: intermodalTransportationNetworksCG update_duals

        Optimization: Cache dual values and only update when changed significantly.
        """
        self.change = False

        # Edge capacity duals (β_e)
        for key in self.cap_constr:
            new_val = self.cap_constr[key].Pi
            if abs(self.shadow_cap[key] - new_val) > 1e-6:
                self.change = True
            self.shadow_cap[key] = new_val

        # Convergence constraint duals (α_t)
        for task in self.tasks:
            if task in self.conv_constr:
                new_val = self.conv_constr[task].Pi
                # For >= 1 constraints, dual represents cost of forcing dummy >= 1
                if abs(self.shadow_conv[task] - new_val) > 1e-6:
                    self.change = True
                self.shadow_conv[task] = new_val

    def pricing_filter(self, epsilon: float = 1e-6):
        """Bounded-arc pricing filter.

        Implements the capacity-priced task-chain filter: only tasks whose
        positive-flow RMP chains use bounded arcs (|β_e| > ε) are re-priced.
        A task with no generated chain remains active.  The outer loop also
        performs periodic full refreshes and the final certification pass
        prices every task, so screening cannot change the certified result.

        Tasks that are unassigned (no current path) are always included since
        they may generate new columns.
        """
        # Identify bounded arcs: those with |β_e| > epsilon
        bounded_arcs = {
            arc for arc, beta in self.shadow_cap.items()
            if abs(beta) > epsilon
        }

        total_tasks = len(self.tasks)
        if not bounded_arcs:
            # No bounded arcs: all tasks use paths with slack, re-price all
            self.rerun_trips = set(self.tasks)
            stats = {
                'bounded_arcs': 0,
                'active_tasks': total_tasks,
                'filtered_tasks': 0,
                'filter_reduction_pct': 0.0,
            }
            self.filter_history.append(stats.copy())
            return stats

        # Find tasks whose *current RMP solution* uses at least one bounded
        # arc, plus tasks without a generated chain.  Historical columns that
        # currently carry zero flow are deliberately excluded: retaining them
        # made the active set approach all tasks as the pool grew, defeating
        # the purpose of the filter and repeatedly pricing duplicate routes.
        active_tasks = set()
        for task in self.tasks:
            task_flows = self.flows[task]
            if not task_flows:
                # Unassigned task - always include.
                active_tasks.add(task)
                continue
            for flow in task_flows:
                var = self.x.get(flow)
                if var is None or var.X <= 1e-8:
                    continue
                if any(arc in bounded_arcs for arc in flow.arcs):
                    active_tasks.add(task)
                    break

        self.rerun_trips = active_tasks
        stats = {
            'bounded_arcs': len(bounded_arcs),
            'active_tasks': len(active_tasks),
            'filtered_tasks': total_tasks - len(active_tasks),
            'filter_reduction_pct': 100.0 * (total_tasks - len(active_tasks)) / total_tasks
            if total_tasks else 0.0,
        }
        self.filter_history.append(stats.copy())
        return stats

    def pricing_filter_no_dual_update(self):
        """Pricing filter without dual update - simplified for correctness.

        To ensure correctness, we re-price all tasks when any constraint is binding.
        This is slower but ensures no columns are missed due to incomplete filtering.
        """
        # Re-price all tasks to ensure correctness
        # Optimization can be added later once correctness is verified
        self.rerun_trips = set(self.tasks)

    def get_dual_variables(self) -> Dict:
        """Get all dual variables for analysis."""
        return {
            'conv': self.shadow_conv.copy(),
            'cap': self.shadow_cap.copy()
        }

    def _get_flow_arc_index(self):
        """Build/cache flow-arc index for faster pricing filter.

        Returns a dict mapping arc -> list of tasks that use that arc.
        This avoids O(tasks * flows * arcs) intersection checks.
        """
        if self._flow_arc_index is not None:
            return self._flow_arc_index

        self._flow_arc_index = {}
        for task, task_flows in self.flows.items():
            for flow in task_flows:
                for arc in flow.arcs:
                    if arc not in self._flow_arc_index:
                        self._flow_arc_index[arc] = set()
                    self._flow_arc_index[arc].add(task)
        return self._flow_arc_index


def solve_last_iter_int(lp: ContinuousRelaxation) -> Tuple[ContinuousRelaxation, float, int]:
    """Solve last iteration as MIP for integer solution.

    Reference: intermodalTransportationNetworksCG solve_last_iter_int
    Simple approach: set variables to BINARY and re-solve with relaxed MIP gap.

    Args:
        lp: ContinuousRelaxation solver after column generation

    Returns:
        Tuple of (Updated solver, integer objective, nodes explored)
    """
    frac_flows = find_frac_flows(lp, epsilon=1e-6)
    if len(frac_flows) == 0:
        print(f"  Solution is already integer ({len(lp.flows)} flows)")

        # Compute TRUE integer objective cost
        int_obj = compute_true_objective(lp)

        return lp, int_obj, 0

    print(f"  Fractional flows detected: {len(frac_flows)}")
    print(f"  Setting flow variables to BINARY and re-solving...")

    # Set all flow variables to BINARY (reference project approach)
    for flow, v in lp.x.items():
        v.setAttr('vtype', GRB.BINARY)

    for var in lp.dummy_vars.values():
        var.VType = GRB.BINARY

    # Use dynamic MIP gap like reference project: 0.5 / objVal
    mipgap = 0.5 / lp.rmp.ObjVal if lp.rmp.ObjVal > 0 else 0.005
    lp.rmp.setParam('MIPGap', mipgap)
    lp.rmp.setParam('OutputFlag', 1)
    lp.rmp.setParam('TimeLimit', 300.0)

    # Reset and re-solve as MIP
    lp.rmp.reset()
    lp.rmp.optimize()
    if lp.rmp.SolCount == 0:
        raise RuntimeError('Integer recovery found no feasible incumbent')

    # Compute true integer objective
    int_obj = compute_true_objective(lp)

    print(f"  Integer solution found: obj={int_obj:.2f}")

    return lp, int_obj, int(lp.rmp.NodeCount)


def compute_true_objective(lp: ContinuousRelaxation) -> float:
    """Compute the true objective value from the CG solution.

    The LP objective (lp.rmp.objVal) uses reduced costs and may not equal
    the true sum of path costs + dummy penalties. This function computes
    the actual cost based on which flows are selected (var.X > 0.5) and
    which tasks are rejected (dummy.X > 0.5).

    Args:
        lp: ContinuousRelaxation solver with solved RMP

    Returns:
        True objective cost (float)
    """
    total_cost = 0.0

    # Sum up path costs for tasks with active flows
    completed_tasks = 0
    rejected_tasks = 0

    for flow, var in lp.x.items():
        if var.X > 0.5:  # Task is completed using this flow
            # flow.cost is the actual path cost
            total_cost += flow.cost
            completed_tasks += 1

    # Sum up dummy costs for rejected tasks (scaled by data_volume)
    for task, var in lp.dummy_vars.items():
        if var.X > 0.5:  # Task is rejected
            total_cost += lp.dummy_cost
            rejected_tasks += 1

    return total_cost


def find_frac_flows(lp: ContinuousRelaxation, epsilon: float = 1e-6) -> List:
    """Find fractional flows in current solution."""
    frac_flows = []
    for flow, var in lp.x.items():
        val = var.X
        if epsilon < val < 1 - epsilon:
            frac_flows.append((flow, val))
    return frac_flows


if __name__ == '__main__':
    # Test column generation solver with Gurobi
    print("Testing column generation solver with Gurobi...")

    from Generator.instance_generator import Instance, TimeWindow, TLEParser, DEFAULT_GS_CONFIG
    from Generator.satellite_generator import build_graph

    parser = TLEParser()
    sats = parser._get_sample_satellites()

    time_window = TimeWindow(start=0.0, end=60.0)

    instance = Instance(
        time_window=time_window,
        rs_count=5,
        cs_count=15,
        gs_config=DEFAULT_GS_CONFIG,
        tle_data=sats[:5],
        subset=0.1
    )

    instance.build_instance(seed=42)

    G = build_graph(instance)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    cr = ContinuousRelaxation(
        graph=G,
        tasks=instance.tasks,
        sources=instance.sources,
        sinks=instance.sinks,
        dummy_cost=1000000,
        heuristic=instance.heuristic,
        sat_params=instance.sat_params
    )

    cr.build_restricted_masterproblem()
    print("RMP built successfully with Gurobi")
    print(f"  Tasks: {len(cr.tasks)}")
    print(f"  Capacity constraints: {len(cr.cap_constr)}")
    print(f"  Note: All constraints handled via edge capacities and convergence constraints")
