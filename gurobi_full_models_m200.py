"""Independent full Gurobi models, sharing no P&B columns or incumbents.

The commodity model retains every potentially useful arc, using only the
exact individual-cost dominance rule d_s(u)+c_uv+d_sink(v) <= M. The aggregate
model is exact for unit-volume, task-independent-cost, common-sink instances.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time

import gurobipy as gp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from alns_validation import audit_schedule
from Solver.column_generation import CAPACITY_ARC_TYPES


def graph_arrays(instance, graph):
    nodes = list(graph)
    index = {node: i for i, node in enumerate(nodes)}
    edges = list(graph.edges())
    tails = np.fromiter((index[u] for u, _ in edges), dtype=np.int32)
    heads = np.fromiter((index[v] for _, v in edges), dtype=np.int32)
    costs = np.fromiter((float(graph.edges[e]['weight']) for e in edges), dtype=float)
    caps = np.fromiter((float(graph.edges[e]['cap']) for e in edges), dtype=float)
    resource = np.fromiter((graph.edges[e]['arc_type'] in CAPACITY_ARC_TYPES for e in edges), dtype=bool)
    imaging = np.fromiter((graph.edges[e]['arc_type'] == 'imaging_convergence_arc' for e in edges), dtype=bool)
    tasks = sorted(instance.tasks, key=lambda t: t.task_id)
    def key(node):
        return node if isinstance(node, str) else f'{node.satellite}_{node.time_slot}_{node.node_type}_{node.route}'
    sources = [index[key(instance.sources[t.task_id])] for t in tasks]
    sinks = [index[key(instance.sinks[t.task_id])] for t in tasks]
    if len(set(sources)) != len(tasks) or len(set(sinks)) != 1:
        raise ValueError('Expected unique task sources and one shared sink')
    if np.any(costs < 0) or not np.all(np.isfinite(costs)):
        raise ValueError('Cost dominance requires finite nonnegative base costs')
    return dict(nodes=nodes, edges=edges, index=index, tails=tails, heads=heads,
                costs=costs, caps=caps, resource=resource, imaging=imaging,
                tasks=tasks, sources=sources, sink=sinks[0])


def configure(model, time_limit, threads, soft_mem_gb, log_file):
    model.Params.OutputFlag = 1 if log_file else 0
    if log_file:
        model.Params.LogFile = str(log_file)
        model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.SoftMemLimit = soft_mem_gb
    model.Params.MIPGap = 0
    model.Params.MIPGapAbs = 0
    model.Params.NumericFocus = 1


def path_in_active_flow(start, sink, adjacency, residual):
    """Find one simple sink path; zero-cost residual cycles do not obstruct it."""
    stack = [(start, iter(adjacency.get(start, ())))]
    seen, parent = {start}, {}
    while stack:
        u, successors = stack[-1]
        if u == sink:
            path = [sink]
            arc_ids = []
            while path[-1] != start:
                prev, edge = parent[path[-1]]
                arc_ids.append(edge)
                path.append(prev)
            return path[::-1], arc_ids[::-1]
        try:
            v, edge = next(successors)
        except StopIteration:
            stack.pop()
            continue
        if residual[edge] <= 0 or v in seen:
            continue
        seen.add(v)
        parent[v] = u, edge
        stack.append((v, iter(adjacency.get(v, ()))))
    raise RuntimeError('Positive task flow cannot be decomposed to the common sink')


def finish_result(instance, graph, a, model, start, cpu_start, build_time,
                  optimize_time, extraction_time, selected, rejected, extra):
    status_names = {gp.GRB.OPTIMAL: 'OPTIMAL', gp.GRB.TIME_LIMIT: 'TIME_LIMIT',
                    gp.GRB.MEM_LIMIT: 'MEM_LIMIT', gp.GRB.INTERRUPTED: 'INTERRUPTED'}
    solved = model.SolCount > 0
    bound = float(model.ObjBound) if model.IsMIP else (float(model.ObjVal) if model.Status == gp.GRB.OPTIMAL else None)
    result = {'obj': float(model.ObjVal) if solved else None,
              'lower_bound': bound if bound is not None and math.isfinite(bound) else None,
              'raw_bound_state': 'finite' if bound is not None and math.isfinite(bound) else str(bound),
              'lp_objective_at_optimum':
                             (float(model.ObjVal) if model.Status == gp.GRB.OPTIMAL else None),
              'status': status_names.get(model.Status, f'STATUS_{model.Status}'),
              'status_code': int(model.Status), 'solution_count': int(model.SolCount),
              'num_variables': int(model.NumVars), 'num_constraints': int(model.NumConstrs),
              'num_nonzeros': int(model.NumNZs), 'gurobi_runtime': float(model.Runtime),
              'model_build_time': build_time, 'optimize_wall_time': optimize_time,
              'extraction_time': extraction_time, 'solve_time': time.perf_counter()-start,
              'process_cpu_time': time.process_time()-cpu_start,
              'schedule': selected, 'rejected_tasks': rejected,
              'completed': len(selected) if solved else None,
              'rejected': len(rejected) if solved else None,
              'node_count': float(model.NodeCount) if model.IsMIP else 0,
              'memory_gb': float(model.MemUsed), 'max_memory_gb': float(model.MaxMemUsed),
              **extra}
    if result['obj'] is not None and result['lower_bound'] is not None:
        result['gap_pct'] = 100 * (result['obj']-result['lower_bound'])/max(abs(result['obj']),1e-12)
    else:
        result['gap_pct'] = None
    if solved:
        result['audit'] = audit_schedule(instance, graph, result, extra['penalty'])
        if not result['audit']['feasible']:
            raise RuntimeError(f'Independent full-model schedule audit failed: {result["audit"]}')
    return result


def run_aggregate(instance, graph, penalty=200, time_limit=600, threads=8,
                  soft_mem_gb=6, log_file=None):
    start, cpu_start = time.perf_counter(), time.process_time()
    a = graph_arrays(instance, graph)
    if any(abs(t.data_volume-1.0) > 1e-12 for t in a['tasks']):
        raise ValueError('Aggregate reduction applies only to unit-volume tasks')
    if any(graph.in_degree(a['nodes'][s]) for s in a['sources']):
        raise ValueError('Aggregate reduction requires task sources with no incoming arcs')
    n, m, nt = len(a['nodes']), len(a['edges']), len(a['tasks'])
    tails = np.concatenate([a['tails'], a['sources']])
    heads = np.concatenate([a['heads'], np.full(nt, a['sink'], dtype=np.int32)])
    cols = np.tile(np.arange(m+nt, dtype=np.int32), 2)
    incidence = coo_matrix((np.concatenate([np.ones(m+nt), -np.ones(m+nt)]),
                           (np.concatenate([tails,heads]), cols)), shape=(n,m+nt)).tocsr()
    rhs = np.zeros(n)
    rhs[a['sources']] = 1
    rhs[a['sink']] = -nt
    # Integer loads imply floor(cap); unconstrained arcs need at most nt units
    # after removal of nonnegative-cost cycles. This does not restrict optimality.
    physical_ub = np.where(a['resource'], np.minimum(nt,np.floor(a['caps']+1e-9)),nt)
    upper = np.concatenate([physical_ub,np.ones(nt)])
    objective = np.concatenate([a['costs'],np.full(nt,penalty)])
    model = gp.Model('full_aggregate_unit_flow')
    try:
        configure(model,time_limit,threads,soft_mem_gb,log_file)
        x = model.addMVar(m+nt,lb=0,ub=upper,obj=objective)
        model.addMConstr(incidence,x,'=',rhs)
        model.update()
        build_time = time.perf_counter()-start
        optimize_start = time.perf_counter()
        model.optimize()
        optimize_time = time.perf_counter()-optimize_start
        extraction_start = time.perf_counter()
        selected, rejected = [], []
        fractional_residual = None
        if model.SolCount:
            values = x.X
            fractional_residual = float(np.max(np.abs(values-np.rint(values))))
            if fractional_residual > 1e-5:
                raise RuntimeError('Network LP did not return an integral basic solution')
            residual = np.rint(values[:m]).astype(np.int32)
            adjacency = defaultdict(list)
            for edge in np.flatnonzero(residual):
                adjacency[int(a['tails'][edge])].append((int(a['heads'][edge]),int(edge)))
            for i, task in enumerate(a['tasks']):
                if values[m+i] > .5:
                    rejected.append(task.task_id)
                else:
                    nodes, edges = path_in_active_flow(a['sources'][i],a['sink'],adjacency,residual)
                    residual[edges] -= 1
                    selected.append({'task':task.task_id,'nodes':[a['nodes'][v] for v in nodes]})
            if float(np.dot(residual,a['costs'])) > 1e-6:
                raise RuntimeError('Positive-cost residual circulation in aggregate optimum')
        extraction_time = time.perf_counter()-extraction_start
        return finish_result(instance,graph,a,model,start,cpu_start,build_time,
                             optimize_time,extraction_time,selected,rejected,
                             {'formulation':'exact aggregate unit-volume flow LP','penalty':penalty,
                              'fractional_residual':fractional_residual,
                              'fractional_resource_capacity_count':int(np.sum(a['resource'] & (np.abs(a['caps']-np.rint(a['caps']))>1e-9))),
                              'task_costs_shared':True,'common_sink':True})
    finally:
        model.dispose()


def commodity_profile(instance,graph,penalty=200,progress=None):
    start = time.perf_counter()
    a = graph_arrays(instance,graph)
    n = len(a['nodes'])
    weighted = coo_matrix((a['costs'],(a['tails'],a['heads'])),shape=(n,n)).tocsr()
    suffix = dijkstra(weighted.T.tocsr(),directed=True,indices=a['sink'])
    blocks, count = [], 0
    for i, (task,source) in enumerate(zip(a['tasks'],a['sources'])):
        prefix = dijkstra(weighted,directed=True,indices=source,limit=penalty)
        coefficients = np.where(a['imaging'],1.0,task.data_volume)
        eligible = (~a['resource']) | (a['caps']+1e-9 >= coefficients)
        with np.errstate(invalid='ignore'):
            keep = eligible & (prefix[a['tails']]+a['costs']+suffix[a['heads']] <= penalty+1e-9)
        edges = np.flatnonzero(keep).astype(np.int32)
        nodes = np.unique(np.concatenate([a['tails'][edges],a['heads'][edges],
                                         np.array([source,a['sink']],dtype=np.int32)]))
        blocks.append({'task':task,'source':source,'edges':edges,'nodes':nodes})
        count += len(edges)
        if progress and (i%100 == 0 or i+1 == len(a['tasks'])):
            progress({'tasks_profiled':i+1,'arc_variables':count,'elapsed':time.perf_counter()-start})
    profile = {'profile_time':time.perf_counter()-start,'tasks':len(blocks),
               'dense_arc_variables':len(a['tasks'])*len(a['edges']),
               'retained_arc_variables':count,'rejection_variables':len(blocks),
               'flow_rows':sum(len(b['nodes']) for b in blocks),
               'per_task_arc_counts':[len(b['edges']) for b in blocks],
               'pruning':'retain every arc with d_source(u)+base_cost(u,v)+d_sink(v)<=M and sufficient individual capacity; all paths costing <=M are retained; no candidate-path pool or P&B duals used'}
    return a,blocks,profile


def run_commodity(instance,graph,penalty=200,time_limit=600,threads=8,
                  soft_mem_gb=6,log_file=None,reference_obj=None,progress=None,
                  max_arc_variables=5_000_000):
    start,cpu_start = time.perf_counter(),time.process_time()
    a,blocks,profile = commodity_profile(instance,graph,penalty,progress)
    if profile['retained_arc_variables'] > max_arc_variables:
        return {'formulation':'full commodity binary arc MIP with exact dominance pruning',
                'penalty':penalty,'profile':profile,'obj':None,'lower_bound':None,'gap_pct':None,
                'status':'NOT_BUILT_RESOURCE_BUDGET','num_variables_planned':profile['retained_arc_variables']+len(blocks),
                'solve_time':None,'profile_only_time':time.perf_counter()-start,
                'process_cpu_time_profile':time.process_time()-cpu_start,
                'max_arc_variables':max_arc_variables,
                'reason':'prespecified variable budget based on pilot memory consumption; no optimize run, no timeout or solver incapability claim',
                'schedule':[],'rejected_tasks':[],'completed':None,'rejected':None}
    nt = len(blocks)
    arc_count = profile['retained_arc_variables']
    resource_edges = np.unique(np.concatenate([b['edges'][a['resource'][b['edges']]] for b in blocks]))
    resource_lookup = np.full(len(a['edges']),-1,dtype=np.int32)
    resource_lookup[resource_edges] = np.arange(len(resource_edges),dtype=np.int32)
    nr_flow = profile['flow_rows']
    nr,nv = nr_flow+len(resource_edges),arc_count+nt
    row_arrays,col_arrays,value_arrays,obj_arrays = [],[],[],[]
    rhs = np.zeros(nr)
    offset,row_offset = 0,0
    for i,b in enumerate(blocks):
        edges,nodes = b['edges'],b['nodes']
        b['offset'] = offset
        columns = np.arange(offset,offset+len(edges),dtype=np.int32)
        out_rows = row_offset+np.searchsorted(nodes,a['tails'][edges])
        in_rows = row_offset+np.searchsorted(nodes,a['heads'][edges])
        source_row = row_offset+int(np.searchsorted(nodes,b['source']))
        sink_row = row_offset+int(np.searchsorted(nodes,a['sink']))
        rhs[source_row],rhs[sink_row] = 1,-1
        row_arrays.extend([out_rows,in_rows,np.array([source_row,sink_row],dtype=np.int32)])
        col_arrays.extend([columns,columns,np.array([arc_count+i,arc_count+i],dtype=np.int32)])
        value_arrays.extend([np.ones(len(edges)),-np.ones(len(edges)),np.array([1.,-1.])])
        is_resource = a['resource'][edges]
        resource_cols = columns[is_resource]
        resource_ids = edges[is_resource]
        row_arrays.append(nr_flow+resource_lookup[resource_ids])
        col_arrays.append(resource_cols)
        value_arrays.append(np.where(a['imaging'][resource_ids],1.,b['task'].data_volume))
        obj_arrays.append(a['costs'][edges])
        offset += len(edges)
        row_offset += len(nodes)
    rhs[nr_flow:] = a['caps'][resource_edges]
    objective = np.concatenate(obj_arrays+[np.full(nt,penalty)])
    matrix = coo_matrix((np.concatenate(value_arrays),
                         (np.concatenate(row_arrays),np.concatenate(col_arrays))),shape=(nr,nv)).tocsr()
    del row_arrays,col_arrays,value_arrays,obj_arrays
    sense = np.full(nr,'=',dtype='U1')
    sense[nr_flow:] = '<'
    model = gp.Model('full_commodity_arc_mip')
    history,first_reference = [],[None]
    try:
        configure(model,time_limit,threads,soft_mem_gb,log_file)
        x = model.addMVar(nv,lb=0,ub=1,obj=objective,vtype=gp.GRB.BINARY)
        model.addMConstr(matrix,x,sense,rhs)
        model.update()
        build_time = time.perf_counter()-start
        if progress:
            progress({'phase':'model_built','variables':nv,'constraints':nr,
                      'nonzeros':int(model.NumNZs),'elapsed':build_time})
        def callback(m,where):
            if where == gp.GRB.Callback.MIPSOL:
                incumbent = float(m.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
                elapsed = time.perf_counter()-start
                bound = float(m.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                history.append({'elapsed_including_build':elapsed,'obj':incumbent,
                                'bound':bound if math.isfinite(bound) else None})
                if reference_obj is not None and incumbent <= reference_obj+1e-5 and first_reference[0] is None:
                    first_reference[0] = elapsed
        optimize_start = time.perf_counter()
        model.optimize(callback)
        optimize_time = time.perf_counter()-optimize_start
        extraction_start = time.perf_counter()
        selected,rejected = [],[]
        if model.SolCount:
            values = x.X
            for i,b in enumerate(blocks):
                if values[arc_count+i] > .5:
                    rejected.append(b['task'].task_id)
                    continue
                local_values = values[b['offset']:b['offset']+len(b['edges'])]
                residual = np.rint(local_values).astype(np.int8)
                adjacency = defaultdict(list)
                for local_edge in np.flatnonzero(residual):
                    edge = b['edges'][local_edge]
                    adjacency[int(a['tails'][edge])].append((int(a['heads'][edge]),int(local_edge)))
                nodes,local_edges = path_in_active_flow(b['source'],a['sink'],adjacency,residual)
                residual[local_edges] -= 1
                if float(np.dot(residual,a['costs'][b['edges']])) > 1e-6:
                    raise RuntimeError('Positive-cost commodity residual circulation')
                selected.append({'task':b['task'].task_id,'nodes':[a['nodes'][v] for v in nodes]})
        extraction_time = time.perf_counter()-extraction_start
        return finish_result(instance,graph,a,model,start,cpu_start,build_time,
                             optimize_time,extraction_time,selected,rejected,
                             {'formulation':'full commodity binary arc MIP with exact dominance pruning',
                              'penalty':penalty,'profile':profile,'incumbent_history':history,
                              'time_first_reaching_pb_obj_including_build':first_reference[0],
                              'warm_start':False})
    finally:
        model.dispose()
