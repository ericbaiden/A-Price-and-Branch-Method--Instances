"""Fresh matched P&B pricing ablation with complete wall-clock timing.

Solver and generator files remain frozen to the ALNS campaign. Outputs are
data only: paper figures and sources are never overwritten by this driver.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import run_alns_benchmark as bench
from Solver.column_generation import (ContinuousRelaxation, find_frac_flows,
                                      solve_last_iter_int, compute_true_objective)

ROOT = Path(__file__).resolve().parent
VARIANTS = ('A*+filter', 'A* no filter', 'Dijkstra+filter')


def run_variant(instance, graph, variant):
    start = time.perf_counter()
    cr = ContinuousRelaxation(graph, instance.tasks, instance.sources, instance.sinks,
                              200, instance.heuristic, instance.sat_params, k_alt=1)
    preprocessing = time.perf_counter() - start
    if cr.native_pricing is None:
        raise RuntimeError('Frozen native backend is unavailable')
    phase_start = time.perf_counter()
    cr.build_restricted_masterproblem()
    master_build = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    use_dijkstra = variant == 'Dijkstra+filter'
    cert = cr.solve(filter_on=variant != 'A* no filter', dijkstra=use_dijkstra,
                    max_iterations=200)
    cg_time = time.perf_counter() - phase_start
    if not cert['certified_no_new_columns']:
        raise RuntimeError('Column generation did not certify')
    # Match the ALNS campaign's P&B runner, which retains a final independent
    # all-task confirmation after ContinuousRelaxation.solve's own certificate.
    phase_start = time.perf_counter()
    passes = []
    for pass_id in range(1, 201):
        cr.solve_rmp()
        cr.update_duals()
        tasks = sorted(instance.tasks, key=lambda t: t.task_id)
        cr.native_pricing.prepare(cr.shadow_cap, [t.data_volume for t in tasks])
        flows = []
        for task in tasks:
            flow = cr.pricing_solver(task, dijkstra=use_dijkstra)
            if flow is not None:
                flows.append(flow)
        before = len(cr.x)
        passes.append({'pass': pass_id, 'negative_paths': len(flows), 'columns_before': before})
        if not flows:
            break
        cr.add_new_flows(flows)
        passes[-1]['columns_after'] = len(cr.x)
        if len(cr.x) == before:
            raise RuntimeError('Only duplicate improving columns found')
    else:
        raise RuntimeError('Final confirmation limit reached')
    confirmation_time = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    lp_objective = float(cr.rmp.ObjVal)
    lower_bound = lp_objective - len(instance.tasks) * 1e-6
    fractional = len(find_frac_flows(cr))
    if fractional:
        cr, objective, _ = solve_last_iter_int(cr)
    else:
        objective = compute_true_objective(cr)
    selected = [{'task': flow.trip.task_id, 'nodes': list(flow.path[1])}
                for flow, var in cr.x.items() if var.X > 0.5]
    rejected = [task.task_id for task, var in cr.dummy_vars.items() if var.X > 0.5]
    integer_time = time.perf_counter() - phase_start
    solve_time = time.perf_counter() - start
    confirmation_calls = len(passes) * len(instance.tasks)
    phases = {'preprocessing': preprocessing, 'master_build': master_build,
              'cg_with_internal_certification': cg_time,
              'final_full_confirmation': confirmation_time,
              'integer_recovery_and_extraction': integer_time}
    result = {'variant': variant, 'obj': float(objective), 'lp_obj': lp_objective,
              'lower_bound': lower_bound, 'solve_time': solve_time, 'time_phases': phases,
              'phase_accounting_residual': solve_time - sum(phases.values()),
              'completed': len(selected), 'rejected': len(rejected),
              'schedule': selected, 'rejected_tasks': rejected,
              'fractional_flows': fractional, 'num_columns': len(cr.x),
              'capacity_rows': len(cr.cap_constr), 'master_constraints': cr.rmp.NumConstrs,
              'iterations': cr.num_it, 'stabilization_time': cr.root_solve_time,
              'rmp_solve_count': cr.rmp_solve_count,
              'filter_enabled': getattr(cr, 'filter_enabled', variant != 'A* no filter'),
              'filter_history': cr.filter_history,
              'internal_certification': cert, 'final_confirmation_passes': passes,
              'certified_no_negative_columns': True,
              'pricing_calls_stabilization': cr.total_pricing_calls - cert['pricing_calls'],
              'pricing_calls_internal_certification': cert['pricing_calls'],
              'pricing_calls_final_confirmation': confirmation_calls,
              'total_pricing_calls': cr.total_pricing_calls + confirmation_calls,
              'iteration_history': cr.iteration_history,
              'gap_pct': 100 * (objective - lower_bound) / lower_bound,
              'backend': 'native CSR shortest_path.dll'}
    cr.rmp.dispose()
    audit_start = time.perf_counter()
    result['audit'] = bench.audit_schedule(instance, graph, result, 200)
    result['validation_time'] = time.perf_counter() - audit_start
    if not result['audit']['feasible']:
        raise RuntimeError(f'Independent schedule audit failed: {result["audit"]}')
    if abs(result['phase_accounting_residual']) > 0.01:
        raise RuntimeError('Timing phases do not account for total solve time')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pb_ablation_m200_20261014')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'results/alns_m200/protocol.json').read_text(encoding='utf-8'))
    for name, expected in prior['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Solver/generator changed since ALNS campaign: {name}')
    protocol = {'version': 1, 'penalty': 200, 'task_counts': [200,500,1000],
                'instance_seeds': list(range(10)), 'variants': list(VARIANTS), 'k_alt': 1,
                'max_iterations': 200, 'variant_order': 'rotate by seed modulo three',
                'source_fingerprint': prior['source_fingerprint'],
                'source_sha256': prior['source_sha256'],
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'environment': {'python': sys.version, 'platform': platform.platform(),
                                'cpu_model': prior['environment']['cpu_model'],
                                'gurobi': list(bench.gp.gurobi.version()), 'gurobi_threads': 8,
                                'execution': 'sequential; fresh master for every variant'},
                'timing': 'perf_counter wall clock from solver initialization through schedule extraction; includes master construction, CG, internal full-pricing certification, final full-pricing confirmation, and integer recovery; excludes instance/graph construction, disposal, independent audit, and output I/O',
                'graph_construction': 'recorded separately; physical contacts cached after first case; each task graph regenerated and canonically ordered',
                'pricing_calls': 'all calls including internal certification and final confirmation',
                'comparison': 'at 500 and 1000 tasks, instance hashes and P&B objectives must match the prior ALNS campaign'}
    frozen = out / 'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Frozen protocol changed; select another output directory')
    if not frozen.exists():
        bench.write_json(frozen, protocol)
    for count in protocol['task_counts']:
        for seed in protocol['instance_seeds']:
            path = out / f'tasks_{count:04d}_seed_{seed:02d}.json'
            if path.exists():
                cached = json.loads(path.read_text(encoding='utf-8'))
                if cached.get('status') == 'complete' and cached.get('driver_sha256') == protocol['driver_sha256']:
                    print(f'CACHED tasks={count} seed={seed}', flush=True)
                    continue
                raise RuntimeError(f'Incomplete or inconsistent case requires investigation: {path}')
            print(f'START tasks={count} seed={seed}', flush=True)
            case_start = time.perf_counter()
            physical_cached = bool(bench.PHYSICAL_CACHE)
            instance = bench.build_instance(seed, count)
            graph = bench.canonical_graph(instance)
            build_time = time.perf_counter() - case_start
            instance_digest = bench.instance_hash(instance, graph)
            previous = None
            if count in (500,1000):
                previous = json.loads((ROOT / f'results/alns_m200/tasks_{count:04d}_seed_{seed:02d}.json').read_text(encoding='utf-8'))
                if instance_digest != previous['instance_sha256']:
                    raise RuntimeError('Instance mismatch with ALNS benchmark')
            offset = seed % 3
            order = VARIANTS[offset:] + VARIANTS[:offset]
            record = {'status': 'running', 'started_at': datetime.now().isoformat(),
                      'tasks': count, 'seed': seed, 'penalty': 200,
                      'source_fingerprint': prior['source_fingerprint'],
                      'driver_sha256': protocol['driver_sha256'],
                      'instance_sha256': instance_digest,
                      'graph_nodes': len(graph), 'graph_edges': graph.number_of_edges(),
                      'graph_build_time': build_time, 'physical_contacts_cached': physical_cached,
                      'execution_order': list(order), 'algorithms': {}}
            bench.write_json(path, record)
            for variant in order:
                result = run_variant(instance, graph, variant)
                if previous is not None and variant == 'A*+filter':
                    if abs(result['obj'] - previous['algorithms']['P&B']['obj']) > 1e-5:
                        raise RuntimeError('P&B objective differs from matched ALNS campaign')
                record['algorithms'][variant] = result
                bench.write_json(path, record)
                print(f'  {variant}: obj={result["obj"]:.1f} solve={result["solve_time"]:.3f}s '
                      f'calls={result["total_pricing_calls"]} audit=OK', flush=True)
                gc.collect()
            values = list(record['algorithms'].values())
            if max(v['lp_obj'] for v in values) - min(v['lp_obj'] for v in values) > 1e-5:
                raise RuntimeError('Certified LP optima differ between variants')
            record.update(status='complete', finished_at=datetime.now().isoformat(),
                          case_total_time=time.perf_counter() - case_start)
            bench.write_json(path, record)
            print(f'DONE tasks={count} seed={seed} build={build_time:.3f}s', flush=True)
            graph.clear()
            del graph, instance, record, values, result
            gc.collect()
    print('ALL 30 CASES / 90 SOLVES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
