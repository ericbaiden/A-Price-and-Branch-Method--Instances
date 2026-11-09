"""Heterogeneous-volume benchmark at the service-first penalty (M=600).

Runs the matched P&B / 20-start GRASP / 60-second ALNS comparison on instances
whose per-task data volumes are drawn from the discrete set {1, 2, 3} GB instead
of the unit volume used by every historical record. With heterogeneous volumes
the arc-indexed model is a genuine unsplittable multicommodity flow, which is the
regime the path formulation is meant to address.

This script is additive: it writes to its own output directory, imports the
frozen benchmark helpers without modifying them, and never writes outside
--output. The default generator path (task_data_volumes=None) is untouched and
reproduces the frozen graph hash b207ec99..., as verified by
verify_volume_option.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if 'GRB_LICENSE_FILE' not in os.environ:
    for license_location in ('D:/software/win64/bin/gurobi.lic',
                             'D:/software/Gurobi/win64/bin/gurobi.lic'):
        if Path(license_location).is_file():
            os.environ['GRB_LICENSE_FILE'] = license_location
            break

import psutil

from alns_validation import audit_schedule
from run_alns_benchmark import (PHYSICAL_CACHE, build_instance, canonical_graph,
                                instance_hash, write_json)
from Solver.alns import run_alns, run_randomized_construction

PENALTY = 600


def discrete_volumes(seed, count, choices=(1.0, 2.0, 3.0)):
    """Per-task data volumes in GB, assigned in the order tasks are generated.

    A dedicated random stream is used so that the volume draw never perturbs the
    task geometry, targets or imaging windows that the frozen generator produces
    for a given seed.
    """
    rng = random.Random(100000 + seed)
    return [rng.choice(choices) for _ in range(count)]


def unit_volumes(seed, count, choices=(1.0,)):
    """The homogeneous convention: every task carries one unit of demand.

    Routed through the same override path as the heterogeneous draw so that the
    two regimes differ only in the volume vector and not in how the task layer is
    built. The resulting graph must still match the frozen hash for these seeds.
    """
    return [1.0] * count


VOLUME_MODES = {'discrete': discrete_volumes, 'unit': unit_volumes}


def volume_rule(mode, seed, count):
    if mode == 'unit':
        return f'q = 1.0 GB for all {count} tasks (homogeneous control)'
    return (f'random.Random({100000}+seed), generation order, '
            f'q in {{1,2,3}} GB, first {count} draws applied')


def run_one(seed, count, out_dir, mode='discrete', alns_time_limit=60.0,
            stall_iterations=None):
    start = time.perf_counter()
    tag = '' if mode == 'discrete' else f'_{mode}'
    path = out_dir / f'tasks_{count:05d}_seed_{seed:02d}{tag}.json'
    vols = VOLUME_MODES[mode](seed, count)
    build_start = time.perf_counter()
    instance = build_instance(seed, count, task_data_volumes=vols)
    graph = canonical_graph(instance)
    build_time = time.perf_counter() - build_start

    applied = [t.data_volume for t in instance.tasks]
    histogram = {str(v): applied.count(v) for v in sorted(set(applied))}
    digest = instance_hash(instance, graph)
    # The homogeneous control is routed through the volume-override path rather
    # than the default one, so assert that it still reproduces the frozen graph
    # of the published unit-volume record. This is what licenses reading the two
    # regimes as one matched comparison.
    uniform = all(v == applied[0] for v in applied) if applied else True
    if mode == 'unit' and count == 100 and seed == 0 and uniform:
        frozen = json.loads(
            (ROOT / 'results/alns_m200/tasks_0100_seed_00.json').read_text(encoding='utf-8')
        )['instance_sha256']
        if digest != frozen:
            raise RuntimeError(
                f'unit-mode control diverged from the frozen 100-task graph: {digest} != {frozen}')
    record = {
        'status': 'running', 'started_at': datetime.now().isoformat(),
        'tasks': count, 'seed': seed, 'penalty': PENALTY,
        'volume_mode': mode,
        'volume_rule': volume_rule(mode, seed, count),
        'volume_histogram': histogram,
        'volume_total_gb': sum(applied),
        'instance_sha256': digest,
        'graph_nodes': len(graph), 'graph_edges': graph.number_of_edges(),
        'build_time': build_time, 'algorithms': {},
    }
    write_json(path, record)

    # run_pb hardcodes penalty 200 in the frozen benchmark module, so the M=600
    # relaxation is constructed here directly from the same class.
    methods = [('P&B', lambda: run_pb_600(instance, graph)),
               ('GRASP', lambda: run_randomized_construction(
                   instance, graph, penalty=PENALTY, seed=20000 + seed, restarts=20)),
               ('ALNS', lambda: run_alns(
                   instance, graph, penalty=PENALTY, seed=10000 + seed,
                   time_limit=alns_time_limit, max_iterations=10000,
                   stall_iterations=stall_iterations))]

    for name, method in methods:
        result = method()
        audit_start = time.perf_counter()
        result['audit'] = audit_schedule(instance, graph, result, PENALTY)
        result['validation_time'] = time.perf_counter() - audit_start
        if not result['audit']['feasible']:
            raise RuntimeError(f'{name} failed independent validation: {result["audit"]}')
        record['algorithms'][name] = result
        write_json(path, record)
        print(f'  {name}: obj={result["obj"]:.1f} completed={result["completed"]} '
              f'time={result["solve_time"]:.3f}s iterations={result.get("iterations", "-")}',
              flush=True)
        gc.collect()

    lower = record['algorithms']['P&B']['lower_bound']
    for result in record['algorithms'].values():
        if result['obj'] < lower - 1e-5:
            raise RuntimeError('Heuristic objective below certified bound')
        result['gap_to_certified_bound_pct'] = 100 * (result['obj'] - lower) / lower
    record.update(status='complete', finished_at=datetime.now().isoformat(),
                  total_time=time.perf_counter() - start,
                  peak_rss_bytes=psutil.Process().memory_info().peak_wset)
    write_json(path, record)
    print(f'DONE tasks={count} seed={seed} total={record["total_time"]:.1f}s', flush=True)
    graph.clear()
    del graph, instance
    gc.collect()


def run_pb_600(instance, graph):
    """Price-and-branch at the service-first penalty.

    Mirrors run_pb in run_alns_benchmark.py with penalty 600. Kept separate so
    the frozen M=200 benchmark module is not edited.
    """
    from run_alns_benchmark import (ContinuousRelaxation, compute_true_objective,
                                    find_frac_flows, solve_last_iter_int)
    start = time.perf_counter()
    cr = ContinuousRelaxation(graph, instance.tasks, instance.sources, instance.sinks,
                              PENALTY, instance.heuristic, instance.sat_params, k_alt=1)
    setup = time.perf_counter() - start
    cr.build_restricted_masterproblem()
    cr.solve(filter_on=True, dijkstra=False, max_iterations=200)

    certification_start = time.perf_counter()
    passes = []
    for p in range(1, 201):
        cr.solve_rmp()
        cr.update_duals()
        tasks = sorted(instance.tasks, key=lambda t: t.task_id)
        if cr.native_pricing:
            cr.native_pricing.prepare(cr.shadow_cap, [t.data_volume for t in tasks])
        flows = []
        for task in tasks:
            flow = cr.pricing_solver(task, dijkstra=False)
            if flow is not None:
                flows.append(flow)
        before = len(cr.x)
        passes.append({'pass': p, 'negative_paths': len(flows), 'columns_before': before})
        if not flows:
            break
        cr.add_new_flows(flows)
        passes[-1]['columns_after'] = len(cr.x)
        if len(cr.x) == before:
            raise RuntimeError('Full pricing found only duplicate negative-cost columns')
    else:
        raise RuntimeError('P&B exact full-pricing certification limit reached')

    lp_obj = float(cr.rmp.ObjVal)
    lower = lp_obj - 1e-6 * len(instance.tasks)
    certification_time = time.perf_counter() - certification_start

    integer_start = time.perf_counter()
    fractional = len(find_frac_flows(cr))
    if fractional:
        cr, value, _ = solve_last_iter_int(cr)
    else:
        value = compute_true_objective(cr)
    integer_time = time.perf_counter() - integer_start

    schedule = [{'task': flow.trip.task_id, 'nodes': list(flow.path[1])}
                for flow, var in cr.x.items() if var.X > 0.5]
    rejected = [t.task_id for t, var in cr.dummy_vars.items() if var.X > 0.5]
    result = {'obj': float(value), 'lp_obj': lp_obj, 'lower_bound': lower,
              'solve_time': time.perf_counter() - start, 'preprocessing_time': setup,
              'certification_time': certification_time, 'integer_recovery_time': integer_time,
              'certified_no_negative_columns': True, 'certification_passes': passes,
              'fractional_flows': fractional, 'completed': len(schedule),
              'rejected': len(rejected), 'schedule': schedule,
              'rejected_tasks': rejected, 'num_columns': len(cr.x),
              'iterations': cr.num_it,
              'gap_pct': 100 * (value - lower) / lower if lower > 0 else None}
    cr.rmp.dispose()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='100')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--mode', default='discrete', choices=sorted(VOLUME_MODES),
                    help='discrete: q in {1,2,3}; unit: homogeneous control')
    ap.add_argument('--output', default='results/heterogeneous_m600')
    ap.add_argument('--alns-time-limit', type=float, default=60.0)
    ap.add_argument('--stall-iterations', type=int, default=None)
    args = ap.parse_args()
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = [int(x) for x in args.tasks.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    print(f'volume mode={args.mode}, M={PENALTY}, tasks={counts}, seeds={seeds}',
          flush=True)

    errors = []
    for count in counts:
        for seed in seeds:
            try:
                run_one(seed, count, out_dir, mode=args.mode,
                        alns_time_limit=args.alns_time_limit,
                        stall_iterations=args.stall_iterations)
            except Exception as exc:  # noqa: BLE001 - record and continue
                errors.append({'tasks': count, 'seed': seed, 'mode': args.mode,
                               'error': repr(exc), 'traceback': traceback.format_exc()})
                print(f'FAILED tasks={count} seed={seed}: {exc}', flush=True)
    write_json(out_dir / f'errors_{args.mode}.json', errors)
    print(f'finished mode={args.mode}, {len(errors)} failures', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
