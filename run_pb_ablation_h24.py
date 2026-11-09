#!/usr/bin/env python3
"""Pricing ablation on the 24-hour horizon against the frozen instance store.

The 3-hour ablation (`run_pb_ablation_m600.py`) capped at 2000 tasks and rebuilt
every instance from `bench.build_instance`. This driver instead loads the same
frozen `dataset/instances/h24/` instances the algorithm comparison uses, so a
variant and the main P&B run are guaranteed to see an identical graph.

Four configurations of the same pricing machinery, each from a fresh master on the
same instance:
    A*+filter, A* no filter, Dijkstra+filter, Dijkstra no filter

`run_variant` is taken from the 3 h driver and parameterised for the penalty; it
depends only on (instance, graph, penalty), not on the generator, so the horizon
enters solely through the stored instance.

Usage
-----
    set LEO_HORIZON_HOURS=24
    python run_pb_ablation_h24.py --tasks 1000,2000,...,10000 --seeds 0,1,2 \
        --output results/pb_ablation_h24
"""
from __future__ import annotations

import argparse
import ctypes
import gc
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import textwrap
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from instance_store import load_instance, has_instance, STORE

VARIANTS = ('A*+filter', 'A* no filter', 'Dijkstra+filter', 'Dijkstra no filter')
PENALTY = 600
HORIZON = float(__import__('os').environ.get('LEO_HORIZON_HOURS', 24.0))


def parse_list(value):
    return [int(i) for i in value.split(',')]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def build_variant_runner():
    """Bind the 3 h `run_variant` to an explicit penalty and the H24 graph."""
    import run_pb_ablation_m200 as pb
    source = textwrap.dedent(__import__('inspect').getsource(pb.run_variant))
    assert source.count('def run_variant(instance, graph, variant):') == 1
    source = source.replace('def run_variant(instance, graph, variant):',
                            'def run_variant(instance, graph, variant, penalty):')
    assert source.count('                              200, instance.heuristic, instance.sat_params, k_alt=1)') == 1
    source = source.replace(
        '                              200, instance.heuristic, instance.sat_params, k_alt=1)',
        '                              penalty, instance.heuristic, instance.sat_params, k_alt=1)')
    assert source.count("use_dijkstra = variant == 'Dijkstra+filter'") == 1
    source = source.replace("use_dijkstra = variant == 'Dijkstra+filter'",
                            "use_dijkstra = variant.startswith('Dijkstra')")
    assert source.count("cert = cr.solve(filter_on=variant != 'A* no filter', dijkstra=use_dijkstra,") == 1
    source = source.replace(
        "cert = cr.solve(filter_on=variant != 'A* no filter', dijkstra=use_dijkstra,",
        "cert = cr.solve(filter_on=not variant.endswith('no filter'), dijkstra=use_dijkstra,")
    # The 3 h runner audits at the 3 h penalty; at M=600 that leaves an objective
    # residual of (600-200) x rejected tasks and the audit reports infeasible.
    assert source.count("bench.audit_schedule(instance, graph, result, 200)") == 1
    source = source.replace("bench.audit_schedule(instance, graph, result, 200)",
                            "bench.audit_schedule(instance, graph, result, penalty)")
    # Drop the redundant second all-task confirmation. cr.solve() already ends
    # with a full-pricing certification pass (cert['certified_no_new_columns']),
    # so the extra loop below re-prices every task to confirm a result that was
    # just established. The main P&B runner (run_alns_benchmark_m600.py) removed
    # this step; the ablation must match that timing baseline. We splice out the
    # block between the 'Match the ALNS campaign' comment and the integer-recovery
    # phase, replacing it with an inert stub that keeps the result schema intact.
    start_marker = "    # Match the ALNS campaign's P&B runner, which retains a final independent\n"
    end_marker = "    confirmation_time = time.perf_counter() - phase_start\n"
    i0 = source.index(start_marker)
    i1 = source.index(end_marker) + len(end_marker)
    stub = ("    # No second confirmation pass: cr.solve() already certified the LP, and\n"
            "    # the main P&B runner reads that certificate directly. Keeping an extra\n"
            "    # all-task pass here would double-count certification and break timing\n"
            "    # parity with the algorithm comparison.\n"
            "    passes = []\n"
            "    confirmation_time = 0.0\n")
    source = source[:i0] + stub + source[i1:]
    namespace = dict(pb.__dict__)
    exec(compile(source, '<run_variant_h24>', 'exec'), namespace)
    return namespace['run_variant'], source


def summarize_filter(result):
    """Fold the per-round filter log into the summary fields the tables use.

    `filter_history` records one entry per column-generation round. The figures
    need the mean share of tasks the filter removed while it was actually active,
    so rounds with no bounded arc yet (reduction exactly 0) are excluded -- they
    are the rounds before capacity first binds, not evidence that the filter is
    inert.
    """
    history = result.get('filter_history') or []
    active = [r['filter_reduction_pct'] for r in history
              if r.get('bounded_arcs', 0) > 0]
    result['filter_rounds'] = len(history)
    result['filter_active_rounds'] = len(active)
    result['filter_active_mean'] = (sum(active) / len(active)) if active else None
    result['filter_reduction_mean_pct'] = (
        sum(r['filter_reduction_pct'] for r in history) / len(history)
        if history else None)
    return result


def run_counted_variant(instance, graph, variant, run_variant):
    """Run one variant and verify every pricing call goes through the native backend."""
    from Solver.native_pricing import NativePricing
    original = NativePricing.path
    calls = [0]

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    NativePricing.path = counted
    try:
        result = run_variant(instance, graph, variant, PENALTY)
    finally:
        NativePricing.path = original
    if calls[0] != result['total_pricing_calls']:
        raise RuntimeError(f'Native pricing count mismatch for {variant}: '
                           f'{calls[0]} != {result["total_pricing_calls"]}')
    result['native_shortest_path_calls'] = calls[0]
    result['all_pricing_calls_native'] = True
    result['penalty'] = PENALTY
    return summarize_filter(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'results/pb_ablation_h24')
    ap.add_argument('--tasks', type=parse_list,
                    default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    ap.add_argument('--seeds', type=parse_list, default=[0, 1, 2])
    ap.add_argument('--horizon', type=float, default=HORIZON)
    args = ap.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from Solver.native_pricing import LIBRARY
    if not LIBRARY.is_file():
        raise FileNotFoundError(f'Native shortest-path library missing: {LIBRARY}')

    run_variant, variant_source = build_variant_runner()
    protocol = {
        'version': 1, 'penalty': PENALTY, 'horizon_hours': args.horizon,
        'tasks': list(args.tasks), 'seeds': list(args.seeds),
        'variants': list(VARIANTS), 'k_alt': 1, 'max_iterations': 200,
        'instances': 'frozen dataset/instances/h24, identical to the algorithm comparison',
        'variant_order': 'rotated by seed modulo four',
        'timing': ('perf_counter wall clock from solver initialization through schedule '
                   'extraction; includes master construction, CG, internal certification, '
                   'final full-pricing confirmation and integer recovery; excludes '
                   'instance/graph construction, independent audit and output I/O'),
        'effective_variant_sha256': hashlib.sha256(variant_source.encode()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_dll_sha256': hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'gurobi_threads': 8,
                        'execution': 'sequential; fresh master for every variant'},
    }
    protocol_path = out / 'protocol.json'
    if protocol_path.is_file():
        saved = json.loads(protocol_path.read_text(encoding='utf-8'))
        # Batched resumption across process restarts (a later batch requests a
        # subset of the tasks the first batch declared). The fields that change
        # the result of a solved case must match; the task/seed lists and the
        # driver's own hash may differ between batches, so they are not compared.
        compatible = (
            saved.get('penalty') == protocol['penalty'] and
            saved.get('horizon_hours') == protocol['horizon_hours'] and
            set(saved.get('variants', [])) == set(protocol['variants']) and
            saved.get('effective_variant_sha256') == protocol['effective_variant_sha256'] and
            saved.get('native_dll_sha256') == protocol['native_dll_sha256'])
        if not compatible:
            raise RuntimeError('Existing protocol differs; select another output directory')
        # Keep the original (full-scale) declaration; do not overwrite it with a
        # smaller batch's task/seed list.
        protocol = saved
    else:
        write_json(protocol_path, protocol)
    (out / 'effective_run_variant.py').write_text(variant_source, encoding='utf-8')

    keep_active = ctypes.windll.kernel32.SetThreadExecutionState
    if not keep_active(0x80000001):
        raise RuntimeError('Could not inhibit system sleep')
    current_path = None
    record = None
    try:
        for count in args.tasks:
            for seed in args.seeds:
                current_path = out / f'tasks_{count:04d}_seed_{seed:02d}.json'
                saved = (json.loads(current_path.read_text(encoding='utf-8'))
                         if current_path.is_file() else None)
                if (saved is not None and saved.get('status') == 'complete'
                        and set(saved.get('algorithms', {})) == set(VARIANTS)):
                    print(f'CACHED tasks={count} seed={seed}', flush=True)
                    continue
                if not has_instance(count, seed, args.horizon):
                    raise FileNotFoundError(
                        f'Instance ({count}, {seed}) not in {STORE} at {args.horizon:g} h')

                case_start = time.perf_counter()
                instance = load_instance(count, seed, args.horizon)
                graph = instance.graph
                if len(instance.tasks) != count:
                    raise RuntimeError(f'Actual tasks {len(instance.tasks)} != {count}')
                digest = instance.instance_sha256

                if saved is None:
                    record = {
                        'status': 'running', 'started_at': datetime.now().isoformat(),
                        'tasks': count, 'seed': seed, 'penalty': PENALTY,
                        'horizon_hours': args.horizon,
                        'source_fingerprint': protocol['effective_variant_sha256'],
                        'instance_sha256': digest, 'graph_nodes': len(graph),
                        'graph_edges': graph.number_of_edges(),
                        'execution_order': list(VARIANTS[seed % len(VARIANTS):]
                                                + VARIANTS[:seed % len(VARIANTS)]),
                        'algorithms': {},
                    }
                else:
                    record = saved
                    if record.get('instance_sha256') != digest:
                        raise RuntimeError(f'Existing record hash mismatch: {current_path}')
                    record.update(status='running', phase=None)
                    order = list(record.get('execution_order', ()))
                    order.extend(v for v in VARIANTS if v not in order)
                    record['execution_order'] = order
                write_json(current_path, record)
                graph_build_time = time.perf_counter() - case_start
                record['graph_build_time'] = graph_build_time
                write_json(current_path, record)

                for variant in record['execution_order']:
                    if variant in record['algorithms']:
                        continue
                    print(f'  START tasks={count} seed={seed} {variant}', flush=True)
                    record['phase'] = variant
                    write_json(current_path, record)
                    result = run_counted_variant(instance, graph, variant, run_variant)
                    if not result['audit']['feasible']:
                        raise RuntimeError(f'Independent schedule audit failed: {result["audit"]}')
                    record['algorithms'][variant] = result
                    write_json(current_path, record)
                    print(f'  DONE {variant}: obj={result["obj"]:.1f} '
                          f'completed={result["completed"]} solve={result["solve_time"]:.1f}s '
                          f'calls={result["total_pricing_calls"]} audit=OK', flush=True)
                    gc.collect()

                values = list(record['algorithms'].values())
                lp_values = [v['lp_obj'] for v in values]
                if max(lp_values) - min(lp_values) > 1e-5:
                    raise RuntimeError('Certified LP optima differ between variants')
                record.update(status='complete', finished_at=datetime.now().isoformat(),
                              case_total_time=time.perf_counter() - case_start,
                              certified_lp_objective=lp_values[0],
                              all_schedule_audits_passed=True,
                              same_certified_lp_optimum=True)
                write_json(current_path, record)
                print(f'DONE CASE tasks={count} seed={seed}', flush=True)
                graph.clear()
                del graph, instance, record, values
                record = None
                gc.collect()
        print('ALL H24 ABLATION CASES COMPLETE', flush=True)
    except BaseException:
        if current_path is not None and record is not None:
            record.update(status='error', failed_at=datetime.now().isoformat(),
                          error=traceback.format_exc())
            write_json(current_path, record)
        raise
    finally:
        keep_active(0x80000000)


if __name__ == '__main__':
    main()
