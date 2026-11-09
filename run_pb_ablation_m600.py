"""Matched M=600 pricing ablation over the complete selected case set.

The four variants are run from fresh masters on the same instance:
``A*+filter``, ``A* no filter``, ``Dijkstra+filter`` and
``Dijkstra no filter``.  The driver keeps
the original M=200 campaign and all manuscript files untouched.  It uses the
expanded task generator that was frozen for the M=600 comparison/scalability
campaign, so the instance hashes can be checked against those campaigns.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import textwrap
import time
import traceback

import run_alns_benchmark as bench
import run_pb_ablation_m200 as pb
import Generator.instance_generator as generator
from Solver.native_pricing import NativePricing, LIBRARY
from run_scalability_m200 import expanded_generator_source

ROOT = Path(__file__).resolve().parent
VARIANTS = ('A*+filter', 'A* no filter', 'Dijkstra+filter', 'Dijkstra no filter')
PENALTY = 600
DEFAULT_TASKS = (100, 200, 500, 1000, 2000)
DEFAULT_SEEDS_BY_TASKS = {
    100: tuple(range(10)),
    200: tuple(range(10)),
    500: tuple(range(10)),
    1000: tuple(range(10)),
    2000: (0,),
}


def parameterized_pb_source():
    """Export the frozen M=200 runner with penalty supplied explicitly."""
    source = textwrap.dedent(inspect.getsource(pb.run_variant))
    old = 'def run_variant(instance, graph, variant):'
    assert source.count(old) == 1
    source = source.replace(old, 'def run_variant(instance, graph, variant, penalty):')
    old = '                              200, instance.heuristic, instance.sat_params, k_alt=1)'
    assert source.count(old) == 1
    source = source.replace(old,
                            '                              penalty, instance.heuristic, instance.sat_params, k_alt=1)')
    old = "    use_dijkstra = variant == 'Dijkstra+filter'"
    assert source.count(old) == 1
    source = source.replace(old, "    use_dijkstra = variant.startswith('Dijkstra')")
    old = "filter_on=variant != 'A* no filter'"
    assert source.count(old) == 1
    source = source.replace(old, "filter_on=not variant.endswith('no filter')")
    old = 'bench.audit_schedule(instance, graph, result, 200)'
    assert source.count(old) == 1
    return source.replace(old, 'bench.audit_schedule(instance, graph, result, penalty)')


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def reference_path(count: int, seed: int) -> Path:
    if count in (100, 500, 1000):
        return ROOT / 'results/alns_m200' / f'tasks_{count:04d}_seed_{seed:02d}.json'
    if count == 200:
        return ROOT / 'results/pb_ablation_m200_20261014' / f'tasks_{count:04d}_seed_{seed:02d}.json'
    if count == 2000:
        return ROOT / 'results/scalability_m600_20261014' / f'tasks_{count:05d}_seed_{seed:02d}.json'
    raise ValueError(f'No frozen reference for task count {count}')


def run_counted_variant(instance, graph, variant, run_variant):
    """Run one variant and verify that every pricing call enters the native backend."""
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
        raise RuntimeError(
            f'Native pricing count mismatch for {variant}: '
            f'{calls[0]} != {result["total_pricing_calls"]}')
    result['native_shortest_path_calls'] = calls[0]
    result['all_pricing_calls_native'] = True
    result['penalty'] = PENALTY
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/pb_ablation_m600_2x2_20261015')
    parser.add_argument('--tasks', nargs='+', type=int, default=list(DEFAULT_TASKS))
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Use the same seed list for every task count.')
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks) or min(args.tasks) <= 0:
        raise ValueError('Tasks must be distinct positive integers')
    seeds_by_tasks = {count: tuple(args.seeds) if args.seeds is not None
                      else DEFAULT_SEEDS_BY_TASKS.get(count, (0,))
                      for count in args.tasks}
    if any(len(set(seeds)) != len(seeds) or min(seeds) < 0
           for seeds in seeds_by_tasks.values()):
        raise ValueError('Seeds must be distinct non-negative integers')
    if not LIBRARY.is_file():
        raise FileNotFoundError(f'Native shortest-path library missing: {LIBRARY}')

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'results/alns_m200/protocol.json').read_text(encoding='utf-8'))
    effective_pb = parameterized_pb_source()
    effective_generator = expanded_generator_source()
    source_sha256 = dict(prior['source_sha256'])
    source_sha256['Solver/column_generation.py'] = hashlib.sha256(
        (ROOT / 'Solver/column_generation.py').read_bytes()).hexdigest()
    source_fingerprint = hashlib.sha256(
        json.dumps(source_sha256, sort_keys=True).encode('utf-8')).hexdigest()
    protocol = {
        'version': 1,
        'penalty': PENALTY,
        'tasks': list(args.tasks),
        'seeds_by_tasks': {str(k): list(v) for k, v in seeds_by_tasks.items()},
        'variants': list(VARIANTS),
        'k_alt': 1,
        'max_iterations': 200,
        'variant_order': 'rotate by seed modulo four',
        'generator': 'expanded candidate cap: 5000 for <=1000 tasks, max(5000,20*n) above 1000',
        'source_fingerprint': source_fingerprint,
        'source_sha256': source_sha256,
        'filter_policy': 'for >200 tasks, reprice tasks with no generated column or with a positive-flow RMP column touching a bounded arc; <=200 tasks use full pricing because screening overhead dominates',
        'rmp_schedule': 'one RMP solve at each stabilization-loop iteration; no post-pricing duplicate solve',
        'effective_pb_sha256': hashlib.sha256(effective_pb.encode()).hexdigest(),
        'effective_generator_sha256': hashlib.sha256(effective_generator.encode()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_dll_sha256': hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'gurobi': list(bench.gp.gurobi.version()),
            'gurobi_threads': 8,
            'execution': 'sequential; fresh master for every variant',
        },
        'timing': ('perf_counter wall clock from solver initialization through schedule extraction; '
                   'includes master construction, CG, internal full-pricing certification, '
                   'final full-pricing confirmation and integer recovery; excludes '
                   'instance/graph construction, independent audit and output I/O'),
        'comparison': 'instance hashes are checked against the frozen ALNS, M=200 ablation and M=600 scalability campaigns',
    }
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        saved_protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        # The first campaign intentionally predates the fourth cell of the
        # 2x2 ablation.  Its three completed cells are retained and extended
        # in place; all other protocol changes still require a fresh output.
        compatible = (
            saved_protocol.get('penalty') == protocol['penalty'] and
            set(protocol['tasks']).issubset(set(saved_protocol.get('tasks', []))) and
            all(set(protocol['seeds_by_tasks'].get(str(k), []))
                .issubset(set(saved_protocol.get('seeds_by_tasks', {}).get(str(k), [])))
                for k in protocol['tasks']) and
            set(saved_protocol.get('variants', [])) <= set(VARIANTS) and
            saved_protocol.get('source_fingerprint') == protocol['source_fingerprint'])
        if not compatible:
            raise RuntimeError('Existing protocol differs; select another output directory')
    write_json(protocol_path, protocol)
    (out / 'effective_run_variant.py').write_text(effective_pb, encoding='utf-8')
    (out / 'effective_generate_tasks.py').write_text(effective_generator, encoding='utf-8')

    pb_namespace = dict(pb.__dict__)
    exec(compile(effective_pb, str(out / 'effective_run_variant.py'), 'exec'), pb_namespace)
    run_variant = pb_namespace['run_variant']
    gen_namespace = dict(generator.__dict__)
    exec(compile(effective_generator, str(out / 'effective_generate_tasks.py'), 'exec'), gen_namespace)
    original_generate = generator.Instance.generate_tasks
    generator.Instance.generate_tasks = gen_namespace['generate_tasks']
    keep_active = ctypes.windll.kernel32.SetThreadExecutionState
    if not keep_active(0x80000001):
        raise RuntimeError('Could not inhibit system sleep')

    current_path = None
    record = None
    try:
        print('BOOTSTRAP frozen 100-task calibration', flush=True)
        calibration = bench.build_instance(0, 100)
        calibration_graph = bench.canonical_graph(calibration)
        frozen = json.loads((ROOT / 'results/alns_m200/tasks_0100_seed_00.json').read_text(encoding='utf-8'))
        if bench.instance_hash(calibration, calibration_graph) != frozen['instance_sha256']:
            raise RuntimeError('100-task calibration graph does not match frozen instance')
        calibration_graph.clear()
        del calibration_graph, calibration
        gc.collect()

        for count in args.tasks:
            for seed in seeds_by_tasks[count]:
                current_path = out / f'tasks_{count:05d}_seed_{seed:02d}.json'
                saved = json.loads(current_path.read_text(encoding='utf-8')) if current_path.exists() else None
                if saved is not None and saved.get('status') == 'complete' \
                        and set(saved.get('algorithms', {})) == set(VARIANTS):
                    print(f'CACHED tasks={count} seed={seed}', flush=True)
                    continue
                if saved is not None and saved.get('status') not in ('complete', 'running'):
                    raise RuntimeError(f'Incomplete or failed result retained: {current_path}')
                ref = reference_path(count, seed)
                if not ref.exists():
                    raise FileNotFoundError(f'Matched reference is missing: {ref}')
                reference = json.loads(ref.read_text(encoding='utf-8'))
                print(f'BUILD tasks={count} seed={seed}', flush=True)
                case_start = time.perf_counter()
                instance = bench.build_instance(seed, count)
                graph = bench.canonical_graph(instance)
                if len(instance.tasks) != count:
                    raise RuntimeError(f'Actual tasks {len(instance.tasks)} != {count}')
                digest = bench.instance_hash(instance, graph)
                if digest != reference['instance_sha256']:
                    raise RuntimeError(
                        f'Graph mismatch for {count}/{seed}: {digest} != {reference["instance_sha256"]}')
                if saved is None:
                    record = {
                        'status': 'running', 'started_at': datetime.now().isoformat(),
                        'tasks': count, 'seed': seed, 'penalty': PENALTY,
                        'source_fingerprint': source_fingerprint,
                        'driver_sha256': protocol['driver_sha256'],
                        'instance_sha256': digest, 'graph_nodes': len(graph),
                        'graph_edges': graph.number_of_edges(),
                        'graph_build_time': time.perf_counter() - case_start,
                        'reference': str(ref.relative_to(ROOT)),
                        'execution_order': list(VARIANTS[seed % len(VARIANTS):] + VARIANTS[:seed % len(VARIANTS)]),
                        'algorithms': {},
                    }
                else:
                    record = saved
                    if record.get('instance_sha256') != digest:
                        raise RuntimeError(f'Existing record hash mismatch: {current_path}')
                    record.update(status='running', phase=None,
                                  driver_sha256=protocol['driver_sha256'])
                    order = list(record.get('execution_order', ()))
                    order.extend(v for v in VARIANTS if v not in order)
                    record['execution_order'] = order
                write_json(current_path, record)
                for variant in record['execution_order']:
                    if variant in record['algorithms']:
                        continue
                    print(f'  START tasks={count} seed={seed} M={PENALTY} {variant}', flush=True)
                    record['phase'] = variant
                    write_json(current_path, record)
                    result = run_counted_variant(instance, graph, variant, run_variant)
                    if not result['audit']['feasible']:
                        raise RuntimeError(f'Independent schedule audit failed: {result["audit"]}')
                    record['algorithms'][variant] = result
                    write_json(current_path, record)
                    print(f'  DONE {variant}: obj={result["obj"]:.1f} '
                          f'completed={result["completed"]} solve={result["solve_time"]:.3f}s '
                          f'calls={result["total_pricing_calls"]} audit=OK', flush=True)
                    gc.collect()
                values = list(record['algorithms'].values())
                lp_values = [value['lp_obj'] for value in values]
                if max(lp_values) - min(lp_values) > 1e-5:
                    raise RuntimeError('Certified LP optima differ between variants')
                record.update(
                    status='complete', finished_at=datetime.now().isoformat(),
                    case_total_time=time.perf_counter() - case_start,
                    certified_lp_objective=lp_values[0],
                    all_schedule_audits_passed=True,
                    same_certified_lp_optimum=True,
                )
                write_json(current_path, record)
                print(f'DONE CASE tasks={count} seed={seed}', flush=True)
                graph.clear()
                del graph, instance, record, values
                record = None
                gc.collect()
        print('ALL M600 ABLATION CASES COMPLETE', flush=True)
    except BaseException:
        if current_path is not None and record is not None:
            record.update(status='error', failed_at=datetime.now().isoformat(),
                          error=traceback.format_exc())
            write_json(current_path, record)
        raise
    finally:
        keep_active(0x80000000)
        generator.Instance.generate_tasks = original_generate


if __name__ == '__main__':
    main()
