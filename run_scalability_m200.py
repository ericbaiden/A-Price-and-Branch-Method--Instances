"""Matched 2k--10k-task scaling: frozen C++ P&B and full-graph Gurobi flow.

Only the candidate-attempt ceiling is extended in the task generator. The
effective method is exported verbatim beside the frozen campaign protocol.
No manuscript, earlier result, Solver, Generator, or DLL file is modified.
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
import textwrap
import threading
import time
import traceback

import psutil
import run_alns_benchmark as bench
import run_pb_ablation_m200 as pb
import Generator.instance_generator as generator
from gurobi_full_models_m200 import run_aggregate
from run_gurobi_full_benchmark_m200 import verify_preservation
from Solver.native_pricing import NativePricing, LIBRARY

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'results/scalability_m200_20261014'


def expanded_generator_source():
    source = textwrap.dedent(inspect.getsource(generator.Instance.generate_tasks))
    old = 'max_attempts = 5000  # Increased for large scale instances'
    assert source.count(old) == 1
    source = source.replace(old,
        'max_attempts = 5000 if num_tasks <= 1000 else max(5000, 20 * num_tasks)')
    marker = '    # Shuffle candidate targets\n'
    assert source.count(marker) == 1
    metadata = '''    self.scaling_generation = {
        'requested_tasks': num_tasks, 'candidate_attempt_budget': max_attempts,
        'visible_candidates': len(candidate_targets),
        'candidate_sampling_rule': 'unchanged RS-0 visibility screen; original RNG and imaging-window rules',
    }
    if len(candidate_targets) < num_tasks:
        raise RuntimeError(f'Insufficient visible targets: {len(candidate_targets)} for {num_tasks}')

'''
    source = source.replace(marker, metadata + marker)
    source = source.replace('    for attempt in range(max_attempts):\n',
        '    for attempt in range(max_attempts):\n'
        '        if attempt % 1000 == 0:\n'
        "            print(f'TARGETS requested={num_tasks} attempts={attempt} visible={len(candidate_targets)}', flush=True)\n")
    return source


class ResourceSampler:
    """Sample this process; graph and Python overhead are included in RSS."""
    def __init__(self):
        self.process = psutil.Process()
        self.stopped = threading.Event()
        self.peak_rss = 0
        self.minimum_available = psutil.virtual_memory().available
        self.samples = 0

    def sample(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
        self.samples += 1

    def loop(self):
        while not self.stopped.wait(2):
            self.sample()

    def __enter__(self):
        self.sample()
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.worker.join()
        self.sample()

    def result(self):
        return {'process_rss_peak_bytes': self.peak_rss,
                'minimum_system_available_bytes': self.minimum_available,
                'sample_count': self.samples, 'sample_period_seconds': 2,
                'scope': 'whole Python process including common instance/graph and allocator state; not isolated solver memory'}


def run_cpp_pb(instance, graph):
    original = NativePricing.path
    counter = [0]
    def counted_path(self, *args, **kwargs):
        counter[0] += 1
        return original(self, *args, **kwargs)
    NativePricing.path = counted_path
    try:
        result = pb.run_variant(instance, graph, 'A*+filter')
    finally:
        NativePricing.path = original
    assert counter[0] == result['total_pricing_calls'], (counter[0], result['total_pricing_calls'])
    result['native_shortest_path_calls'] = counter[0]
    result['all_pricing_calls_native'] = True
    result['status'] = ('CERTIFIED_LP_EQUALS_IP' if abs(result['obj']-result['lp_obj']) < 1e-5
                        else 'CERTIFIED_BOUND')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', type=int, default=[2000,3000,5000,10000])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--time-limit', type=float, default=600)
    parser.add_argument('--soft-mem-gb', type=float, default=8)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    assert len(set(args.tasks)) == len(args.tasks) and min(args.tasks) > 0
    assert len(set(args.seeds)) == len(args.seeds) and min(args.seeds) >= 0
    assert LIBRARY.is_file(), 'Required C++ backend missing'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT/'results/alns_m200/protocol.json').read_text(encoding='utf-8'))
    preserved = verify_preservation(prior)
    old_results = json.loads((ROOT/'results/gurobi_full_m200_20261014_summary/verification.json').read_text(encoding='utf-8'))
    for name,digest in old_results['raw_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    effective_source = expanded_generator_source()
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ['run_scalability_m200.py', 'run_pb_ablation_m200.py', 'gurobi_full_models_m200.py']}
    protocol = {'version':1, 'penalty':200, 'tasks':args.tasks, 'seeds':args.seeds,
                'rs_count':136, 'cs_count':1321, 'methods':['P&B','aggregate'],
                'native_backend':'frozen C++ CSR exact A* shortest_path.dll; every pricing call counted',
                'native_dll_sha256':hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
                'k_alt':1, 'pb_max_iterations_per_phase':200,
                'gurobi_threads':8, 'aggregate_optimize_time_limit_seconds':args.time_limit,
                'aggregate_soft_mem_gb':args.soft_mem_gb,
                'generator_extension':'candidate cap 5000 for <=1000 tasks; max(5000,20*n) for larger scales; all other sampling/model rules unchanged; actual task count must equal requested count',
                'effective_generate_tasks_sha256':hashlib.sha256(effective_source.encode()).hexdigest(),
                'source_sha256':prior['source_sha256'], 'source_fingerprint':prior['source_fingerprint'],
                'driver_sha256':hashes, 'environment':prior['environment'],
                'execution':'one case/method at a time, seed outer loop; rotate P&B/aggregate order by seed; fresh models, no shared paths/duals/incumbents',
                'timing':'algorithm setup through complete schedule extraction including full pricing certification; graph construction, audit, disposal, resource sampling and I/O separately recorded; physical contacts cached after 100-task calibration',
                'resource_measurement':'2-second process RSS samples include common input graph and Python allocations; not standalone solver peak',
                'calibration':'before large cases, adapted generator must match the frozen 100-task seed-0 graph hash',
                'preservation':preserved, 'prior_gurobi_raw_sha256':old_results['raw_sha256']}
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    else:
        bench.write_json(protocol_path, protocol)
    effective_path = out/'effective_generate_tasks.py'
    if effective_path.exists():
        assert effective_path.read_text(encoding='utf-8') == effective_source
    else:
        effective_path.write_text(effective_source, encoding='utf-8')
    namespace = dict(generator.__dict__)
    exec(compile(effective_source, str(effective_path), 'exec'), namespace)
    original_generate = generator.Instance.generate_tasks
    generator.Instance.generate_tasks = namespace['generate_tasks']
    keep_active = ctypes.windll.kernel32.SetThreadExecutionState
    assert keep_active(0x80000001), 'Could not inhibit sleep during experiment'
    current_path = None
    record = None
    try:
        print('BOOTSTRAP frozen 100-task calibration / physical contacts', flush=True)
        started = time.perf_counter()
        instance = bench.build_instance(0,100)
        graph = bench.canonical_graph(instance)
        digest = bench.instance_hash(instance,graph)
        reference = json.loads((ROOT/'results/alns_m200/tasks_0100_seed_00.json').read_text(encoding='utf-8'))
        assert digest == reference['instance_sha256'], 'Small-scale graph changed'
        calibration = {'recorded_at':datetime.now().isoformat(), 'elapsed_seconds':time.perf_counter()-started,
                       'tasks':len(instance.tasks), 'instance_sha256':digest, 'matches_frozen_100_task_seed_0':True}
        bench.write_json(out/'calibration.json', calibration)
        graph.clear()
        del instance,graph
        gc.collect()
        for seed in args.seeds:
            for count in args.tasks:
                current_path = out/f'tasks_{count:05d}_seed_{seed:02d}.json'
                if current_path.exists():
                    saved = json.loads(current_path.read_text(encoding='utf-8'))
                    assert saved['status'] == 'complete', 'Incomplete results retained; investigate before rerunning'
                    print(f'CACHED tasks={count} seed={seed}', flush=True)
                    continue
                verify_preservation(prior)
                record = {'status':'running','tasks':count,'seed':seed,'penalty':200,
                          'started_at':datetime.now().isoformat(), 'phase':'graph_build',
                          'source_fingerprint':prior['source_fingerprint'],
                          'effective_generate_tasks_sha256':protocol['effective_generate_tasks_sha256'],
                          'algorithms':{}}
                bench.write_json(current_path,record)
                print(f'BUILD tasks={count} seed={seed}', flush=True)
                started = time.perf_counter()
                with ResourceSampler() as sampled:
                    instance = bench.build_instance(seed,count)
                    assert len(instance.tasks) == count, f'Actual tasks {len(instance.tasks)} != {count}'
                    graph = bench.canonical_graph(instance)
                    build_time = time.perf_counter()-started
                graph_hash = bench.instance_hash(instance,graph)
                record.update(actual_tasks=len(instance.tasks), generation=instance.scaling_generation,
                              graph_build_time=build_time, graph_build_resources=sampled.result(),
                              instance_sha256=graph_hash,graph_nodes=len(graph),graph_edges=graph.number_of_edges(),
                              task_descriptors=[{'task_id':t.task_id,'latitude':t.target_lat,'longitude':t.target_lon,
                                                 'data_volume':t.data_volume,'priority':t.priority,
                                                 'release_slot':t.release_slot,'deadline_slot':t.deadline_slot}
                                                for t in instance.tasks])
                order = ['P&B','aggregate'] if seed%2 == 0 else ['aggregate','P&B']
                record['execution_order'] = order
                for method in order:
                    record['phase'] = method
                    bench.write_json(current_path,record)
                    print(f'START tasks={count} seed={seed} {method} nodes={len(graph)} edges={graph.number_of_edges()}',flush=True)
                    with ResourceSampler() as sampled:
                        if method == 'P&B':
                            result = run_cpp_pb(instance,graph)
                        else:
                            log = out/f'tasks_{count:05d}_seed_{seed:02d}_aggregate.gurobi.log'
                            result = run_aggregate(instance,graph,200,args.time_limit,8,args.soft_mem_gb,log)
                    result['resource_sampling'] = sampled.result()
                    result['finished_at'] = datetime.now().isoformat()
                    record['algorithms'][method] = result
                    bench.write_json(current_path,record)
                    print(f'DONE tasks={count} seed={seed} {method} obj={result["obj"]} time={result["solve_time"]:.3f}s status={result["status"]}',flush=True)
                    gc.collect()
                p,f = (record['algorithms'][m] for m in ('P&B','aggregate'))
                if f['status'] == 'OPTIMAL':
                    assert p['lower_bound'] <= f['obj']+1e-5, 'Invalid certified P&B lower bound'
                    assert p['obj'] >= f['obj']-1e-5, 'P&B improves a full-model optimum'
                    record['pb_gap_to_full_optimum_pct'] = 100*(p['obj']-f['obj'])/max(abs(f['obj']),1e-12)
                    record['objectives_match'] = abs(p['obj']-f['obj']) < 1e-5
                else:
                    record['objectives_match'] = None
                record.update(status='complete',phase='complete',finished_at=datetime.now().isoformat(),
                              preservation_after=verify_preservation(prior))
                bench.write_json(current_path,record)
                print(f'CASE COMPLETE tasks={count} seed={seed} objective_match={record["objectives_match"]}',flush=True)
                graph.clear()
                del graph,instance,result,p,f,record
                record = None
                gc.collect()
        print('ALL LARGE-SCALE CASES COMPLETE',flush=True)
    except BaseException:
        if current_path is not None and record is not None:
            record.update(status='error',failed_at=datetime.now().isoformat(),error=traceback.format_exc())
            bench.write_json(current_path,record)
        raise
    finally:
        keep_active(0x80000000)
        generator.Instance.generate_tasks = original_generate


if __name__ == '__main__':
    main()
