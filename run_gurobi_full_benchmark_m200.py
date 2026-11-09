"""Matched full-constellation P&B / direct Gurobi comparison; data only."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import time

import psutil
import run_alns_benchmark as bench
import run_pb_ablation_m200 as pb
from gurobi_full_models_m200 import run_aggregate,run_commodity

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/gurobi_full_m200_20261014_pilot'


def verify_preservation(prior):
    for name,digest in prior['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen model source changed: {name}')
    path = ROOT.parent/'audit/2026-10-14/timing_ablation_preservation_sha256.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    for name,digest in saved.items():
        if hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Existing manuscript or results changed: {name}')
    return {'preserved_files':len(saved),'source_files':len(prior['source_sha256']),
            'all_sha256_match':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',type=int,nargs='+',default=[100,500,1000])
    parser.add_argument('--seeds',type=int,nargs='+',default=[0])
    parser.add_argument('--time-limit',type=float,default=600)
    parser.add_argument('--soft-mem-gb',type=float,default=6)
    parser.add_argument('--max-arc-variables',type=int,default=5_000_000)
    parser.add_argument('--formulations',nargs='+',choices=['aggregate','commodity'],default=['aggregate','commodity'])
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    prior = json.loads((ROOT/'results/alns_m200/protocol.json').read_text(encoding='utf-8'))
    preserved = verify_preservation(prior)
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ['run_gurobi_full_benchmark_m200.py','gurobi_full_models_m200.py','run_pb_ablation_m200.py']}
    plan = {'version':1,'penalty':200,'tasks':args.tasks,'seeds':args.seeds,
            'rs_count':136,'cs_count':1321,'formulations':args.formulations,
            'time_limit_optimize_seconds':args.time_limit,'soft_mem_gb_per_model':args.soft_mem_gb,
            'max_commodity_arc_variables':args.max_arc_variables,
            'resource_budget':'do not construct commodity models exceeding the stated variable budget; record exact profile and NOT_BUILT_RESOURCE_BUDGET, not TIME_LIMIT; pilot 100-task commodity reached 6 GB soft cap in presolve',
            'threads':8,'source_fingerprint':prior['source_fingerprint'],
            'source_sha256':prior['source_sha256'],'driver_sha256':hashes,
            'environment':prior['environment'],
            'execution':'sequential fresh models; rotate method order by seed; no P&B columns, incumbents, MIP starts, or duals supplied to Gurobi baselines',
            'timing':'perf_counter from algorithm setup through complete schedule extraction; model build/optimize/extraction reported separately; input graph, independent audit and I/O excluded for every method',
            'commodity':'binary task-arc model; sparse matrix build; all arcs satisfying individual cost lower bound <=M retained; optimum preserved by rejecting any task path costing >M; all resource coefficients identical to P&B',
            'aggregate':'continuous full arc flow with one rejection arc per source; exact unit-volume/common-sink reduction with integer-capacity floors; all original graph arcs included',
            'comparison':'match graph hashes and objective to prior ALNS; compare optimum or explicit incumbent/bound/gap at limit; report outcomes regardless of which method is faster'}
    path = out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Existing protocol differs; select a new output directory')
    else:
        bench.write_json(path,plan)
    keep_active = ctypes.windll.kernel32.SetThreadExecutionState
    if not keep_active(0x80000001):
        raise RuntimeError('Could not request system availability')
    try:
        for count in args.tasks:
            for seed in args.seeds:
                case_path = out/f'tasks_{count:04d}_seed_{seed:02d}.json'
                if case_path.exists():
                    old = json.loads(case_path.read_text(encoding='utf-8'))
                    if old['status'] == 'complete':
                        print(f'CACHED tasks={count} seed={seed}',flush=True)
                        continue
                    raise RuntimeError('Existing incomplete case retained; do not silently retry')
                print(f'BUILD tasks={count} seed={seed}',flush=True)
                start = time.perf_counter()
                instance = bench.build_instance(seed,count)
                graph = bench.canonical_graph(instance)
                build_time = time.perf_counter()-start
                digest = bench.instance_hash(instance,graph)
                reference = json.loads((ROOT/f'results/alns_m200/tasks_{count:04d}_seed_{seed:02d}.json').read_text(encoding='utf-8'))
                if digest != reference['instance_sha256']:
                    raise RuntimeError('Input graph differs from matched ALNS campaign')
                record = {'status':'running','started_at':datetime.now().isoformat(),
                          'tasks':count,'seed':seed,'penalty':200,'instance_sha256':digest,
                          'source_fingerprint':prior['source_fingerprint'],
                          'graph_nodes':len(graph),'graph_edges':graph.number_of_edges(),
                          'graph_build_time':build_time,'available_ram_before_models':psutil.virtual_memory().available,
                          'preservation_before':preserved,'algorithms':{}}
                bench.write_json(case_path,record)
                methods = ['P&B']+args.formulations
                shift = seed%len(methods)
                methods = methods[shift:]+methods[:shift]
                record['execution_order'] = methods
                for method in methods:
                    print(f'START tasks={count} seed={seed} {method}',flush=True)
                    log = out/f'tasks_{count:04d}_seed_{seed:02d}_{method.replace("&","")}.gurobi.log'
                    if method == 'P&B':
                        result = pb.run_variant(instance,graph,'A*+filter')
                        result['status'] = 'CERTIFIED_LP_EQUALS_IP' if abs(result['obj']-result['lp_obj']) < 1e-5 else 'CERTIFIED_BOUND'
                    elif method == 'aggregate':
                        result = run_aggregate(instance,graph,200,args.time_limit,8,args.soft_mem_gb,log)
                    else:
                        def progress(data):
                            print(f'PROGRESS tasks={count} seed={seed} commodity {json.dumps(data)}',flush=True)
                        result = run_commodity(instance,graph,200,args.time_limit,8,args.soft_mem_gb,log,
                                               reference['algorithms']['P&B']['obj'],progress,
                                               args.max_arc_variables)
                    result['finished_at'] = datetime.now().isoformat()
                    if result['obj'] is not None and result['status'] in ('OPTIMAL','CERTIFIED_LP_EQUALS_IP'):
                        if abs(result['obj']-reference['algorithms']['P&B']['obj']) > 1e-5:
                            raise RuntimeError('Exact full-model optimum disagrees with certified P&B')
                    record['algorithms'][method] = result
                    bench.write_json(case_path,record)
                    print(f'DONE tasks={count} seed={seed} {method} obj={result["obj"]} '
                          f'time={result["solve_time"]}s status={result["status"]} '
                          f'variables={result.get("num_variables",result.get("num_variables_planned",result.get("num_columns")))}',flush=True)
                    gc.collect()
                record.update(status='complete',finished_at=datetime.now().isoformat(),
                              preservation_after=verify_preservation(prior))
                bench.write_json(case_path,record)
                graph.clear()
                del graph,instance,record,result
                gc.collect()
        print('ALL MATCHED GUROBI CASES COMPLETE',flush=True)
    finally:
        keep_active(0x80000000)


if __name__ == '__main__':
    main()
