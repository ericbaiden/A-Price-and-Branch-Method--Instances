"""Fresh C++ P&B / full Gurobi experiments after exact M selection.

The frozen P&B runner is compiled with only its two penalty constants changed
to a function argument. Iteration limits, k=1, full certification and complete
wall-clock timing remain identical. The effective function is exported.
"""
import argparse
import ctypes
from datetime import datetime
import gc
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time
import traceback

import run_alns_benchmark as bench
import run_pb_ablation_m200 as pb
import Generator.instance_generator as generator
from gurobi_full_models_m200 import run_aggregate
from run_gurobi_full_benchmark_m200 import verify_preservation
from run_scalability_m200 import expanded_generator_source, ResourceSampler
from Solver.native_pricing import NativePricing, LIBRARY

ROOT=Path(__file__).resolve().parent


def parameterized_pb_source():
    source=textwrap.dedent(inspect.getsource(pb.run_variant))
    old='def run_variant(instance, graph, variant):'
    assert source.count(old)==1
    source=source.replace(old,'def run_variant(instance, graph, variant, penalty):')
    old='                              200, instance.heuristic, instance.sat_params, k_alt=1)'
    assert source.count(old)==1
    source=source.replace(old,'                              penalty, instance.heuristic, instance.sat_params, k_alt=1)')
    old='bench.audit_schedule(instance, graph, result, 200)'
    assert source.count(old)==1
    return source.replace(old,'bench.audit_schedule(instance, graph, result, penalty)')


def run_cpp(instance,graph,penalty,run_variant):
    original=NativePricing.path; calls=[0]
    def counted(self,*args,**kwargs):
        calls[0]+=1; return original(self,*args,**kwargs)
    NativePricing.path=counted
    try: result=run_variant(instance,graph,'A*+filter',penalty)
    finally: NativePricing.path=original
    assert calls[0]==result['total_pricing_calls']
    result.update(native_shortest_path_calls=calls[0],all_pricing_calls_native=True,
                  penalty=penalty,status='CERTIFIED_LP_EQUALS_IP' if abs(result['obj']-result['lp_obj'])<1e-5 else 'CERTIFIED_BOUND')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--penalty',required=True,type=int)
    parser.add_argument('--selection-summary',required=True,type=Path)
    parser.add_argument('--tasks',nargs='+',type=int,default=[2000,3000,5000,10000])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    selection=args.selection_summary.resolve()
    decision=json.loads((selection/'parameter_decision.json').read_text(encoding='utf-8'))
    verification=json.loads((selection/'verification.json').read_text(encoding='utf-8'))
    assert verification['complete'], 'Finish the parameter-selection sample first'
    assert args.penalty>decision['largest_sampled_critical_penalty']
    prior=json.loads((ROOT/'results/alns_m200/protocol.json').read_text(encoding='utf-8'))
    verify_preservation(prior)
    pb_source=parameterized_pb_source(); gen_source=expanded_generator_source()
    ns=dict(pb.__dict__); exec(compile(pb_source,str(out/'effective_run_variant.py'),'exec'),ns)
    run_variant=ns['run_variant']
    ns=dict(generator.__dict__); exec(compile(gen_source,str(out/'effective_generate_tasks.py'),'exec'),ns)
    protocol={'version':1,'penalty':args.penalty,'tasks':args.tasks,'seeds':args.seeds,
              'selection_summary':str(selection),'selection_decision':decision,
              'selection_raw_sha256':verification['raw_sha256'],
              'selection_decision_sha256':hashlib.sha256((selection/'parameter_decision.json').read_bytes()).hexdigest(),
              'source_sha256':prior['source_sha256'],'source_fingerprint':prior['source_fingerprint'],
              'driver_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                               ['run_scalability_selected_m.py','run_pb_ablation_m200.py','gurobi_full_models_m200.py','run_scalability_m200.py']},
              'effective_run_variant_sha256':hashlib.sha256(pb_source.encode()).hexdigest(),
              'effective_generate_tasks_sha256':hashlib.sha256(gen_source.encode()).hexdigest(),
              'native_dll_sha256':hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
              'methods':['P&B','aggregate'],'k_alt':1,'pb_max_iterations_per_phase':200,
              'gurobi_threads':8,'aggregate_optimize_limit_seconds':600,'soft_mem_gb':8,
              'timing':'fresh models; setup through schedule extraction including full pricing certification and integer recovery; graph generation, audit and I/O excluded; not a matched total wall-clock limit study',
              'execution':'one case and one method at a time; method order rotates by seed; no P&B solutions, columns or duals passed to aggregate flow',
              'environment':prior['environment']}
    path=out/'protocol.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==protocol
    else: bench.write_json(path,protocol)
    for name,source in [('effective_run_variant.py',pb_source),('effective_generate_tasks.py',gen_source)]:
        (out/name).write_text(source,encoding='utf-8')
    original=generator.Instance.generate_tasks; generator.Instance.generate_tasks=ns['generate_tasks']
    awake=ctypes.windll.kernel32.SetThreadExecutionState; assert awake(0x80000001)
    current=None; record=None
    try:
        print('BOOTSTRAP frozen 100-task calibration',flush=True)
        instance=bench.build_instance(0,100); graph=bench.canonical_graph(instance)
        reference=json.loads((ROOT/'results/alns_m200/tasks_0100_seed_00.json').read_text(encoding='utf-8'))
        assert bench.instance_hash(instance,graph)==reference['instance_sha256']
        graph.clear(); del graph,instance; gc.collect()
        for seed in args.seeds:
            for count in args.tasks:
                current=out/f'tasks_{count:05d}_seed_{seed:02d}.json'
                if current.exists():
                    assert json.loads(current.read_text(encoding='utf-8'))['status']=='complete', 'Incomplete run requires investigation'
                    continue
                verify_preservation(prior)
                record={'status':'running','tasks':count,'seed':seed,'penalty':args.penalty,
                        'phase':'graph_build','started_at':datetime.now().isoformat(),'algorithms':{},
                        'source_fingerprint':prior['source_fingerprint']}
                bench.write_json(current,record); print(f'BUILD tasks={count} seed={seed}',flush=True)
                start=time.perf_counter(); instance=bench.build_instance(seed,count); graph=bench.canonical_graph(instance)
                assert len(instance.tasks)==count
                record.update(graph_build_time=time.perf_counter()-start,actual_tasks=count,
                              instance_sha256=bench.instance_hash(instance,graph),graph_nodes=len(graph),graph_edges=graph.number_of_edges())
                diagnosis_path=selection.parent/f'tasks_{count:05d}_seed_{seed:02d}.json'
                if diagnosis_path.exists():
                    diagnosis=json.loads(diagnosis_path.read_text(encoding='utf-8'))
                    assert diagnosis['status']=='complete' and diagnosis['instance_sha256']==record['instance_sha256']
                    assert args.penalty>diagnosis['endpoint']['critical_penalty']
                    record['matched_maximum_completed']=diagnosis['endpoint']['maximum_completed']
                order=['P&B','aggregate'] if seed%2==0 else ['aggregate','P&B']; record['execution_order']=order
                for method in order:
                    record['phase']=method; bench.write_json(current,record)
                    print(f'START tasks={count} seed={seed} M={args.penalty} {method}',flush=True)
                    with ResourceSampler() as sampler:
                        result=(run_cpp(instance,graph,args.penalty,run_variant) if method=='P&B' else
                                run_aggregate(instance,graph,args.penalty,600,8,8,out/f'tasks_{count:05d}_seed_{seed:02d}_aggregate.gurobi.log'))
                    result['resource_sampling']=sampler.result()
                    record['algorithms'][method]=result; bench.write_json(current,record)
                    print(f'DONE tasks={count} seed={seed} {method} obj={result["obj"]} completed={result["completed"]} seconds={result["solve_time"]:.3f}',flush=True)
                    gc.collect()
                p,f=(record['algorithms'][method] for method in ('P&B','aggregate'))
                if f['status']=='OPTIMAL':
                    assert p['lower_bound']<=f['obj']+1e-5 and p['obj']>=f['obj']-1e-5
                    record['objectives_match']=abs(p['obj']-f['obj'])<1e-5
                    record['pb_gap_to_full_optimum_pct']=100*(p['obj']-f['obj'])/max(abs(f['obj']),1e-12)
                    if 'matched_maximum_completed' in record:
                        assert f['completed']==record['matched_maximum_completed']
                        if record['objectives_match']: assert p['completed']==f['completed']
                record.update(status='complete',phase='complete',finished_at=datetime.now().isoformat(),preservation_after=verify_preservation(prior))
                bench.write_json(current,record); print(f'CASE COMPLETE tasks={count} seed={seed}',flush=True)
                graph.clear(); del graph,instance,record,result,p,f; record=None; gc.collect()
        print('ALL SELECTED-M CASES COMPLETE',flush=True)
    except BaseException:
        if current and record:
            record.update(status='error',error=traceback.format_exc(),failed_at=datetime.now().isoformat()); bench.write_json(current,record)
        raise
    finally: awake(0x80000000); generator.Instance.generate_tasks=original


if __name__=='__main__': main()
