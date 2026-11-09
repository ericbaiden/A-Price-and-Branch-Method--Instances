#!/usr/bin/env python3
"""Recompute only the ALNS stage of the H24 campaign.

Why this exists
---------------
The first H24 campaign charged ALNS's setup (residual-router construction plus one
unconstrained shortest path per task) and its initial construction against the
60 s budget. On 8k-10k tasks that consumed the whole budget, so ALNS returned an
empty schedule and its objective degenerated to M * n. The budget now bounds the
adaptive search only (see `Solver/alns.py`); this script reruns ALNS under the
new semantics while keeping the P&B and GRASP results, which the change cannot
affect.

What is copied and what is recomputed
-------------------------------------
For every case in `--source`:
  * P&B and GRASP results are copied verbatim.
  * ALNS is recomputed and independently validated.
  * The record's `status`, `algorithms`, and aggregate fields are rewritten;
    `source_fingerprint` is updated to the new code fingerprint.

The instance is loaded from the frozen store, so the graph is bit-identical to
the one the first campaign used.

Usage
-----
    set LEO_HORIZON_HOURS=24
    python rerun_alns_h24.py --source results/h24_campaign --output results/h24_campaign_v2
"""
from __future__ import annotations

import argparse
import copy
import gc
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from instance_store import load_instance, has_instance, STORE
from Solver.alns import run_alns
from alns_validation import audit_schedule
import run_alns_benchmark_m600 as campaign


def parse_list(value):
    return [int(i) for i in value.split(',')]


def rerun_case(source: Path, output: Path, count: int, seed: int, horizon: float,
               args, protocol) -> dict:
    src_path = source / f'tasks_{count:04d}_seed_{seed:02d}.json'
    dst_path = output / f'tasks_{count:04d}_seed_{seed:02d}.json'
    if not src_path.is_file():
        raise FileNotFoundError(f'No source record: {src_path}')

    old = campaign.json.loads(src_path.read_text(encoding='utf-8'))
    if old.get('status') != 'complete':
        raise RuntimeError(f'Source record is not complete: {src_path}')

    # Idempotent resume: a finished v2 record for the same code is kept.
    if dst_path.is_file():
        cached = campaign.json.loads(dst_path.read_text(encoding='utf-8'))
        if (cached.get('status') == 'complete'
                and cached.get('source_fingerprint') == protocol['source_fingerprint']):
            print(f'CACHED tasks={count} seed={seed}', flush=True)
            return cached

    print(f'START tasks={count} seed={seed}', flush=True)
    start = time.perf_counter()
    if not has_instance(count, seed, horizon):
        raise FileNotFoundError(f'Instance ({count}, {seed}) is not in {STORE} at {horizon:g} h')
    instance = load_instance(count, seed, horizon)
    graph = instance.graph

    record = copy.deepcopy(old)
    record['alns_budget'] = (
        'time_limit bounds the adaptive search only; preprocessing_time and '
        'construction_time are reported separately')
    record['retained_from'] = str(src_path.relative_to(ROOT))
    record['source_fingerprint'] = protocol['source_fingerprint']
    record['rerun_started_at'] = datetime.now().isoformat()
    record['algorithms'].pop('ALNS', None)
    record['status'] = 'running'
    campaign.write_json(dst_path, record)

    result = run_alns(instance, graph, penalty=600, seed=10000 + seed,
                      time_limit=args.time_limit, max_iterations=args.max_iterations,
                      stall_iterations=args.stall_iterations)
    audit_start = time.perf_counter()
    result['audit'] = audit_schedule(instance, graph, result, 600)
    result['validation_time'] = time.perf_counter() - audit_start
    if not result['audit']['feasible']:
        raise RuntimeError(f'ALNS failed independent validation: {result["audit"]}')
    record['algorithms']['ALNS'] = result
    print(f'  ALNS: obj={result["obj"]:.1f} completed={result["completed"]} '
          f'time={result["solve_time"]:.1f}s (preproc={result["preprocessing_time"]:.1f}s '
          f'construct={result["construction_time"]:.1f}s search={result["search_time"]:.1f}s) '
          f'iters={result["iterations"]}', flush=True)

    lower = record['algorithms']['P&B']['lower_bound']
    for name, alg in record['algorithms'].items():
        if alg['obj'] < lower - 1e-5:
            raise RuntimeError(f'{name} objective below certified bound')
        alg['gap_to_certified_bound_pct'] = 100 * (alg['obj'] - lower) / lower

    record.update(status='complete',
                  finished_at=datetime.now().isoformat(),
                  rerun_time=time.perf_counter() - start)
    campaign.write_json(dst_path, record)
    print(f'DONE tasks={count} seed={seed} rerun={record["rerun_time"]:.1f}s', flush=True)
    graph.clear()
    del graph, instance
    gc.collect()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/h24_campaign')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/h24_campaign_v2')
    parser.add_argument('--tasks', type=parse_list,
                        default=[1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000])
    parser.add_argument('--seeds', type=parse_list, default=[0, 1, 2])
    parser.add_argument('--time-limit', type=float, default=60)
    parser.add_argument('--stall-iterations', type=int, default=None,
                        help='End the search after this many consecutive iterations '
                             'without an incumbent improvement (None disables). The '
                             'stop also requires a minimum share of the budget to be '
                             'spent, so a cheap-instance transient cannot fire it.')
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--horizon', type=float, default=24.0)
    args = parser.parse_args()

    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = campaign.make_protocol(args)
    frozen = args.output / 'protocol.json'
    if frozen.is_file():
        previous = campaign.json.loads(frozen.read_text(encoding='utf-8'))
        if previous != protocol:
            raise RuntimeError('Frozen protocol changed: use a separate output directory')
    else:
        campaign.write_json(frozen, protocol)
    print(f'PROTOCOL {protocol["source_fingerprint"]} output={args.output}', flush=True)

    errors = []
    for count in args.tasks:
        for seed in args.seeds:
            try:
                rerun_case(args.source, args.output, count, seed, args.horizon,
                           args, protocol)
            except Exception:
                detail = traceback.format_exc()
                errors.append({'tasks': count, 'seed': seed, 'traceback': detail})
                print(detail, flush=True)
                campaign.write_json(args.output / 'errors.json', errors)
                raise
    print('ALL ALNS RERUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
