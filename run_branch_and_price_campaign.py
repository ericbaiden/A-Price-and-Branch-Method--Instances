#!/usr/bin/env python3
"""Branch-and-price campaign on the frozen dataset, for all ten scales.

Runs the exact branch-and-price reference solver on exactly the instances the main
comparison used, so its objective can be checked against P&B run for run. Records
are written next to the main campaign's data but in their own directory, because
B&P is an exactness reference rather than a competing heuristic.

The tree closes at the root whenever integer recovery already attains the certified
bound; in that case B&P is the same computation as P&B and the node count is 1. The
interesting cases are the ones the main campaign flagged with `fractional_flows > 0`,
where the tree actually branches.

Resumable: a case whose record is already complete is skipped.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'Solver'))
sys.path.insert(2, str(ROOT / 'Generator'))

if 'GRB_LICENSE_FILE' not in os.environ:
    for lic in ('D:/software/win64/bin/gurobi.lic',
                'D:/software/Gurobi/win64/bin/gurobi.lic'):
        if Path(lic).is_file():
            os.environ['GRB_LICENSE_FILE'] = lic
            break

from instance_store import load_instance
from Solver.branch_and_price import solve_branch_and_price

PENALTY = 600
SCALES = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
SEEDS = tuple(range(10))
OUT = ROOT / 'results' / 'bp_m600'
REF = ROOT / 'results' / 'alns_m600_final2'
TIME_LIMIT = 1800.0          # per instance; the reference solver is not accelerated


def reference(count: int, seed: int):
    path = REF / f'tasks_{count:04d}_seed_{seed:02d}.json'
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding='utf-8'))
    return rec['algorithms']['P&B'] if rec.get('status') == 'complete' else None


def run(count: int, seed: int) -> dict:
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    inst = load_instance(count, seed)
    graph = inst.graph
    build = time.perf_counter() - t0

    res = solve_branch_and_price(inst, graph, penalty=PENALTY, time_limit=TIME_LIMIT)
    elapsed = time.perf_counter() - t0

    ref = reference(count, seed)
    record = {
        'status': 'complete' if res['tree_closed'] else 'time_limit',
        'tasks': count, 'seed': seed, 'penalty': PENALTY,
        'started_at': started, 'finished_at': datetime.now().isoformat(),
        'load_time': build, 'solve_time': res['solve_time'], 'total_time': elapsed,
        'obj': res['obj'], 'lower_bound': res['lower_bound'],
        'tree_closed': res['tree_closed'], 'nodes': res['nodes'],
        'completed': res['completed'],
        'schedule': res['schedule'],
        'instance_sha256': getattr(inst, 'instance_sha256', ''),
    }
    if ref is not None:
        record['pb_obj'] = ref['obj']
        record['pb_completed'] = ref['completed']
        record['pb_fractional_flows'] = ref['fractional_flows']
        record['pb_solve_time'] = ref['solve_time']
        record['same_objective_as_pb'] = abs(ref['obj'] - res['obj']) < 1e-6
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pending = []
    for n in SCALES:
        for s in SEEDS:
            path = OUT / f'tasks_{n:04d}_seed_{s:02d}.json'
            if path.is_file():
                try:
                    if json.loads(path.read_text(encoding='utf-8')).get('status') == 'complete':
                        continue
                except json.JSONDecodeError:
                    pass
            pending.append((n, s))
    print(f'{len(SCALES) * len(SEEDS) - len(pending)} done, {len(pending)} to run',
          flush=True)

    for i, (n, s) in enumerate(pending, 1):
        path = OUT / f'tasks_{n:04d}_seed_{s:02d}.json'
        print(f'[{i}/{len(pending)}] tasks={n} seed={s}', flush=True)
        try:
            rec = run(n, s)
            path.write_text(json.dumps(rec, indent=2), encoding='utf-8')
            flag = '' if rec.get('same_objective_as_pb', True) else '  <<< DIFFERS FROM P&B'
            print(f"  obj={rec['obj']:.0f} nodes={rec['nodes']} "
                  f"closed={rec['tree_closed']} time={rec['solve_time']:.1f}s{flag}",
                  flush=True)
        except Exception as exc:
            detail = traceback.format_exc()
            path.write_text(json.dumps({'status': 'error', 'tasks': n, 'seed': s,
                                        'error': str(exc), 'traceback': detail.splitlines()},
                                       indent=2), encoding='utf-8')
            print(f'  FAILED: {exc}', flush=True)
    print('B&P CAMPAIGN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
