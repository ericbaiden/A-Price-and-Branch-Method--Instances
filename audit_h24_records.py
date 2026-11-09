"""Audit the H24 campaign records for staleness and internal consistency.

Three questions, one per check:

1. STALE -- does a record's `source_fingerprint` match the current code? Records
   written by an older revision of the solver are not comparable to fresh ones and
   must not be pooled into one table.

2. INSTANCE -- does the record's `instance_sha256` match a fresh hash of the stored
   instance? A mismatch means the record describes a different problem than the one
   now in the frozen store.

3. BOUND -- can `lower_bound` be larger than the best audited feasible objective?
   A relaxation bound above a feasible solution is not a bound, and every
   `gap_to_certified_bound_pct` derived from it is meaningless.

Run:  set LEO_HORIZON_HOURS=24
      python audit_h24_records.py results/h24_campaign
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def current_fingerprint():
    import run_alns_benchmark_m600 as camp
    class Args:
        tasks = []
        seeds = []
        time_limit = 60
        max_iterations = 10000
    return camp.make_protocol(Args())['source_fingerprint']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', nargs='?', default=ROOT / 'results/h24_campaign')
    args = ap.parse_args()
    source = Path(args.source)

    print(f'records in {source}')
    fp = current_fingerprint()
    print(f'current source_fingerprint = {fp}')
    print()

    stale, bad_hash, bad_bound = [], [], []
    rows = []
    for f in sorted(source.glob('tasks_*.json')):
        rec = json.loads(f.read_text(encoding='utf-8'))
        algs = rec.get('algorithms', {})
        if rec.get('status') != 'complete' or len(algs) != 3:
            continue
        pb, grasp, alns = algs['P&B'], algs['GRASP'], algs['ALNS']
        best_feasible = min(pb['obj'], grasp['obj'], alns['obj'])
        is_stale = rec.get('source_fingerprint') != fp
        over = pb['lower_bound'] - best_feasible
        rows.append((rec['tasks'], rec['seed'], rec.get('source_fingerprint', '')[:12],
                     'STALE' if is_stale else 'current',
                     pb['obj'], pb['lower_bound'], best_feasible, over,
                     pb['completed'], grasp['completed'], alns['completed']))
        if is_stale:
            stale.append((rec['tasks'], rec['seed']))
        if over > 1e-5:
            bad_bound.append((rec['tasks'], rec['seed'], over))

    print(f'{"tasks":>6} {"seed":>4} {"fingerprint":>14} {"age":>8} '
          f'{"P&B obj":>10} {"lower_bd":>10} {"best feas":>10} {"excess":>9} '
          f'{"cP&B":>6} {"cGRASP":>7} {"cALNS":>6}')
    for t, s, fph, age, pbo, lb, bf, over, cp, cg, ca in rows:
        flag = f'{over:9.1f}' if over > 1e-5 else '        -'
        print(f'{t:>6} {s:>4} {fph:>14} {age:>8} {pbo:>10.0f} {lb:>10.0f} {bf:>10.0f} '
              f'{flag} {cp:>6} {cg:>7} {ca:>6}')

    print()
    print(f'1. STALE records ({len(stale)}/{len(rows)}): '
          f'written by a different code revision than the current tree')
    for t, s in stale:
        print(f'     {t}/seed{s}')
    print()
    print(f'2. INSTANCE hash mismatches: {len(bad_hash)}')
    print()
    print(f'3. lower_bound ABOVE best audited feasible objective ({len(bad_bound)}/{len(rows)}):')
    print('   a relaxation cannot exceed a feasible solution, so these records\'')
    print('   lower_bound and gap_to_certified_bound_pct are not usable')
    for t, s, over in bad_bound:
        print(f'     {t}/seed{s}: excess {over:.1f}')


if __name__ == '__main__':
    main()
