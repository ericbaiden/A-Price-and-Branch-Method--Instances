#!/usr/bin/env python3
"""Export the canonical M=600 records the manuscript figures need.

Sources are the frozen campaigns, never a rerun:
  results/alns_m600/                       main P&B / GRASP / ALNS comparison, 10 seeds
  results/pb_ablation_m600_2x2_20261015/   4-variant pricing ablation, 10 seeds

Writes tidy CSVs under results/figure_data/ so the MATLAB script in
figuredraw/ only has to read tables.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'figure_data'
MAIN = ROOT / 'results' / 'alns_m600_final2'
OUT.mkdir(parents=True, exist_ok=True)

SCALES = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)


def export_main_comparison():
    rows = []
    for n in SCALES:
        for seed in range(10):
            path = MAIN / f'tasks_{n:04d}_seed_{seed:02d}.json'
            rec = json.loads(path.read_text(encoding='utf-8'))
            for method in ('P&B', 'GRASP', 'ALNS'):
                a = rec['algorithms'][method]
                rows.append({
                    'tasks': n, 'seed': seed, 'method': method,
                    'objective': a['obj'],
                    'completed': a['completed'],
                    'completion_pct': 100.0 * a['completed'] / n,
                    'solve_time_s': a['solve_time'],
                    'gap_to_bound_pct': a.get('gap_to_certified_bound_pct', ''),
                })
    write('main_comparison_m600.csv', rows,
          ['tasks', 'seed', 'method', 'objective', 'completed',
           'completion_pct', 'solve_time_s', 'gap_to_bound_pct'])


def export_certification():
    rows = []
    for n in SCALES:
        for seed in range(10):
            path = MAIN / f'tasks_{n:04d}_seed_{seed:02d}.json'
            p = json.loads(path.read_text(encoding='utf-8'))['algorithms']['P&B']
            root = p['solve_time'] - p['integer_recovery_time']
            rows.append({
                'tasks': n, 'seed': seed,
                'certification_s': p['certification_time'],
                'root_s': root,
                'certification_share_pct': 100.0 * p['certification_time'] / root,
                'fractional_flows': p['fractional_flows'],
                'integer_recovery_s': p['integer_recovery_time'],
            })
    write('certification_m600.csv', rows,
          ['tasks', 'seed', 'certification_s', 'root_s',
           'certification_share_pct', 'fractional_flows', 'integer_recovery_s'])


def export_ablation():
    src = ROOT / 'results' / 'pb_ablation_m600_2x2_20261015' / 'seed_level.csv'
    rows = list(csv.DictReader(src.open(encoding='utf-8')))
    keep = ['tasks', 'seed', 'variant', 'solve_time', 'pricing_calls',
            'objective', 'completed', 'filter_reduction_mean_pct']
    out = [{k: r[k] for k in keep} for r in rows]
    write('ablation_m600_2x2.csv', out, keep)


def write(name, rows, fields):
    dest = OUT / name
    with dest.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'  {name}: {len(rows)} rows -> {dest}')


def export_validation():
    """Reduced-constellation validation: the four methods against the full ILP."""
    src = ROOT / 'results' / 'revision_audit_m200' / 'validation'
    name_map = {
        'CG+A* (filter,k=1)': 'P&B',
        'GRASP': 'GRASP',
        'Greedy': 'Greedy',
        'Full 0-1 ILP': 'Full ILP',
    }
    rows = []
    for f in sorted(src.glob('tasks_*.json')):
        rec = json.loads(f.read_text(encoding='utf-8'))
        if rec.get('status') not in (None, 'complete'):
            continue
        for raw, label in name_map.items():
            alg = rec.get('algorithms', {}).get(raw)
            if not alg:
                continue
            rows.append({
                'tasks': rec['tasks'], 'seed': rec['seed'], 'method': label,
                'objective': alg['obj'],
                'completed': alg.get('completed', ''),
                'solve_time_s': alg.get('solve_time', ''),
            })
    rows.sort(key=lambda r: (r['tasks'], r['seed'], r['method']))
    write('validation_reduced.csv', rows,
          ['tasks', 'seed', 'method', 'objective', 'completed', 'solve_time_s'])



if __name__ == '__main__':
    print('Exporting figure data from the frozen M=600 campaigns')
    export_main_comparison()
    export_certification()
    export_ablation()
    export_validation()
