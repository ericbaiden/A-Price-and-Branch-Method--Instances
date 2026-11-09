#!/usr/bin/env python3
"""Export 24-hour pricing ablation from results/pb_ablation_h24 to CSV.

Reads the complete JSON campaign and produces a tidy seed-level CSV compatible
with plot_pb_ablation_h24.m and the LaTeX table generator.

Output: results/pb_ablation_h24/seed_level.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'results' / 'pb_ablation_h24'
OUT = SRC / 'seed_level.csv'

VARIANTS = ['A*+filter', 'A* no filter', 'Dijkstra+filter', 'Dijkstra no filter']


def main():
    rows = []
    for p in sorted(SRC.glob('tasks_*.json')):
        if not p.is_file():
            continue
        rec = json.loads(p.read_text(encoding='utf-8'))
        if rec.get('status') != 'complete':
            print(f'WARNING: {p.name} status={rec.get("status")}')
            continue
        n = rec['tasks']
        s = rec['seed']
        for v in VARIANTS:
            r = rec['algorithms'].get(v)
            if not r:
                continue
            rows.append({
                'tasks': n,
                'seed': s,
                'variant': v,
                'objective': r['obj'],
                'solve_time': r['solve_time'],
                'pricing_calls': r['total_pricing_calls'],
                'filter_reduction_mean_pct': r.get('filter_reduction_mean_pct', ''),
            })
    rows.sort(key=lambda x: (x['tasks'], x['seed'], VARIANTS.index(x['variant'])))
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['tasks', 'seed', 'variant', 'objective',
                                          'solve_time', 'pricing_calls',
                                          'filter_reduction_mean_pct'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {OUT}')


if __name__ == '__main__':
    main()
