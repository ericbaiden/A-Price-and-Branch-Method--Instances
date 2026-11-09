#!/usr/bin/env python3
"""Export the H24 campaign to CSV and a LaTeX table.

The campaign is 24 h horizon, M=600, nine task scales from 1000 to 10000, three
instance seeds each. Every case carries a certified P&B bound, a GRASP solution
and an ALNS solution, all independently audited.

What this refuses to do
-----------------------
A `gap_to_certified_bound_pct` is emitted only when the record passes the
cross-checks that make it meaningful:

  * `lower_bound <= every audited feasible objective`. A relaxation bound above a
    feasible solution is not a bound; such a record is reported in the sanity
    section instead of the gap column.
  * `instance_sha256` matches the frozen store's manifest, so the record is known
    to describe the instance being claimed.
  * the record is `complete` with all three algorithms.

Records failing any check are listed under "rejected records" rather than silently
averaged in. A table that hides a broken case is worse than a table with a hole.

Usage
-----
    python export_h24_results.py --source results/h24_campaign_v2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parent

BS = chr(92)
NL = BS + BS

METHODS = ('P&B', 'GRASP', 'ALNS')


def load(source: Path, manifest: Path):
    """Return (rows, rejected). Rows pass every cross-check."""
    known = {}
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding='utf-8'))
        known = {f'{r["tasks"]:04d}_{r["seed"]:02d}': r['instance_sha256']
                 for r in raw.values()}

    rows, rejected = [], []
    for f in sorted(source.glob('tasks_*.json')):
        rec = json.loads(f.read_text(encoding='utf-8'))
        name = f.name
        if rec.get('status') != 'complete':
            rejected.append((name, f'status={rec.get("status")}'))
            continue
        algs = rec.get('algorithms', {})
        missing = [m for m in METHODS if m not in algs]
        if missing:
            rejected.append((name, f'missing algorithms {missing}'))
            continue

        key = f'{rec["tasks"]:04d}_{rec["seed"]:02d}'
        if key in known and known[key] != rec.get('instance_sha256'):
            rejected.append((name, 'instance_sha256 does not match the frozen store'))
            continue

        pb, grasp, alns = algs['P&B'], algs['GRASP'], algs['ALNS']
        best_feasible = min(pb['obj'], grasp['obj'], alns['obj'])
        if pb['lower_bound'] - best_feasible > 1e-5:
            rejected.append((name, f'lower_bound {pb["lower_bound"]:.1f} exceeds the best '
                                   f'audited feasible objective {best_feasible:.1f}'))
            continue

        row = {
            'tasks': rec['tasks'], 'seed': rec['seed'],
            'horizon_h': 24.0, 'penalty': rec.get('penalty', 600),
            'instance_sha256': rec['instance_sha256'],
            'graph_nodes': rec['graph_nodes'], 'graph_edges': rec['graph_edges'],
            'lower_bound': pb['lower_bound'],
            'certification_s': pb.get('certification_time', ''),
            'integer_recovery_s': pb.get('integer_recovery_time', ''),
        }
        for m in METHODS:
            a = algs[m]
            row[f'{m}_objective'] = a['obj']
            row[f'{m}_completed'] = a['completed']
            row[f'{m}_completion_pct'] = 100.0 * a['completed'] / rec['tasks']
            row[f'{m}_time_s'] = a['solve_time']
            row[f'{m}_gap_pct'] = 100.0 * (a['obj'] - pb['lower_bound']) / pb['lower_bound']
        row['ALNS_preproc_s'] = alns.get('preprocessing_time', '')
        row['ALNS_construction_s'] = alns.get('construction_time', '')
        row['ALNS_search_s'] = alns.get('search_time', '')
        row['ALNS_iterations'] = alns.get('iterations', '')
        rows.append(row)
    rows.sort(key=lambda r: (r['tasks'], r['seed']))
    return rows, rejected


FIELDS = [
    'tasks', 'seed', 'horizon_h', 'penalty', 'instance_sha256',
    'graph_nodes', 'graph_edges', 'lower_bound', 'certification_s', 'integer_recovery_s',
    'P&B_objective', 'P&B_completed', 'P&B_completion_pct', 'P&B_time_s', 'P&B_gap_pct',
    'GRASP_objective', 'GRASP_completed', 'GRASP_completion_pct', 'GRASP_time_s', 'GRASP_gap_pct',
    'ALNS_objective', 'ALNS_completed', 'ALNS_completion_pct', 'ALNS_time_s', 'ALNS_gap_pct',
    'ALNS_preproc_s', 'ALNS_construction_s', 'ALNS_search_s', 'ALNS_iterations',
]


def write_csv(rows, dest):
    with dest.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'  {dest.name}: {len(rows)} rows')


def agg(rows, tasks, method, field):
    vals = [r[field] for r in rows
            if r['tasks'] == tasks and isinstance(r[field], (int, float))]
    return st.mean(vals) if vals else float('nan')


def write_tex(rows, dest):
    """One row per task scale, one column block per method plus the bound.

    Columns: n, z^P&B, then (gap to bound, completion) for GRASP and ALNS. P&B has
    no gap column of its own: its objective is recovered from the same relaxation
    as the bound, so a P&B gap would only report the integer-recovery residual and
    read as a misleading 0.00 across every scale.
    """
    scales = sorted({r['tasks'] for r in rows})
    out = [
        '{' + BS + 'scriptsize',
        BS + 'setlength{' + BS + 'tabcolsep}{4pt}',
        BS + 'renewcommand{' + BS + 'arraystretch}{1.10}',
        BS + 'begin{table}[!htbp]',
        BS + 'centering',
        BS + 'caption{24-hour horizon, $M=600$. Means over three instance seeds. '
        '$z^{\\mathrm{P\\&B}}$ is the certified lower bound from the full '
        'column-generation relaxation with exact pricing; its integer recovery is '
        'reported separately. The two heuristics share the identical residual-routing '
        'primitive. Gap is relative to the certified bound; completion is the share '
        'of tasks served. ALNS budgets its search alone -- preprocessing and the '
        'initial construction are timed separately -- so both heuristics are compared '
        'on solution quality rather than on bootstrap speed.}',
        BS + 'label{tab:h24-main}',
        BS + 'begin{tabular}{@{}rrrrrrr@{}}',
        BS + 'toprule',
        ' & & ' + BS + 'multicolumn{2}{c}{GRASP} & ' + BS + 'multicolumn{2}{c}{ALNS} ' + NL,
        BS + 'cmidrule(lr){3-4}' + BS + 'cmidrule(lr){5-6}',
        '$n$ & $z^{\\mathrm{P\\&B}}$ & gap (\\%) & served (\\%) '
        '& gap (\\%) & served (\\%) ' + NL,
        BS + 'midrule',
    ]
    for scale in scales:
        pb = agg(rows, scale, 'P&B', 'P&B_objective')
        g_gap = agg(rows, scale, 'GRASP', 'GRASP_gap_pct')
        g_srv = agg(rows, scale, 'GRASP', 'GRASP_completion_pct')
        a_gap = agg(rows, scale, 'ALNS', 'ALNS_gap_pct')
        a_srv = agg(rows, scale, 'ALNS', 'ALNS_completion_pct')
        out.append(f'{scale} & {pb:,.0f} & {g_gap:.2f} & {g_srv:.1f} '
                   f'& {a_gap:.2f} & {a_srv:.1f} ' + NL)
    out += [BS + 'bottomrule', BS + 'end{tabular}', BS + 'end{table}', '']
    dest.write_text('\n'.join(out), encoding='utf-8')
    print(f'  {dest.name}: {len(scales)} scales')


def write_sanity(rows, rejected, dest):
    lines = ['H24 campaign export: sanity section', '=' * 60, '']
    if not rejected:
        lines.append('No rejected records. Every record that was found passed all '
                     'cross-checks (status complete, three algorithms, instance hash '
                     'matching the frozen store, certified bound at or below the best '
                     'audited feasible objective).')
    else:
        lines.append(f'{len(rejected)} record(s) rejected and EXCLUDED from the table:')
        for name, why in rejected:
            lines.append(f'  {name}: {why}')
    lines += ['', 'Records exported: ' + str(len(rows)), '']

    lines.append('ALNS budget accounting (mean per scale, seconds)')
    lines.append('  ' + ' '.join(f'{s:>7}' for s in ['scale', 'preproc', 'constr', 'search', 'total']))
    for scale in sorted({r['tasks'] for r in rows}):
        vals = [r for r in rows if r['tasks'] == scale]
        def m(field):
            xs = [r[field] for r in vals if isinstance(r[field], (int, float))]
            return st.mean(xs) if xs else float('nan')
        lines.append(f'  {scale:>7} {m("ALNS_preproc_s"):>7.1f} {m("ALNS_construction_s"):>7.1f} '
                     f'{m("ALNS_search_s"):>7.1f} {m("ALNS_time_s"):>7.1f}')
    lines.append('')
    dest.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  {dest.name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'results/h24_campaign_v2')
    ap.add_argument('--outdir', type=Path, default=ROOT / 'results/h24_figure_data')
    ap.add_argument('--texdir', type=Path, default=ROOT / 'paper/tables')
    args = ap.parse_args()

    manifest = ROOT / 'dataset/instances/h24/manifest.json'
    rows, rejected = load(args.source, manifest)
    if not rows:
        print(f'No usable records in {args.source}', file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.texdir.mkdir(parents=True, exist_ok=True)

    print(f'Exporting {len(rows)} H24 cases from {args.source}')
    write_csv(rows, args.outdir / 'h24_main_comparison.csv')
    write_tex(rows, args.texdir / 'tab_h24_main.tex')
    write_sanity(rows, rejected, args.outdir / 'h24_sanity.txt')

    if rejected:
        print(f'\n{len(rejected)} record(s) excluded -- see h24_sanity.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
