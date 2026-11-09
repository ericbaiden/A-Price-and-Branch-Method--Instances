"""Single source of truth for every experiment that appears in the manuscript.

Nothing in `experiment/` reads a command-line default that disagrees with this file:
the CLI is a thin shell over `SUITES`, and each suite translates its entry here into
whatever the underlying runner expects.

Adding an experiment means adding one entry to `SUITES` and one thin module in
`experiment/suites/`. Historical campaigns and one-off repair scripts live in
`archive/`; they are provenance, not part of the testbed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Shared protocol values
# ---------------------------------------------------------------------------
# The service-first penalty selected in Section 5.3. Every reported comparison uses it.
PENALTY = 600

# Task counts: one instance every 100 tasks, ten seeds each (Section 5.1.1).
SCALES = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
SEEDS = tuple(range(10))

# Heuristic budgets fixed in Section 5.1.3.
ALNS_TIME_LIMIT = 60.0
ALNS_MAX_ITERATIONS = 10_000
GRASP_RESTARTS = 20

# Frozen instances. Nothing is regenerated: every suite loads from here.
DATASET = ROOT / 'dataset' / 'instances'
RESULTS = ROOT / 'results'

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# runner      module in the repository root that implements the experiment
# anchors     manuscript anchors this experiment supports
# scales/seeds  the configuration the manuscript reports
# output      where the campaign writes its records
SUITES = {
    'main_comparison': {
        'runner': 'run_alns_benchmark_m600',
        'title': 'Matched P&B / GRASP / ALNS comparison',
        'manuscript': 'Section 5.2.3 (Integer Solution Quality), Table 2, Figure 5',
        'scales': SCALES,
        'seeds': SEEDS,
        'output': RESULTS / 'alns_m600_final',
        'notes': 'Large-scale constellation, unit volume, M=600, ten seeds per scale.',
    },
    'pricing_ablation': {
        'runner': 'run_pb_ablation_m600',
        'title': 'Pricing acceleration ablation',
        'manuscript': 'Section 5.4 (Contribution of Pricing Accelerations), Figure 9',
        'scales': (100, 500, 1000),
        'seeds': SEEDS,
        'output': RESULTS / 'pb_ablation_m600_2x2_20261015',
        'notes': 'A*/Dijkstra x filter on/off; four variants, identical objectives.',
    },
    'heterogeneous': {
        'runner': 'run_heterogeneous_m600',
        'title': 'Robustness to task-volume heterogeneity',
        'manuscript': 'Section 5.2.4 (Robustness to Task-Volume Heterogeneity), Table 4',
        'scales': (100, 500, 1000),
        'seeds': SEEDS,
        'output': RESULTS / 'heterogeneous_m600',
        'notes': 'q_t drawn uniformly from {1,2,3} GB; M=600.',
    },
    'reduced': {
        'runner': 'run_reduced_m600',
        'title': 'Reduced-constellation validation',
        'manuscript': 'Section 5.2.1 (Reduced-Constellation Validation), Table 1',
        'scales': (50, 100),
        'seeds': SEEDS,
        'output': RESULTS / 'reduced_m600',
        'notes': '10-RS/50-CS constellation; Greedy vs GRASP vs P&B.',
    },
    'scalability': {
        'runner': 'run_scalability_selected_m',
        'title': 'Large-scale scaling against full Gurobi',
        'manuscript': 'Section 5.2 / supplementary scaling results',
        'scales': (2000, 3000, 5000, 10000),
        'seeds': (0,),
        'output': RESULTS / 'scalability_m600_20261014',
        'notes': 'Seed 0 only: each case is a multi-hour full-model solve.',
    },
    'm_sensitivity': {
        'runner': 'run_m_selection_diagnostic',
        'title': 'Rejection-penalty diagnosis and selection',
        'manuscript': 'Section 5.3 (Rejection-Penalty Sensitivity)',
        'scales': (500, 1000, 2000, 3000, 5000, 10000),
        'seeds': (0,),
        'output': RESULTS / 'm_selection_20261014_pilot',
        'notes': 'Exact cost frontier F(K)-F(K-1); yields the critical penalty M*.',
    },
}


def get(name: str) -> dict:
    if name not in SUITES:
        raise KeyError(f'Unknown experiment {name!r}. Known: {", ".join(SUITES)}')
    return SUITES[name]
