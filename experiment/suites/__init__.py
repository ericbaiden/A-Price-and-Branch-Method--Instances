"""Thin adapters from `experiment.config` to the runners that implement each suite.

Each suite does exactly one thing: translate the configuration into the arguments its
runner understands, then hand control to that runner. No solver logic is duplicated
here -- the runners stay the single implementation, and are called rather than copied.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ..config import SUITES, DATASET, PENALTY


def _require_dataset():
    if not DATASET.is_dir() or not any(DATASET.glob('tasks_*.npz')):
        raise RuntimeError(
            f'No frozen instances in {DATASET}. Run build_instance_dataset.py first.')


def _run_module(name: str, argv: list[str]) -> int:
    """Import `name` from the repository root and call its main() with `argv`."""
    saved = sys.argv
    sys.argv = [f'{name}.py'] + argv
    try:
        module = importlib.import_module(name)
        result = module.main()
        return 0 if result is None else int(result)
    finally:
        sys.argv = saved


def _relative_output(output) -> str:
    """Some runners join `--output` onto ROOT, so an absolute path would be mangled."""
    from ..config import ROOT
    p = Path(output)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def main_comparison(*, scales, seeds, output, extra=()):
    """Section 5.2.3: matched P&B / GRASP / ALNS comparison."""
    _require_dataset()
    return _run_module('run_alns_benchmark_m600', [
        '--tasks', ','.join(map(str, scales)),
        '--seeds', ','.join(map(str, seeds)),
        '--output', _relative_output(output),
        *extra,
    ])


def pricing_ablation(*, scales, seeds, output, extra=()):
    """Section 5.4: A*/Dijkstra x filter on/off."""
    _require_dataset()
    return _run_module('run_pb_ablation_m600', [
        '--tasks', *map(str, scales),
        '--seeds', *map(str, seeds),
        '--output', _relative_output(output),
        *extra,
    ])


def heterogeneous(*, scales, seeds, output, extra=()):
    """Section 5.2.4: heterogeneous task volumes."""
    _require_dataset()
    return _run_module('run_heterogeneous_m600', [
        '--tasks', ','.join(map(str, scales)),
        '--seeds', ','.join(map(str, seeds)),
        '--output', _relative_output(output),
        *extra,
    ])


def reduced(*, scales, seeds, output, extra=()):
    """Section 5.2.1: reduced-constellation validation.

    `run_reduced_m600` has no CLI: its task counts and seeds are fixed at 50/100 and
    0--9, which is exactly what `config.SUITES['reduced']` declares. It is run as-is,
    and the configuration is checked rather than silently overridden.
    """
    if tuple(scales) != (50, 100) or tuple(seeds) != tuple(range(10)):
        raise ValueError(
            f'run_reduced_m600 is fixed at scales (50, 100) and seeds 0-9; '
            f'got scales={tuple(scales)} seeds={tuple(seeds)}')
    return _run_module('run_reduced_m600', list(extra))


def scalability(*, scales, seeds, output, extra=()):
    """Large-scale scaling against the full aggregate Gurobi model."""
    return _run_module('run_scalability_selected_m', [
        '--tasks', *map(str, scales),
        '--seeds', *map(str, seeds),
        '--output', str(output),
        *extra,
    ])


def m_sensitivity(*, scales, seeds, output, extra=()):
    """Section 5.3: exact penalty threshold diagnosis.

    The runner takes cases as `tasks:seed` pairs, so the configured scales and seeds
    are flattened here.
    """
    cases = [f'{n}:{s}' for n in scales for s in seeds]
    return _run_module('run_m_selection_diagnostic', [
        '--cases', *cases,
        '--output', str(output),
        *extra,
    ])


HANDLERS = {
    'main_comparison': main_comparison,
    'pricing_ablation': pricing_ablation,
    'heterogeneous': heterogeneous,
    'reduced': reduced,
    'scalability': scalability,
    'm_sensitivity': m_sensitivity,
}


def dispatch(name: str, *, scales=None, seeds=None, output=None, extra=()) -> int:
    if name not in SUITES:
        raise KeyError(f'Unknown experiment {name!r}. Known: {", ".join(SUITES)}')
    cfg = SUITES[name]
    handler = HANDLERS[name]
    return handler(
        scales=tuple(scales) if scales else cfg['scales'],
        seeds=tuple(seeds) if seeds else cfg['seeds'],
        output=output if output is not None else cfg['output'],
        extra=tuple(extra),
    )
