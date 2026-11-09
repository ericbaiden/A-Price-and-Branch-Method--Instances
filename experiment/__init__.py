"""Unified entry point for the experiments reported in the manuscript.

    python -m experiment --list
    python -m experiment main_comparison --dry-run

Configuration lives in `experiment.config`; `experiment.suites` adapts it to the
runner modules. Historical campaigns and repair scripts are kept in `archive/`.
"""

__all__ = ['config', 'common', 'suites', 'cli']
