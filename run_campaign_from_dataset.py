#!/usr/bin/env python3
"""Run the M=600 comparison campaign against the frozen instance dataset.

`run_alns_benchmark_m600.py` regenerates every instance from a seed, which made the
result depend on the order instances were requested (see `instance_store`). This
wrapper monkeypatches `build_instance` to read from `dataset/instances/` instead, so
every run of every campaign sees exactly the same instance.

`run_alns_benchmark_m600.py` itself is not touched, which keeps its SHA-256 -- and
therefore every existing result record's `source_fingerprint` -- valid.

The campaigns still record `instance_sha256` per case, so results remain traceable to
the stored instance and any mismatch is caught immediately.

Usage
-----
    python run_campaign_from_dataset.py --tasks 100,200,300 --seeds 0,1,2 \
        --output results/alns_m600_final
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import run_alns_benchmark as bench
from instance_store import load_instance, has_instance, STORE, HORIZON_DEFAULT

_REAL_BUILD = bench.build_instance


# The horizon selects which stored instance family to load. It is a module-level
# constant rather than a CLI flag because the campaign runner takes its own argument
# parser; setting LEO_HORIZON_HOURS in the environment is enough to switch.
HORIZON = float(os.environ.get('LEO_HORIZON_HOURS', HORIZON_DEFAULT))


def build_from_store(seed, count, task_data_volumes=None):
    """Drop-in replacement for `bench.build_instance` backed by the frozen store."""
    if task_data_volumes is not None:
        raise NotImplementedError(
            'The store holds unit-volume instances; heterogeneous volumes are not '
            'supported by this wrapper yet.')
    if not has_instance(count, seed, HORIZON):
        raise FileNotFoundError(
            f'Instance ({count}, {seed}) is not in {STORE} at {HORIZON:g} h. '
            f'Run build_instance_dataset.py or build_horizon_dataset.py.')
    return load_instance(count, seed, HORIZON)


def canonical_graph_from_store(instance):
    """The store already holds the canonical graph; hand it back unchanged.

    `run_case` calls `canonical_graph(instance)`, which rebuilds the graph from the
    generator's temporal network. A stored instance has no temporal network, and
    rebuilding would defeat the point of the store, so the stored graph is returned.
    """
    return instance.graph


def install(module):
    """Point a campaign module at the store for the rest of the process."""
    module.build_instance = build_from_store
    module.canonical_graph = canonical_graph_from_store


if __name__ == '__main__':
    import run_alns_benchmark_m600 as campaign
    install(campaign)
    sys.argv[0] = 'run_alns_benchmark_m600.py'
    campaign.main()
