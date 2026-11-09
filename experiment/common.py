"""Helpers shared by the experiment suites.

These are copies of functions the historical runners each carried in their own file.
They live here so a new suite does not have to import a runner just to write JSON.

`source_hashes` deliberately differs from the historical version: it also covers
`experiment/**`, so a campaign launched through this package records the code that
actually drove it. Campaigns launched from the archived runners keep their own
(unchanged) fingerprints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import ROOT


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def parse_list(value: str):
    """`'100,500,1000'` -> `[100, 500, 1000]`."""
    return [int(v) for v in value.split(',') if v.strip()]


def source_hashes(include_experiment: bool = True) -> dict:
    """SHA-256 of every file whose content can change a result."""
    paths = [ROOT / 'experiments.py', ROOT / 'alns_validation.py',
             ROOT / 'instance_store.py']
    paths += sorted((ROOT / 'Solver').glob('*.py'))
    paths += sorted((ROOT / 'Generator').glob('*.py'))
    paths += sorted((ROOT / 'Solver/native').glob('*.cpp'))
    paths += sorted((ROOT / 'Solver/native').glob('*.dll'))
    paths += sorted((ROOT / 'data/tle').glob('*.tle'))
    if include_experiment:
        paths += sorted((ROOT / 'experiment').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def fingerprint(hashes: dict | None = None) -> str:
    hashes = source_hashes() if hashes is None else hashes
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def load_dataset_instance(tasks: int, seed: int):
    """Load a frozen instance. The dataset is the only supported source."""
    from instance_store import load_instance
    return load_instance(tasks, seed)


def describe(name: str) -> str:
    from .config import SUITES
    s = SUITES[name]
    return (f'{name}\n'
            f'  {s["title"]}\n'
            f'  manuscript : {s["manuscript"]}\n'
            f'  scales     : {s["scales"]}\n'
            f'  seeds      : {s["seeds"]}\n'
            f'  output     : {s["output"]}\n'
            f'  notes      : {s["notes"]}')
