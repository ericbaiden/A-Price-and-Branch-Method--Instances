"""Persist the immutable physical contact layer so it is built once per horizon.

`run_alns_benchmark.build_instance` caches the physical layer in a module-level
`PHYSICAL_CACHE`, which lives only as long as the process. Building that layer at a
long planning horizon costs minutes -- 874 s at 24 hours against about 100 s at three
hours -- and every new process would pay it again.

This module writes the template to disk, keyed by horizon and by the generator
fingerprint, so a later process loads it instead of rebuilding. The template holds
only orbit-determined contact nodes and arcs; task layers are stripped by the
caller before saving, exactly as the in-process cache does.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'dataset' / 'physical'


def _key(horizon_hours: float, rs: int, cs: int) -> str:
    """Cache name. Includes the horizon: a template is invalid across horizons."""
    return f'template_h{horizon_hours:g}_rs{rs}_cs{cs}.pkl'


def _fingerprint() -> str:
    """Hash the generator sources that determine the contact layer."""
    import hashlib as _h
    d = _h.sha256()
    for p in sorted((ROOT / 'Generator').glob('*.py')):
        d.update(p.read_bytes())
    data = ROOT / 'data' / 'tle'
    if data.is_dir():
        for p in sorted(data.glob('*.tle')):
            d.update(p.read_bytes())
    return d.hexdigest()


def load(horizon_hours: float, rs: int, cs: int):
    """Return a cached template, or None when absent or built by other sources."""
    path = CACHE / _key(horizon_hours, rs, cs)
    if not path.is_file():
        return None
    try:
        with path.open('rb') as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if payload.get('fingerprint') != _fingerprint():
        return None
    return payload.get('template')


def save(template, horizon_hours: float, rs: int, cs: int) -> Path:
    """Write a stripped physical template for reuse by later processes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / _key(horizon_hours, rs, cs)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as fh:
        pickle.dump({'fingerprint': _fingerprint(), 'horizon_hours': horizon_hours,
                     'rs': rs, 'cs': cs, 'template': template}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path
