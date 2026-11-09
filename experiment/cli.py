"""One command for every experiment in the manuscript.

    python -m experiment --list
    python -m experiment main_comparison --dry-run
    python -m experiment main_comparison
    python -m experiment pricing_ablation --scales 100 500
    python -m experiment heterogeneous --seeds 0 1 2

Defaults come from `experiment.config`; the flags only exist to run a subset (for a
smoke test) or to point at a scratch output directory.
"""
from __future__ import annotations

import argparse
import sys

from .common import describe
from .config import SUITES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m experiment',
        description='Run the experiments reported in the manuscript.',
    )
    p.add_argument('suite', nargs='?', help='experiment to run; omit with --list')
    p.add_argument('--list', action='store_true', help='list the available experiments')
    p.add_argument('--dry-run', action='store_true',
                   help='print the resolved configuration and exit')
    p.add_argument('--scales', type=int, nargs='+', help='override the task counts')
    p.add_argument('--seeds', type=int, nargs='+', help='override the seeds')
    p.add_argument('--output', help='override the output directory')
    return p


def main(argv=None) -> int:
    # Unknown flags are forwarded to the runner verbatim (e.g. --time-limit).
    args, extra = build_parser().parse_known_args(argv)

    if args.list or not args.suite:
        for name in SUITES:
            print(describe(name))
            print()
        return 0

    if args.suite not in SUITES:
        print(f'Unknown experiment {args.suite!r}. Known: {", ".join(SUITES)}',
              file=sys.stderr)
        return 2

    cfg = SUITES[args.suite]
    scales = tuple(args.scales) if args.scales else cfg['scales']
    seeds = tuple(args.seeds) if args.seeds else cfg['seeds']
    output = args.output or cfg['output']

    print(f'experiment : {args.suite}')
    print(f'  {cfg["title"]}')
    print(f'  manuscript: {cfg["manuscript"]}')
    print(f'  runner    : {cfg["runner"]}')
    print(f'  scales    : {scales}')
    print(f'  seeds     : {seeds}')
    print(f'  output    : {output}')
    if extra:
        print(f'  extra     : {" ".join(extra)}')

    if args.dry_run:
        print('\n(dry run: nothing executed)')
        return 0

    from .suites import dispatch
    return dispatch(args.suite, scales=scales, seeds=seeds,
                    output=output, extra=extra)


if __name__ == '__main__':
    sys.exit(main())
