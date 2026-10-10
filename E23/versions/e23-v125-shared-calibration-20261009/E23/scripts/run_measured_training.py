"""Measure legacy Python training entrypoints inside their own training process.

Use for E22, SAITS, S4M, CRIB, CoIFNet, BiTGraph and ChannelTokenFormer without
editing their identity-checked source files.
Keep the target's original interpreter, working directory and PYTHONPATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from time import perf_counter

_PROCESS_STARTED = perf_counter()

from integrations.forecast_compare.resources import TrainingResources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--module')
    target.add_argument('--script', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--precision', default='float32')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('target_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.target_args[1:] if args.target_args[:1] == ['--'] else args.target_args
    script = args.script.resolve() if args.script else None
    with TrainingResources(args.output, device=args.device, batch_size=args.batch_size,
                           model=args.model, started_at=_PROCESS_STARTED,
                           precision=args.precision, amp=args.amp) as resources:
        with resources.stage(args.stage):
            sys.argv = [str(script) if script else args.module, *arguments]
            if script:
                sys.path.insert(0, str(script.parent))
            try:
                if script:
                    runpy.run_path(str(script), run_name='__main__')
                else:
                    runpy.run_module(args.module, run_name='__main__', alter_sys=True)
            except SystemExit as exit_result:
                if exit_result.code not in (0, None):
                    raise
        resources.selection_manifest = args.selection_manifest
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
