"""Run frozen author profiles sequentially on GPU 0 and publish incremental reports."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from integrations.forecast_compare.author_profiles import ROOT, DEFAULT_PROFILES, load_author_profiles
from malr.evaluation.report import file_sha256
from scripts.report_author_p4_comparison import write_comparison

PYTHON = '/home/xyw/miniforge3/envs/malr/bin/python'
S4M_PYTHON = '/home/xyw/miniforge3/envs/malr-s4m/bin/python'
MASKS = ROOT.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def _atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def commands(name, directory):
    if name == 'E23_v12':
        common = ['--protocol-config', str(ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
                  '--e23-config', str(ROOT / 'configs/e23_multi_expert_p4_width64_window_v12.yaml'),
                  '--migration-catalog', str(ROOT.parent / 'E22/migration/paper_catalog.jsonl'),
                  '--mask-artifacts', str(MASKS), '--device', 'cuda:0']
        return [
            [PYTHON, '-m', 'integrations.e23.run', 'train-repair', *common, '--output', str(directory / 'stage1')],
            [PYTHON, '-m', 'integrations.e23.run', 'train-forecast', *common, '--output', str(directory / 'stage2'),
             '--stage1-manifest', str(directory / 'stage1/runs/e23-stage1.json'),
             '--forecast-loss', 'mse_plus_0p2_mae', '--selection-metric', 'mae'],
            [PYTHON, str(ROOT / 'scripts/evaluate_e23_three_test_draws.py'),
             '--selection', str(directory / 'stage2/runs/e23-stage2.json'),
             '--output', str(directory / 'three-test-draws.json')],
        ]
    if name == 'S4M':
        base = [S4M_PYTHON, str(ROOT / 'scripts/run_author_s4m_p4.py'), '--output', str(directory)]
    else:
        base = [PYTHON, str(ROOT / 'scripts/run_author_aligned_p4.py'), '--model', name, '--output', str(directory)]
    return [[*base, '--train'], [*base, '--evaluate-selected']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--table', type=Path, default=ROOT / 'docs/experiments/ETTh1_P4_author_three_test_draws_2026-10-02.csv')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve()
    names = ['E23_v12', *load_author_profiles()['models']]
    if args.dry_run:
        print(json.dumps({name: commands(name, args.output / name) for name in names}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep this entire run tied to the source used at launch. Stop on later edits.
    files = [DEFAULT_PROFILES, ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml',
             ROOT / 'configs/e23_multi_expert_p4_width64_window_v12.yaml',
             *sorted((ROOT / 'scripts').glob('*author*.py')),
             ROOT / 'scripts/evaluate_e23_three_test_draws.py',
             *sorted((ROOT / 'integrations').rglob('*.py')), *sorted((ROOT / 'src').rglob('*.py'))]
    frozen = {str(path): file_sha256(path) for path in files}
    _atomic(args.output / 'frozen_sources.json', frozen)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
    env['PATH'] = '/home/xyw/miniforge3/envs/malr/bin:' + env.get('PATH', '')
    env['WANDB_MODE'] = 'disabled'
    for name in names:
        directory = args.output / name; directory.mkdir()
        status = {'state': 'running', 'model': name, 'started_utc': datetime.now(timezone.utc).isoformat()}
        _atomic(directory / 'progress.json', status)
        write_comparison(args.output, args.table)
        try:
            for index, command in enumerate(commands(name, directory)):
                if any(file_sha256(Path(path)) != expected for path, expected in frozen.items()):
                    raise RuntimeError('frozen experiment code/config changed; refusing mixed-version results')
                _atomic(directory / f'command_{index}.json', command)
                print(f'{name} command {index} started', flush=True)
                with (directory / f'command_{index}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            status['state'] = 'completed'
        except Exception as exc:
            status.update(state='failed', error=str(exc))
            raise
        finally:
            status['finished_utc'] = datetime.now(timezone.utc).isoformat()
            _atomic(directory / 'progress.json', status)
            write_comparison(args.output, args.table)
        print(f'{name} completed', flush=True)


if __name__ == '__main__':
    main()
