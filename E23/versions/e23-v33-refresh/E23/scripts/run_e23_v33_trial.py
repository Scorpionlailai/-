"""Fixed Stage1 matrix; unchanged pure-MSE Stage2 and three test draws per condition."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from scripts.run_e23_v30_trial import _execute, verify_freeze, ROOT, MASKS, BANK
from scripts.report_e23_v33_trial import CONFIGS, load_candidate, report, digest


def build_commands(root, device):
    rows = []
    for condition, config in CONFIGS.items():
        cell = root/condition
        common = ['--protocol-config', str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--e23-config', str(config), '--migration-catalog', str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),
            '--mask-artifacts', str(MASKS), '--device', device]
        stage1_manifest = (ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v29_iterative_tv_h_r3/stage1/runs/e23-stage1.json'
                           if condition == 'C0' else cell/'stage1/runs/e23-stage1.json')
        rows.append(dict(condition=condition, config=str(config), stage1_manifest=str(stage1_manifest),
            stage1=None if condition == 'C0' else [sys.executable, '-m', 'integrations.e23.run', 'train-repair', *common, '--output', str(cell/'stage1')],
            stage2=[sys.executable, '-m', 'integrations.e23.run', 'train-forecast', *common, '--output', str(cell/'stage2'),
                '--stage1-manifest', str(stage1_manifest), '--selection-metric', 'mae',
                '--forecast-loss', 'mse', '--forecast-input', 'explicit_v1', '--forecast-mixer', 'none']))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', default='cuda:1'); p.add_argument('--mode', choices=('train','report'), required=True)
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); root = a.root.resolve()
    commands = build_commands(root, a.device)
    if a.dry_run: print(json.dumps(commands, indent=2)); return
    verify_freeze(root)
    if a.mode == 'report': report(root, ROOT/'docs/experiments'); return
    if not a.resume and any((root/r['condition']/s).exists() for r in commands for s in ('stage1','stage2')):
        raise FileExistsError('v33 training outputs already exist')
    environment_path = root/('resume_training_environment.json' if a.resume else 'formal_training_environment.json')
    with environment_path.open('x') as stream:
        json.dump(dict(
        started_utc=datetime.now(timezone.utc).isoformat(), device=a.device, workspace=str(ROOT),
        gpu_processes_at_start=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
            '--format=csv,noheader'], text=True)), stream, indent=2)
    for row in commands:
        cell = root/row['condition']; cell.mkdir(exist_ok=True)
        if row['condition'] == 'C0' and not (cell/'stage1').exists():
            (cell/'stage1').symlink_to(Path(row['stage1_manifest']).parents[1],target_is_directory=True)
        for stage in ('stage1','stage2'):
            if row[stage] is not None:
                manifest = cell/stage/'runs'/f'e23-{stage}.json'
                if a.resume and manifest.exists():
                    saved = json.loads(manifest.read_text())
                    if saved['epochs_completed'] != 100 or saved['e23_config_sha256'] != digest(row['config']) or digest(saved['checkpoint']) != saved['checkpoint_sha256']:
                        raise ValueError('cannot resume mismatched/incomplete training stage')
                    print(json.dumps(dict(event='reuse_completed_stage',condition=row['condition'],stage=stage)),flush=True)
                    continue
                _execute(row[stage], cell/(stage+'.log'), root/'code-freeze.json'); verify_freeze(root)
        selection = cell/'stage2/runs/e23-stage2.json'
        if not (cell/'validation_replay.json').exists():
            _execute([sys.executable, 'scripts/replay_e23_p4_validation.py', '--e23-config', row['config'],
            '--stage2-manifest', str(selection), '--output', str(cell/'validation_replay.json'), '--device', a.device],
            cell/'validation_replay.log', root/'code-freeze.json')
        if not (cell/'round_diagnostics.json').exists():
            if (cell/'diagnostic.log').exists():
                (cell/'diagnostic.log').rename(cell/'diagnostic-failed-padding.log')
            _execute([sys.executable, 'scripts/diagnose_e23_v33.py', '--config', row['config'],
            '--cell', str(cell), '--device', a.device], cell/'diagnostic.log', root/'code-freeze.json')
        record = load_candidate(root, row['condition'])
        (cell/'locked-validation.json').write_text(json.dumps(record, indent=2)+'\n')
        if (cell/'three-test-draws.json').exists():
            continue
        started = perf_counter()
        _execute([sys.executable, 'scripts/evaluate_e23_three_test_draws.py', '--e23-config', row['config'],
            '--selection', str(selection), '--output', str(cell/'three-test-draws.json'),
            '--model-label', record['model'], '--test-mask-bank', str(BANK), '--device', a.device],
            cell/'three-test-draws.log', root/'code-freeze.json')
        (cell/'test_runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,
            training_time_included=False), indent=2)+'\n'); verify_freeze(root)
    report(root, ROOT/'docs/experiments')


if __name__ == '__main__': main()
