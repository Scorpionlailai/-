"""Frozen v31-B/T1 P4 comparison: five author seeds and twenty test masks."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from integrations.forecast_compare.author_provenance import write_new_record
from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.replicate_statistics import TRAINING_SEEDS
from integrations.forecast_compare.test_mask_bank_v2 import create_test_bank, TEST_SEEDS
from malr.evaluation.report import file_sha256
from scripts.summarize_e23_t1_five_seed import summarize

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def source_files():
    paths = []
    for folder in ['src', 'integrations', 'scripts', 'configs', 'tests']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths.append(ROOT/'pyproject.toml')
    return sorted(paths)


def verify_frozen(root):
    seal = root/'trial-definition.sha256'
    if seal.read_text().strip() != file_sha256(root/'trial-definition.json'):
        raise ValueError('trial definition seal changed')
    definition = json.loads((root/'trial-definition.json').read_text())
    current = {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files()}
    if current != definition['source_files']:
        raise ValueError('training/evaluation source changed after freeze')
    for path, digest in definition['run_configs'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('frozen run configuration changed')
    if file_sha256(root/'test_masks/manifest.json') != definition['test_bank_sha256']:
        raise ValueError('frozen test bank manifest changed')
    return definition


def prepare(root, device):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'trial-definition.json').exists():
        return verify_frozen(root)
    protocol = load_p4_protocol(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS)
    bank = create_test_bank(protocol, root/'test_masks')
    (root/'configs').mkdir(exist_ok=True)
    source_config = ROOT/'configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml'
    author_config = ROOT/'configs/baselines/etth1_p4_author_20261002.json'
    configs = {}
    for seed in TRAINING_SEEDS:
        cfg = root/f'configs/e23_seed_{seed}.yaml'
        cfg.write_text(source_config.read_text().replace('seed: 2021', f'seed: {seed}'))
        configs[str(cfg.resolve())] = file_sha256(cfg)
        profile = json.loads(author_config.read_text())
        profile['protocol'].update(model_seed=seed, test='twenty new frozen test masks',
                                   extra_test_seeds=list(TEST_SEEDS),
                                   control_notes='Author paper five seeds; P4 forecasting extension; fixed original validation only.')
        path = root/f'configs/author_seed_{seed}.json'
        write_new_record(path, profile)
        configs[str(path.resolve())] = file_sha256(path)
    files = source_files()
    with tarfile.open(root/'frozen-source.tar.gz', 'w:gz') as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    definition = dict(schema_version=1, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                      model_names=['E23-v31-B', 'T1'], training_seeds=list(TRAINING_SEEDS),
                      test_mask_seeds=list(TEST_SEEDS), device=device, dataset='ETTh1', scenario='P4',
                      seq_len=96, pred_len=96, evaluation_windows=2688,
                      train_start_rate=.10, block_len=8, overlap=True,
                      fixed_validation_mask_sha256=protocol.mask_sha256['validation'],
                      original_test_mask_excluded=True, test_bank_sha256=file_sha256(bank),
                      selection='original fixed validation MAE; T1 repair selects MSE; E23 repair global MSE',
                      ours='selected v31-B; S1 global MSE + retrieval; S2 MSE+.2MAE+.05 uniform repair after epoch10',
                      T1='official ETTh1 repair architecture and training hyperparameters; shared original PatchTST forecast extension',
                      prediction_head='unchanged PatchTST D16, 35427 parameters; equal initial state within seed',
                      experimental_unit='5 independent complete training runs; 20 paired missing-position draws per checkpoint',
                      statistics='mean; sample std over mask-averaged training seeds and seed-averaged masks; paired crossed bootstrap 10000 draws',
                      resources='same physical GPU sequential; full entrypoint time incl imports/preparation/all stages/validation/saving; allocator allocated/reserved peaks',
                      paper_source='https://arxiv.org/html/2602.21043',
                      official_script_discrepancy='run.py uses seed2021 and itr1; ETTh1.sh trains each test mask rate, unlike paper fixed40% training. Paper seeds implemented explicitly.',
                      run_configs=configs, source_files={str(p.relative_to(ROOT)): file_sha256(p) for p in files},
                      source_archive_sha256=file_sha256(root/'frozen-source.tar.gz'))
    write_new_record(root/'trial-definition.json', definition)
    (root/'trial-definition.sha256').write_text(file_sha256(root/'trial-definition.json')+'\n')
    return definition


def commands(root, seed, model, device):
    base = root/f'seed_{seed}'/model
    if model == 'T1':
        training = [[sys.executable, 'scripts/run_author_aligned_p4.py', '--model', 'T1', '--train',
                     '--profiles', str(root/f'configs/author_seed_{seed}.json'),
                     '--training-seed', str(seed), '--device', device, '--output', str(base/'training')]]
        selection = base/'training/forecast-selection.json'
        evalcfg = ['--profiles', str(root/f'configs/author_seed_{seed}.json')]
    else:
        common = ['--protocol-config', str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
                  '--migration-catalog', str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),
                  '--mask-artifacts', str(MASKS), '--e23-config', str(root/f'configs/e23_seed_{seed}.yaml'),
                  '--training-seed', str(seed), '--device', device]
        training = [[sys.executable, '-m', 'integrations.e23.run', 'train-repair', *common,
                     '--output', str(base/'stage1')],
                    [sys.executable, '-m', 'integrations.e23.run', 'train-forecast', *common,
                     '--stage1-manifest', str(base/'stage1/runs/e23-stage1.json'),
                     '--selection-metric', 'mae', '--forecast-loss', 'mse_plus_0p2_mae',
                     '--repair-retention', 'uniform_v1', '--output', str(base/'stage2')]]
        selection = base/'stage2/runs/e23-stage2.json'
        evalcfg = ['--e23-config', str(root/f'configs/e23_seed_{seed}.yaml')]
    evaluation = [sys.executable, 'scripts/evaluate_e23_t1_replicate.py', '--model', model,
                  '--training-seed', str(seed), '--selection', str(selection), *evalcfg,
                  '--test-mask-bank', str(root/'test_masks/manifest.json'),
                  '--output', str(base/'evaluation'), '--device', device]
    return training, evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    definition = prepare(root, args.device)
    if args.device != definition['device']:
        raise ValueError('frozen device mismatch')
    if args.prepare_only:
        print(json.dumps(dict(root=str(root), status='prepared', runs=10, test_masks=20)))
        return
    root.joinpath('logs').mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': 'src:.', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    for seed in TRAINING_SEEDS:
        for model in definition['model_names']:
            final = root/f'seed_{seed}'/model/'evaluation/twenty-test-draws.json'
            if final.exists():
                summarize(root)
                continue
            verify_frozen(root)
            training, evaluation = commands(root, seed, model, args.device)
            steps = training+[evaluation]
            for index, command in enumerate(steps):
                # Resume only completed immutable stages. Partial stages require investigation.
                if command[1:4] == ['-m', 'integrations.e23.run', 'train-repair']:
                    selected = root/f'seed_{seed}'/model/'stage1/runs/e23-stage1.json'
                elif command[1:4] == ['-m', 'integrations.e23.run', 'train-forecast']:
                    selected = root/f'seed_{seed}'/model/'stage2/runs/e23-stage2.json'
                elif index == 0 and model == 'T1':
                    selected = root/f'seed_{seed}'/model/'training/forecast-selection.json'
                else:
                    selected = final
                if selected.exists():
                    continue
                log = root/f'logs/{model}_seed_{seed}_step_{index}.log'
                print(json.dumps(dict(event='start', seed=seed, model=model, step=index,
                                      log=str(log), utc=datetime.now(timezone.utc).isoformat())), flush=True)
                with log.open('x') as stream:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                verify_frozen(root)
            status = summarize(root)
            print(json.dumps(dict(event='replicate_completed', seed=seed, model=model,
                                  completed_runs=status['completed_runs'])), flush=True)
    status = summarize(root)
    if status['status'] != 'complete':
        raise RuntimeError('formal comparison is incomplete')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
