"""Author S4M profile using the existing protected-core bridge and environment."""

from time import perf_counter
_PROCESS_STARTED = perf_counter()

import argparse
from hashlib import sha1
import json
import os
from pathlib import Path
import sys
from types import MethodType

import torch
from torch.utils.data import DataLoader

from integrations.forecast_compare.author_profiles import ROOT, DEFAULT_PROFILES, load_author_profiles
from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from integrations.forecast_compare.three_draw_evaluation import evaluate_three_draws
from integrations.s4m.search import SearchCandidate, build_train_command, _replace_argument
from malr.evaluation.report import file_sha256
from integrations.forecast_compare.author_provenance import (
    author_source_identity as _code_identity, write_new_record as _write,
)

DEFAULT_MASKS = ROOT.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'
DEFAULT_BANK = ROOT / 'artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json'


def author_command(cfg, config, masks, output, *, mode):
    candidate = SearchCandidate(cfg['model']['e_layers'], cfg['training']['learning_rate'],
                                cfg['model']['dropout'], 'official')
    command = build_train_command(candidate, python=sys.executable, s4m_root=ROOT / cfg['root'],
                                   config=config, artifacts=masks, output=output,
                                   device='cuda', variant='missing',
                                   train_epochs=cfg['training']['epochs'], patience=cfg['training']['patience'])
    for name, value in cfg['model'].items():
        _replace_argument(command, '--' + name, str(value))
    _replace_argument(command, '--fair_lradj', 'type1')
    _replace_argument(command, '--fair_mode', mode if mode != 'build' else 'smoke')
    return command


def _verify_core(root):
    metadata = json.loads((ROOT / 'integrations/s4m/upstream.json').read_text())
    for relative, expected in metadata['protected_blobs'].items():
        raw = (root / relative).read_bytes().replace(b'\r\n', b'\n')
        actual = sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if actual != expected:
            raise ValueError(f'protected S4M core changed: {relative}')


def _invoke(command, *, build_only=False):
    source = Path(command[1]); directory = source.parent
    sys.path[:0] = [str(directory), str(directory / 'model')]
    old_cwd, old_argv = Path.cwd(), sys.argv
    namespace = {'__name__': '__main__', '__file__': str(source)}
    text = source.read_text()
    marker = "    if args.fair_protocol:\n        setting ="
    if text.count(marker) != 1:
        raise ValueError('S4M build adapter no longer matches the pinned bridge')
    text = text.split(marker)[0]
    try:
        os.chdir(directory); sys.argv = command[1:]
        exec(compile(text, str(source), 'exec'), namespace)
        exp, args = namespace['Exp'](namespace['args']), namespace['args']
        original_validate = exp._fair_validate

        def fixed_validate(self, loader):
            # Upstream randomly samples topM bank entries during inference too.
            # Hold this RNG fixed so differences between draws come from the masks.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(2021)
                return original_validate(loader)

        exp._fair_validate = MethodType(fixed_validate, exp)
        if not build_only:
            exp.smoke() if args.fair_mode == 'smoke' else exp.train(args.fair_run_id)
        return exp, args
    finally:
        os.chdir(old_cwd); sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml')
    parser.add_argument('--masks', type=Path, default=DEFAULT_MASKS)
    parser.add_argument('--test-mask-bank', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--output', type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--smoke', action='store_true'); action.add_argument('--train', action='store_true')
    action.add_argument('--evaluate-selected', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve()
    cfg = load_author_profiles(args.profiles)['models']['S4M']
    if not torch.cuda.is_available():
        raise ValueError('the published S4M profile requires its CUDA environment')
    torch.set_num_threads(4)
    with TrainingResources(args.output, device='cuda:0', batch_size=16, model='S4M',
                           mode='evaluation' if args.evaluate_selected else 'smoke' if args.smoke else 'training',
                           started_at=_PROCESS_STARTED) as resources:
        protocol = load_p4_protocol(args.config, args.masks)
        _verify_core((ROOT / cfg['root']).resolve())
        identity = _code_identity(args.profiles, cfg, protocol)
        identity['s4m_entry_sha256'] = file_sha256(Path(__file__))
        selection = args.output / 'forecast-selection.json'
        if args.evaluate_selected:
            record = json.loads(selection.read_text())
            if record['status'] != 'validation_finalized':
                raise ValueError('S4M selection is not finalized')
            if any(record.get(k) != v for k, v in identity.items()):
                raise ValueError('S4M selected source/config identity changed')
            checkpoint = Path(record['checkpoint'])
            if file_sha256(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('S4M selected checkpoint SHA-256 mismatch')
            exp, run_args = _invoke(author_command(cfg, args.config, args.masks, args.output, mode='build'), build_only=True)
            exp.load_model(str(checkpoint))
            _, validation_loader = exp._get_data('val')
            validation = exp._fair_validate(validation_loader)
            if any(abs(validation[k] - record['validation_' + k]) > 1e-6 for k in ('mse', 'mae')):
                raise ValueError('S4M selected validation failed replay')
            dataset, _ = exp._get_data('test')

            def score(group):
                dataset.observed_timeline = group.fixed_masks['test']
                loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
                metrics = exp._fair_validate(loader)
                return {**metrics, 'windows': len(dataset.refs), 'target_elements': len(dataset.refs) * 96 * 7}

            result = evaluate_three_draws(protocol, args.test_mask_bank, selection, score)
            result.update(model='S4M', profile_version=cfg['version'], validation=validation,
                          training_resources=selected_training_resources([selection]))
            _write(args.output / 'three-test-draws.json', result)
        else:
            if selection.exists():
                raise FileExistsError(f'S4M selection already exists: {selection}')
            with resources.stage('forecast'):
                _, run_args = _invoke(author_command(cfg, args.config, args.masks, args.output,
                                                     mode='smoke' if args.smoke else 'train'))
                if not args.smoke:
                    manifest = args.output / 'runs' / f'{run_args.fair_run_id}.json'
                    record = json.loads(manifest.read_text())
                    _write(selection, {**record, **identity, 'model': 'S4M', 'profile': cfg,
                                       'evaluation_rng_seed': 2021,
                                       's4_kernel_backend': 'PyTorch fallback; compiled structured kernels unavailable in this environment',
                                       'profile_version': cfg['version']})
                    resources.selection_manifest = selection


if __name__ == '__main__':
    main()
