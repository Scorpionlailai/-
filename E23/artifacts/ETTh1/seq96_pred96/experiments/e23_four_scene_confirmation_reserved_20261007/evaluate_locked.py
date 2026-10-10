"""Evaluate exactly one prelocked winner on the reserved bank, without training."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007')


def read(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['E23', 'T1'], required=True)
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    lock = read(ROOT / 'candidate-lock-before-confirmation.json')
    from malr.evaluation.report import file_sha256
    assert all(file_sha256(Path(path)) == digest for path, digest in lock['protected'].items())
    cell = next(s for s in lock['scenarios'] if s['scene'] == args.scene)
    os.environ.pop('MALR_MISSING_SCENARIO', None)
    if cell['scenario_manifest']:
        os.environ['MALR_MISSING_SCENARIO'] = cell['scenario_manifest']
    from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
    seeds = tuple(cell['seeds'])

    def load_reserved(protocol, manifest):
        assert file_sha256(Path(manifest)) == cell['bank_sha256']
        return load_test_bank(protocol, manifest, expected_seeds=seeds)

    if args.model == 'E23':
        module = load_module('e23_reserved_evaluator', Path(lock['candidate_work']) / 'scripts/evaluate_e23_t1_replicate.py')
        module.load_test_bank = load_reserved
        module.TEST_SEEDS = seeds
        sys.argv = [str(ROOT / 'evaluate_locked.py'), '--model', 'E23-v31-B',
            '--model-label', lock['candidate_label'] + '-reserved-confirmation', '--training-seed', '2023',
            '--e23-config', lock['candidate_config'], '--selection', cell['e23_selection'],
            '--test-mask-bank', cell['test_bank'], '--output', str(ROOT / 'E23' / args.scene / 'evaluation_mae'),
            '--device', 'cuda:1']
        module.main()
    else:
        plan = read(BASELINE / 't1-plan.json')
        scene = next(s for s in plan['scenarios'] if s['name'] == args.scene)
        scene.update(test_bank=cell['test_bank'], test_bank_sha256=cell['bank_sha256'])
        plan['root'] = str(ROOT / 'T1')
        module = load_module('t1_reserved_evaluator', Path(plan['work']) / 'scripts/run_t1_frozen_stage2.py')
        module.load_test_bank = load_reserved
        module.TEST_SEEDS = seeds
        module.torch.set_num_threads(4)
        protocol, views, _, model = module.setup(plan, scene)
        module.evaluate(plan, scene, protocol, views, model, Path(cell['t1_selection']), 'mae')
    assert all(file_sha256(Path(path)) == digest for path, digest in lock['protected'].items())


if __name__ == '__main__':
    main()
