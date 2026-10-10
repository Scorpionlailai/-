"""Seal unseen positions before candidate training; generate no model scores."""
import hashlib
import json
import os
from pathlib import Path

from integrations.e23.config import load_e23_p4_protocol
from integrations.forecast_compare.test_mask_bank_v2 import create_test_bank, load_test_bank
from integrations.forecast_compare.test_mask_bank import _missing_fractions
from malr.data.masks import generate_variable_missing_mask, save_fixed_mask
from malr.migration import MigrationCatalog

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007')
SEEDS = tuple(range(2026100701, 2026100721))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')


def main():
    plan = json.loads((BASELINE / 'plan.json').read_text())
    records = []
    for scene in plan['scenarios']:
        os.environ.pop('MALR_MISSING_SCENARIO', None)
        if scene['scenario_manifest']:
            os.environ['MALR_MISSING_SCENARIO'] = scene['scenario_manifest']
        protocol = load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl', project='E23'),
            protocol_config=Path(plan['protocol_config']), mask_artifacts=Path(plan['mask_artifacts']), training_seed=2023)
        output = ROOT / scene['name'] / 'test_masks'
        if scene['name'] == 'sr010_bl8':
            manifest = create_test_bank(protocol, output, seeds=SEEDS)
        else:
            output.mkdir(parents=True, exist_ok=False)
            identity = dict(schema_version=2, dataset='ETTh1', scenario=scene['name'],
                source_csv_sha256=protocol.source_sha256, validation_mask_sha256=protocol.mask_sha256['validation'],
                seq_len=96, pred_len=96, n_vars=7, start_rate=scene['start_rate'], block_len=scene['block_len'],
                allow_block_overlap=True, support_start=protocol.bounds.test.start-96, support_end=protocol.bounds.test.end)
            groups = []
            for index, seed in enumerate(SEEDS):
                path = output / f'test_group_{index:02d}.npz'
                mask, metadata = generate_variable_missing_mask(identity['support_end']-identity['support_start'], 7,
                    scene['start_rate'], scene['block_len'], seed, dataset='ETTh1', split='test', support_start=identity['support_start'])
                save_fixed_mask(path, mask, metadata)
                timeline, window, count = _missing_fractions(protocol, mask)
                groups.append(dict(group_id=f'group_{index:02d}', split='test', seed=seed, path=path.name,
                    sha256=sha(path), missing_timeline_fraction=timeline, missing_evaluation_fraction=window, evaluation_windows=count))
            manifest = output / 'manifest.json'
            write(manifest, dict(**identity, group_count=20, seeds=SEEDS, groups=groups,
                purpose='reserved confirmation; not evaluated during development; one locked candidate',
                training_seeds='independent of mask seeds'))
            manifest.with_suffix('.sha256').write_text(sha(manifest)+'\n')
        views = load_test_bank(protocol, manifest, expected_seeds=SEEDS)
        assert len(views) == 20
        old = json.loads(Path(scene['test_bank']).read_text())
        assert not set(old['seeds']).intersection(SEEDS)
        assert not set(g['sha256'] for g in old['groups']).intersection(g['sha256'] for g in json.loads(manifest.read_text())['groups'])
        records.append(dict(scene=scene['name'], test_bank=str(manifest), bank_sha256=sha(manifest), seeds=SEEDS, groups=20))
    write(ROOT / 'reserved-lock.json', dict(status='sealed_not_evaluated', created_before_candidate_training=True,
        purpose='One locked shared candidate versus locked pure-MSE frozen T1; no prediction or repair scoring now.',
        training_seed_count=1, scenarios=records, generator_sha256=sha(__file__)))
    print(json.dumps(dict(status='sealed_not_evaluated', scenarios=4, groups_per_scene=20, scoring_performed=False)))


if __name__ == '__main__':
    main()
