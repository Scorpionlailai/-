"""Guards for fair RNG pairing, test-only draws, and crossed uncertainty."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import load_p4_protocol, iter_p4_batches, with_training_seed
from integrations.forecast_compare.replicate_statistics import summarize_matrix, crossed_paired_ci
from integrations.forecast_compare.test_mask_bank_v2 import create_test_bank, load_test_bank
from scripts.run_author_aligned_p4 import _forecast_head

ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


@pytest.fixture(scope='module')
def protocol():
    return load_p4_protocol(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS)


def test_training_seed_varies_training_only_and_pairs_author_batches(protocol):
    first, second = (with_training_seed(protocol, s) for s in [102, 202])
    a = next(iter_p4_batches(first, 'train', epoch=0, batch_size=128))
    b = next(iter_p4_batches(first, 'train', epoch=0, batch_size=16))
    c = next(iter_p4_batches(second, 'train', epoch=0, batch_size=128))
    assert a.refs[:16] == b.refs
    torch.testing.assert_close(a.observed_mask[:16], b.observed_mask, rtol=0, atol=0)
    assert a.refs != c.refs
    assert not torch.equal(a.observed_mask, c.observed_mask)
    for split in ['validation', 'test']:
        a, b = (next(iter_p4_batches(p, split)) for p in [first, second])
        assert a.refs == b.refs
        for field in ['observed_mask', 'forecast_target', 'normalized_observed']:
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    with pytest.raises(ValueError):
        with_training_seed(protocol, True)


@pytest.mark.parametrize('seed', [102, 202, 302, 402, 502])
def test_each_seed_uses_identical_original_predictor_initialization(protocol, tmp_path, seed):
    path = tmp_path/'config.yaml'
    path.write_text((ROOT/'configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml').read_text().replace('seed: 2021', f'seed: {seed}'))
    view = with_training_seed(protocol, seed)
    settings = load_e23_settings(path, protocol=view)
    torch.manual_seed(seed)
    ours = build_forecast_model(view, settings, device=torch.device('cpu'))
    torch.manual_seed(seed)
    head = _forecast_head(view)
    assert parameter_hash(ours.predictor) == parameter_hash(head)
    assert sum(p.numel() for p in head.parameters()) == 35427
    with pytest.raises(ValueError):
        load_e23_settings(path, protocol=protocol)


def test_v2_bank_is_shared_across_training_seeds_and_rejects_tampering(protocol, tmp_path):
    path = create_test_bank(protocol, tmp_path/'bank', seeds=(601, 602))
    views_a = load_test_bank(with_training_seed(protocol, 102), path, expected_seeds=(601, 602))
    views_b = load_test_bank(with_training_seed(protocol, 202), path, expected_seeds=(601, 602))
    for a, b in zip(views_a, views_b):
        assert a.training_seed == 102 and b.training_seed == 202
        assert a.mask_sha256 == b.mask_sha256
        np.testing.assert_array_equal(a.fixed_masks['test'], b.fixed_masks['test'])
        assert a.fixed_masks['validation'] is protocol.fixed_masks['validation']
    entry = json.loads(path.read_text())['groups'][0]
    mask = path.parent/entry['path']
    mask.write_bytes(mask.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='hash'):
        load_test_bank(protocol, path, expected_seeds=(601, 602))


def test_repeated_masks_cannot_artificially_shrink_training_seed_uncertainty():
    data = np.array([-2, -1, 0, 1, 2])[:, None]*np.ones((1, 20))
    repeated = np.tile(data, (1, 10))
    a = crossed_paired_ci(data, np.zeros_like(data), draws=1000)
    b = crossed_paired_ci(repeated, np.zeros_like(repeated), draws=1000)
    assert a['crossed_bootstrap_ci95'] == b['crossed_bootstrap_ci95']
    summary = summarize_matrix(data)
    assert summary['training_seed_std_ddof1'] == pytest.approx(np.std([-2, -1, 0, 1, 2], ddof=1))
    assert summary['test_mask_std_ddof1'] == 0


def test_prepared_trial_reloads_configs_as_paths_and_refuses_config_changes(tmp_path):
    from scripts.run_e23_t1_five_seed import prepare, verify_frozen
    definition = prepare(tmp_path/'trial', 'cuda:1')
    assert verify_frozen(tmp_path/'trial') == definition
    config = Path(next(iter(definition['run_configs'])))
    config.write_text(config.read_text()+'\n# changed\n')
    with pytest.raises(ValueError, match='configuration'):
        verify_frozen(tmp_path/'trial')


def test_report_uses_complete_resource_stages_and_keeps_partial_status(tmp_path):
    from integrations.forecast_compare.test_mask_bank_v2 import TEST_SEEDS
    from malr.evaluation.report import file_sha256
    from scripts.summarize_e23_t1_five_seed import summarize
    (tmp_path/'test_masks').mkdir()
    (tmp_path/'test_masks/manifest.json').write_text('{}')
    bank_hash = file_sha256(tmp_path/'test_masks/manifest.json')
    for seed in [102, 202]:
        for model in ['E23-v31-B', 'T1']:
            root = tmp_path/f'seed_{seed}'/model
            root.mkdir(parents=True)
            repair = root/'repair.json'
            repair.write_text(json.dumps({'epochs_completed': 100}))
            selection = root/'forecast.json'
            selection.write_text(json.dumps({'epochs_completed': 100,
                                             'stage1_manifest': str(repair),
                                             'repair_selection_manifest': str(repair)}))
            resource = root/'resources.json'
            resource.write_text(json.dumps({'device': {'uuid': 'fake-GPU'},
                                            'stages': {'repair': {'wall_seconds': 10.},
                                                       'forecast': {'wall_seconds': 20.}}}))
            (root/'evaluation').mkdir()
            result = dict(model=model, training_seed=seed, mask_seeds=list(TEST_SEEDS),
                          test_mask_bank_sha256=bank_hash, selection_manifest=str(selection),
                          selection_manifest_sha256=file_sha256(selection), checkpoint_sha256='fake',
                          predictor_initial_sha256=f'same-{seed}',
                          validation_replay={'mse': .5, 'mae': .4},
                          test_mean={'mse': .3, 'mae': .2}, test_mask_std_ddof1={'mse': .01, 'mae': .01},
                          test_draws=[dict(mask_seed=s, mask_sha256=f'mask-{s}', mse=.3, mae=.2,
                                           windows=2688, target_elements=2688*96*7) for s in TEST_SEEDS],
                          training_resources=dict(resource_status='measured_all_stages',
                              full_workflow_seconds=35., peak_allocated_gib=2., peak_reserved_gib=3.,
                              gpu='fake GPU', resource_record_paths=[str(resource)]))
            (root/'evaluation/twenty-test-draws.json').write_text(json.dumps(result))
    status = summarize(tmp_path)
    assert status['status'] == 'partial' and status['completed_runs'] == 4
    assert status['models']['T1']['mse']['mean'] == pytest.approx(.3)
    import csv
    rows = list(csv.DictReader((tmp_path/'comparison_summary.csv').open()))
    assert float(rows[0]['full_training_seconds_mean']) == 35.
    assert float(rows[0]['repair_stage_seconds_mean']) == 10.
    assert float(rows[0]['forecast_stage_seconds_mean']) == 20.
