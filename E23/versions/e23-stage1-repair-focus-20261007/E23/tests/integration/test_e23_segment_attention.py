"""Protect long-gap scope, observation isolation, units and original head identity."""
from dataclasses import replace
from pathlib import Path
import json

import pytest
import torch
import yaml

from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def configuration(tmp_path):
    protocol, _ = setup(tmp_path)
    path = tmp_path / 'True-True-False.yaml'
    doc = yaml.safe_load(path.read_text())
    doc['model'].update(variable_calibration='none', front_variable_context='none',
                        repair_coordinates='robust_local_v1', point_variable_readout='none',
                        historical_key_policy='legacy_v1', recent_mean_projection='mean_v1',
                        temporal_evidence_readout='none', repair_input_scaling='legacy_v1',
                        point_context_readout='segment_attention_v1')
    doc['training'].update(stage1_structure='recent_mean_v2', stage1_gap_structure='long_gap_moments_v1')
    path.write_text(yaml.safe_dump(doc))
    return protocol, load_e23_settings(path, protocol=protocol)


def long_batch():
    batch = _batch('train')
    mask = torch.ones_like(batch.observed_mask)
    mask[:, :32] = False
    mask[:, 40:48] = False
    mask[:, 64:88] = False
    return replace(batch, observed_mask=mask,
                   normalized_observed=torch.where(mask, batch.normalized_observed, 0))


def test_zero_new_readout_preserves_existing_rng_parameters_and_forecast(tmp_path):
    protocol, cfg = configuration(tmp_path)
    torch.manual_seed(2023)
    old = tr.build_forecast_model(protocol, replace(cfg, point_context_readout='none'), device=torch.device('cpu')).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    new = tr.build_forecast_model(protocol, cfg, device=torch.device('cpu')).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert parameter_hash(old.predictor) == parameter_hash(new.predictor)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    batch = long_batch()
    starts = tr._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    with torch.no_grad():
        a, repair_a = old(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
        b, repair_b = new(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
    assert torch.equal(a, b)
    assert torch.equal(repair_a.explicit.normalized_sequence, repair_b.explicit.normalized_sequence)


def test_global_residual_reaches_before64_long_runs_but_preserves_short_and_split_runs(tmp_path):
    protocol, cfg = configuration(tmp_path)
    model = tr.build_forecast_model(protocol, cfg, device=torch.device('cpu')).eval()
    batch = long_batch()
    valid = torch.ones_like(batch.observed_mask)
    valid[:, 75, 0] = False  # Splits this variable's 24-gap into 11 and 12.
    std = torch.full_like(batch.query_std, 1e-5)
    starts = tr._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    with torch.no_grad():
        before = model.repair(batch.normalized_observed, batch.observed_mask, batch.query_mean, std, starts, valid_mask=valid).explicit
        model.repair.point_context_readout.output[-1].bias.fill_(.25)
        after = model.repair(batch.normalized_observed, batch.observed_mask, batch.query_mean, std, starts, valid_mask=valid).explicit
    expected = torch.zeros_like(batch.normalized_observed)
    expected[:, :32] = .25
    expected[:, 64:88, 1:] = .25
    torch.testing.assert_close(after.standardized_sequence-before.standardized_sequence, expected, atol=2e-6, rtol=1e-5)
    assert torch.equal(after.normalized_sequence[:, 40:48], before.normalized_sequence[:, 40:48])
    observed = batch.observed_mask & valid
    assert torch.equal(after.normalized_sequence[observed], batch.normalized_observed[observed])
    assert torch.equal(after.coverage, before.coverage)
    assert torch.equal(after.sequence_observed, before.sequence_observed)


def test_hidden_payload_and_invalid_point_features_cannot_leak_into_repair(tmp_path):
    protocol, cfg = configuration(tmp_path)
    model = tr.build_forecast_model(protocol, cfg, device=torch.device('cpu')).eval()
    batch = long_batch()
    starts = tr._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    with torch.no_grad():
        model.repair.point_context_readout.output[-1].weight.normal_(std=.1)
        a = model.repair(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts).explicit
        for poison in [float('nan'), float('inf'), -1e20]:
            values = torch.where(batch.observed_mask, batch.normalized_observed, poison)
            b = model.repair(values, batch.observed_mask, batch.query_mean, batch.query_std, starts).explicit
            assert torch.equal(a.normalized_sequence, b.normalized_sequence)
    assert torch.isfinite(a.normalized_sequence).all()


def test_empty_observations_stay_finite_and_invalid_windows_keep_original_rejection(tmp_path):
    protocol, cfg = configuration(tmp_path)
    model = tr.build_forecast_model(protocol, cfg, device=torch.device('cpu')).eval()
    batch = long_batch()
    mask = torch.zeros_like(batch.observed_mask)
    starts = tr._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    with torch.no_grad():
        model.repair.point_context_readout.output[-1].weight.normal_(std=.1)
        output = model.repair(torch.full_like(batch.normalized_observed, float('nan')), mask,
                              batch.query_mean, batch.query_std, starts).explicit
        assert torch.isfinite(output.normalized_sequence).all()
        assert output.sequence_validity.all()
        # The existing patch pipeline rejects an entirely invalid window before any readout.
        with pytest.raises(ValueError, match='no valid time points'):
            model.repair(torch.full_like(batch.normalized_observed, float('nan')), mask,
                         batch.query_mean, batch.query_std, starts, valid_mask=torch.zeros_like(mask))


def test_hidden_reconstruction_gradient_trains_attention_after_zero_output_start(tmp_path):
    protocol, cfg = configuration(tmp_path)
    model = tr.build_forecast_model(protocol, cfg, device=torch.device('cpu'))
    batch = long_batch()
    starts = tr._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    optimizer = torch.optim.Adam(model.repair.parameters(), lr=.001)
    for step in range(2):
        optimizer.zero_grad()
        output = model.repair(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
        loss, _ = tr.stage1_objective(output.explicit.normalized_sequence, batch.normalized_target,
                                     batch.observed_mask, torch.tensor(0.), retrieval_weight=.1,
                                     loss_mode='global_mse_v1', query_std=batch.query_std,
                                     structure_mode='recent_mean_v2', gap_structure_mode='long_gap_moments_v1')
        loss.backward()
        readout = model.repair.point_context_readout
        assert readout.output[-1].weight.grad.abs().sum() > 0
        if step:
            assert readout.input[0].weight.grad.abs().sum() > 0
            assert readout.blocks[0].attention.in_proj_weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()


def test_checkpoint_identity_and_full_frozen_stage2_roundtrip(tmp_path, monkeypatch):
    protocol, cfg = configuration(tmp_path)
    monkeypatch.setattr(tr, 'iter_p4_batches', lambda _p, split, **kwargs: iter((long_batch(),)))
    stage1 = tr.train_repair(protocol, cfg, output=tmp_path/'s1', device=torch.device('cpu'), epochs=1)
    stage2 = tr.train_forecast(protocol, cfg, stage1_manifest=stage1, output=tmp_path/'s2',
                             device=torch.device('cpu'), epochs=1, selection_metric='mae',
                             forecast_loss='mse_plus_0p2_mae', repair_retention='uniform_v1',
                             repair_update_policy='frozen_v1', extra_mse_checkpoint=True)
    state1 = torch.load(json.loads(stage1.read_text())['checkpoint'], weights_only=False)['model_state']
    for choice in [stage2, stage2.parent/'e23-stage2-mse.json']:
        payload = tr.load_stage2_selection(choice, protocol, cfg, device=torch.device('cpu'))
        assert payload['point_context_readout'] == 'segment_attention_v1'
        assert payload['point_context_scope'] == 96
        assert payload['point_context_min_gap'] == 16
        state2 = payload['model_state']
        assert all(torch.equal(value, state2['repair.'+key]) for key, value in state1.items())
    record = json.loads(stage2.read_text())
    record['point_context_scope'] = 32
    wrong = tmp_path/'wrong.json'
    wrong.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='point_context_scope.*mismatch'):
        tr.load_stage2_selection(wrong, protocol, cfg, device=torch.device('cpu'))
