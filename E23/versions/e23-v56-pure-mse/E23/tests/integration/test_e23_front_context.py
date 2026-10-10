from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from integrations.e23 import train as training
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.data.patching import patchify_with_state
from malr.models.e23.window_context import WindowContextEncoder
from tests.integration.test_e23_stage1 import _protocol_settings, _batch

ROOT = Path(__file__).resolve().parents[2]


def settings(tmp_path, protocol):
    document = yaml.safe_load((ROOT/'configs/e23_v46_front_context_seed502.yaml').read_text())
    document['training']['seed'] = protocol.training_seed
    path = tmp_path/'context.yaml'
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_e23_settings(path, protocol=protocol)


def inputs():
    torch.manual_seed(11)
    values = torch.randn(2, 96, 3)
    mask = torch.rand_like(values) > .73
    valid = torch.ones_like(mask);valid[:, -1] = False
    patches = patchify_with_state(values, mask, 16, 8, validity=valid)
    mean = torch.randn(2, 3);std = torch.rand(2, 3)+.1
    return values, mask, valid, patches.validity, mean, std


def test_front_context_preserves_original_initial_weights_rng_and_function(tmp_path):
    protocol, _ = _protocol_settings();trial = settings(tmp_path, protocol)
    torch.manual_seed(502)
    old = training.build_forecast_model(protocol, replace(trial, front_context='none'), device=torch.device('cpu')).eval()
    old_rng = torch.get_rng_state()
    torch.manual_seed(502)
    new = training.build_forecast_model(protocol, trial, device=torch.device('cpu')).eval()
    assert torch.equal(old_rng, torch.get_rng_state())
    assert parameter_hash(old.predictor) == parameter_hash(new.predictor)
    assert sum(p.numel() for p in old.repair.parameters()) == 174362
    for name, value in old.repair.state_dict().items():
        assert torch.equal(value, new.repair.state_dict()[name]), name
    batch = _batch('validation')
    starts = training._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    with torch.no_grad():
        a, oa = old(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
        b, ob = new(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
    assert torch.equal(a, b)
    assert torch.equal(oa.explicit.normalized_sequence, ob.explicit.normalized_sequence)


def test_features_use_global_visible_values_gaps_and_finite_empty_fallback():
    context = WindowContextEncoder(64, 96, 16, 8)
    values = torch.full((1, 96, 1), float('nan'));mask = torch.zeros_like(values, dtype=torch.bool)
    values[:, [2, 8]] = 2;mask[:, [2, 8]] = True
    features, _ = context.features(values, mask, torch.tensor([[3.]]), torch.tensor([[.001]]))
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[0, 0, [2, 8]], torch.full((2,), 3.002))
    assert features[0, 0, ~mask[0, :, 0]].eq(0).all()
    torch.testing.assert_close(features[0, 3, 5], torch.tensor(3/96))
    torch.testing.assert_close(features[0, 4, 5], torch.tensor(3/96))
    assert features[0, 3, 0] == 1 and features[0, 4, 95] == 1
    mask[:] = False
    features, _ = context.features(values, mask, torch.zeros(1, 1), torch.ones(1, 1))
    assert torch.isfinite(features).all() and features[:, 3:].eq(1).all()
    pv = patchify_with_state(torch.zeros_like(values), mask, 16, 8).validity
    assert torch.isfinite(context(values, mask, torch.zeros(1, 1), torch.ones(1, 1), pv)).all()


def test_nonzero_context_isolates_hidden_values_and_other_variables_and_reads_outside_patch():
    values, mask, valid, pv, mean, std = inputs()
    context = WindowContextEncoder(64, 96, 16, 8).eval()
    with torch.no_grad():context.output.weight.normal_(std=.05)
    a = context(values, mask, mean, std, pv, valid_mask=valid)
    polluted = values.clone();polluted[~(mask & valid)] = float('inf')
    b = context(polluted, mask, mean, std, pv, valid_mask=valid)
    assert torch.equal(a, b)
    changed = values.clone();changed[:, :, 1:] += 100
    c = context(changed, mask, mean, std, pv, valid_mask=valid)
    assert torch.equal(a[:, 0], c[:, 0])
    values, mask = torch.zeros(1, 96, 1), torch.ones(1, 96, 1, dtype=torch.bool)
    pv = patchify_with_state(values, mask, 16, 8).validity
    mean, std = torch.zeros(1, 1), torch.ones(1, 1)
    a = context(values, mask, mean, std, pv)
    values[:, 24] = 10  # outside first 16-point patch, inside CNN receptive field
    b = context(values, mask, mean, std, pv)
    assert not torch.equal(a[:, :, 0], b[:, :, 0])


def test_pool_excludes_runtime_invalid_points_and_padded_last_patch():
    _, _, valid, pv, _, _ = inputs()
    context = WindowContextEncoder(64, 96, 16, 8)
    points = torch.ones(6, 64, 96);points[:, :, -1] = 123456
    pooled = context.pool(points, pv)
    assert pooled.eq(1).all()


def test_full_repair_keeps_visible_values_and_learns_frontend_from_repair_loss(tmp_path):
    protocol, _ = _protocol_settings();trial = settings(tmp_path, protocol)
    torch.manual_seed(502)
    model = training.build_forecast_model(protocol, trial, device=torch.device('cpu'))
    batch = _batch('train')
    starts = training._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    opt = torch.optim.Adam(model.repair.parameters(), lr=.001)
    for step in range(2):
        opt.zero_grad()
        out = model.repair(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
        loss, _ = training.stage1_objective(out.explicit.normalized_sequence, batch.normalized_target,
            batch.observed_mask, torch.tensor(0.), retrieval_weight=.1,
            loss_mode='global_mse_v1', query_std=batch.query_std)
        loss.backward()
        front = model.repair.front_context
        assert front.output.weight.grad.abs().sum() > 0
        if step:
            assert front.input[0].weight.grad.abs().sum() > 0
            assert front.blocks[0].long.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.repair.parameters() if p.grad is not None)
        assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask], batch.normalized_observed[batch.observed_mask])
        opt.step()
    model.eval()
    with torch.no_grad():
        polluted = batch.normalized_observed.clone();polluted[~batch.observed_mask] = 1e9
        a, oa = model(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
        b, ob = model(polluted, batch.observed_mask, batch.query_mean, batch.query_std, starts)
    assert torch.equal(a, b) and torch.equal(oa.base_latents, ob.base_latents)


def test_front_context_checkpoint_roundtrip_and_rejects_wrong_version(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings();trial = settings(tmp_path, protocol)
    monkeypatch.setattr(training, 'iter_p4_batches', lambda _protocol, split, **kw: iter((_batch(split),)))
    manifest = training.train_repair(protocol, trial, output=tmp_path/'train', device=torch.device('cpu'), epochs=1)
    payload = training.load_stage1_selection(manifest, protocol, trial, device=torch.device('cpu'))
    assert payload['front_context'] == 'window_cnn_v1'
    assert payload['front_context_dilations'] == [1, 2]
    model = training.build_repair_model(protocol, trial, device=torch.device('cpu'))
    model.load_state_dict(payload['model_state'], strict=True)
    record = json.loads(manifest.read_text());record['front_context'] = 'wrong'
    tampered = tmp_path/'tampered.json';tampered.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='front_context.*mismatch'):
        training.load_stage1_selection(tampered, protocol, trial, device=torch.device('cpu'))


@pytest.mark.parametrize('value', ['unknown', True, 2])
def test_config_rejects_unknown_frontend(tmp_path, value):
    protocol, _ = _protocol_settings()
    doc = yaml.safe_load((ROOT/'configs/e23_v46_front_context_seed502.yaml').read_text())
    doc['training']['seed'] = protocol.training_seed;doc['model']['front_context'] = value
    path = tmp_path/'config.yaml';path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path, protocol=protocol)
