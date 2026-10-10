import json
from pathlib import Path

import pytest
import torch

from integrations.e23 import train as training
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_front_context import settings
from tests.integration.test_e23_stage1 import _protocol_settings, _batch


def test_only_front_frozen_and_exact_optimizer_coverage_and_other_components_learn(tmp_path):
    protocol, _ = _protocol_settings();cfg = settings(tmp_path, protocol)
    torch.manual_seed(502)
    model = training.build_forecast_model(protocol, cfg, device=torch.device('cpu'))
    for p in model.repair.historical.key_encoder.parameters():p.requires_grad_(False)
    optimizer = build_stage2_optimizer(model, mode='freeze_front_v1', learning_rate=1e-4, weight_decay=0)
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert not any(p.requires_grad for p in model.repair.front_context.parameters())
    assert not any(p.requires_grad for p in model.repair.historical.key_encoder.parameters())
    rates = {id(p):g['lr'] for g in optimizer.param_groups for p in g['params']}
    assert rates[id(model.repair.decoder.weight)] == 1e-5
    assert all(rates[id(p)] == 1e-4 for p in model.predictor.parameters() if p.requires_grad)
    before = parameter_hash(model.repair.front_context);decoder = model.repair.decoder.weight.detach().clone()
    predictor_before = parameter_hash(model.predictor)
    b = _batch('train');starts = training._starts(b, stride=8, patches=12, device=torch.device('cpu'))
    for _ in range(2):
        stage2_training_mode(model, 'freeze_front_v1')
        assert model.predictor.training and model.repair.training
        assert not model.repair.front_context.training and model.repair.point_refiner.training
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        loss = training.stage2_objective(prediction, b.forecast_target, loss_mode='mse_plus_0p2_mae')
        loss.backward()
        assert all(p.grad is None for p in model.repair.front_context.parameters())
        assert model.repair.decoder.weight.grad.abs().sum() > 0
        optimizer.step();assert before == parameter_hash(model.repair.front_context)
    assert not torch.equal(decoder, model.repair.decoder.weight)
    assert predictor_before != parameter_hash(model.predictor)


def test_frozen_front_selection_roundtrip_tamper_and_policy_relabel(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings();cfg = settings(tmp_path, protocol)
    monkeypatch.setattr(training, 'iter_p4_batches', lambda _protocol, split, **kw: iter((_batch(split),)))
    s1 = training.train_repair(protocol, cfg, output=tmp_path/'stage1', device=torch.device('cpu'), epochs=1)
    s2 = training.train_forecast(protocol, cfg, stage1_manifest=s1, output=tmp_path/'stage2', device=torch.device('cpu'), epochs=2,
        selection_metric='mae', forecast_loss='mse_plus_0p2_mae', repair_retention='uniform_v1', repair_update_policy='freeze_front_v1')
    payload = training.load_stage2_selection(s2, protocol, cfg, device=torch.device('cpu'))
    first = training.load_stage1_selection(s1, protocol, cfg, device=torch.device('cpu'))
    keys = [k for k in first['model_state'] if k.startswith('front_context.')]
    assert keys and all(torch.equal(first['model_state'][k], payload['model_state']['repair.'+k]) for k in keys)
    record = json.loads(s2.read_text());wrong = tmp_path/'wrong.json'
    wrong.write_text(json.dumps({**record, 'repair_update_policy':'slow_lr_0p1_v1'}))
    with pytest.raises(ValueError, match='repair update identity mismatch'):
        training.load_stage2_selection(wrong, protocol, cfg, device=torch.device('cpu'))
    payload['model_state']['repair.front_context.output.bias'] += .01
    checkpoint = Path(record['checkpoint']);torch.save(payload, checkpoint)
    record['checkpoint_sha256'] = training._file_sha256(checkpoint);s2.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='changed frozen Stage 1 front context state'):
        training.load_stage2_selection(s2, protocol, cfg, device=torch.device('cpu'))


def test_freezing_missing_front_rejected_before_training(tmp_path):
    protocol, cfg = _protocol_settings()
    with pytest.raises(ValueError, match='requires v46 window context'):
        training.train_forecast(protocol, cfg, stage1_manifest=tmp_path/'missing', output=tmp_path/'output', device=torch.device('cpu'),
            selection_metric='mae', forecast_loss='mse_plus_0p2_mae', repair_retention='uniform_v1', repair_update_policy='freeze_front_v1')
    assert not (tmp_path/'output').exists()
