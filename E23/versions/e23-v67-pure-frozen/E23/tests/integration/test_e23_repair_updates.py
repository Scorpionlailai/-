import json
from pathlib import Path

import pytest
import torch
from torch import nn

from integrations.e23 import train as training
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_stage1 import _protocol_settings, _batch
from tests.integration.test_e23_iterative_training import _settings


def toy():
    model = nn.Module()
    model.repair = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.8))
    model.predictor = nn.Linear(4, 2)
    return model


def test_frozen_repair_keeps_outputs_and_state_while_predictor_learns():
    model = toy()
    optimizer = build_stage2_optimizer(model, mode="frozen_v1", learning_rate=.001, weight_decay=0)
    x = torch.randn(8, 4)
    before = parameter_hash(model.repair)
    head = parameter_hash(model.predictor)
    output = model.repair(x).clone()
    for _ in range(3):
        stage2_training_mode(model, "frozen_v1")
        assert model.predictor.training and not model.repair.training
        optimizer.zero_grad()
        model.predictor(model.repair(x)).square().mean().backward()
        assert all(p.grad is None for p in model.repair.parameters())
        optimizer.step()
        assert torch.equal(output, model.repair(x))
        assert parameter_hash(model.repair) == before
    assert parameter_hash(model.predictor) != head


@pytest.mark.parametrize("policy", ["joint", "slow_lr_0p1_v1"])
def test_optimizer_rates_coverage_and_preexisting_frozen_parameters(policy):
    model = toy()
    model.repair[0].bias.requires_grad_(False)
    optimizer = build_stage2_optimizer(model, mode=policy, learning_rate=.0001, weight_decay=0)
    rates = {id(p):g['lr'] for g in optimizer.param_groups for p in g['params']}
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert rates[id(model.predictor.weight)] == .0001
    assert rates[id(model.repair[0].weight)] == (.00001 if policy != 'joint' else .0001)
    stage2_training_mode(model, policy)
    assert model.repair.training and model.predictor.training


@pytest.mark.parametrize("policy", ["frozen_v1", "slow_lr_0p1_v1"])
def test_stage2_saved_policy_and_frozen_state_cannot_be_relabelled(monkeypatch, tmp_path, policy):
    protocol, _ = _protocol_settings()
    settings = _settings(tmp_path, protocol, rounds=1)
    monkeypatch.setattr(training, 'iter_p4_batches', lambda _protocol, split, **kwargs: iter((_batch(split),)))
    stage1 = training.train_repair(protocol, settings, output=tmp_path/'stage1', device=torch.device('cpu'), epochs=1)
    stage2 = training.train_forecast(protocol, settings, stage1_manifest=stage1, output=tmp_path/'stage2',
        device=torch.device('cpu'), epochs=2, selection_metric='mae', forecast_loss='mse_plus_0p2_mae',
        repair_retention='uniform_v1', repair_update_policy=policy)
    record = json.loads(stage2.read_text())
    payload = training.load_stage2_selection(stage2, protocol, settings, device=torch.device('cpu'))
    assert record['repair_update_policy'] == payload['repair_update_policy'] == policy
    tampered = tmp_path/'wrong-policy.json'
    tampered.write_text(json.dumps({k:v for k,v in record.items() if k != 'repair_update_policy'}))
    with pytest.raises(ValueError, match='repair update checkpoint mismatch'):
        training.load_stage2_selection(tampered, protocol, settings, device=torch.device('cpu'))
    if policy == 'frozen_v1':
        first = training.load_stage1_selection(stage1, protocol, settings, device=torch.device('cpu'))
        assert all(torch.equal(v, payload['model_state']['repair.'+k]) for k,v in first['model_state'].items())
        key = 'repair.decoder.weight'
        payload['model_state'][key] = payload['model_state'][key] + .01
        checkpoint = Path(record['checkpoint'])
        torch.save(payload, checkpoint)
        record['checkpoint_sha256'] = training._file_sha256(checkpoint)
        stage2.write_text(json.dumps(record))
        with pytest.raises(ValueError, match='changed frozen Stage 1 repair state'):
            training.load_stage2_selection(stage2, protocol, settings, device=torch.device('cpu'))
