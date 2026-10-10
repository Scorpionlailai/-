"""Protect warmup, observed-value preservation and independent predictor initialization."""
import importlib
from dataclasses import replace

import pytest
import torch

from integrations.forecast_compare.protocol import P4Batch


def api():
    try:
        return importlib.import_module('integrations.forecast_compare.auxiliary_forecast')
    except ModuleNotFoundError:
        pytest.fail('Stage1 auxiliary forecasting implementation is missing')


class MeanPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, values, mask, mean, std, *, patch_validity):
        return (values * patch_validity).mean((2, 3))[:, :, None].expand(-1, -1, 96) * self.scale


def batch():
    observed = torch.zeros(2, 96, 7)
    mask = torch.zeros_like(observed, dtype=torch.bool)
    mask[:, :16] = True
    return P4Batch(observed, torch.full_like(observed, 1000), mask,
                   torch.zeros(2, 7), torch.ones(2, 7), torch.zeros(2, 7, 96))


@pytest.mark.parametrize('epoch,expected_gradient', [(1, False), (10, False), (11, True)])
def test_forecast_gradient_reaches_only_missing_repairs_after_warmup(epoch, expected_gradient):
    x = torch.ones(2, 96, 7, requires_grad=True)
    head = MeanPredictor()
    loss = api().auxiliary_forecast_loss(head, x, batch(), epoch=epoch)
    loss.backward()
    assert head.scale.grad is not None and head.scale.grad.abs() > 0
    if expected_gradient:
        assert x.grad is not None and x.grad[:, 16:].abs().sum() > 0
        assert x.grad[:, :16].count_nonzero() == 0
    else:
        assert x.grad is None


def test_future_and_missing_history_targets_are_not_predictor_inputs():
    b = batch()
    x = torch.ones_like(b.normalized_observed)
    head = MeanPredictor()
    before = api().predict_repaired(head, x, b)
    altered = replace(b, normalized_target=torch.full_like(x, -999),
                      forecast_target=torch.ones_like(b.forecast_target))
    after = api().predict_repaired(head, x, altered)
    torch.testing.assert_close(before, after)
    assert api().auxiliary_forecast_loss(head, x, b, epoch=11) != api().auxiliary_forecast_loss(head, x, altered, epoch=11)


def test_auxiliary_dropout_does_not_change_repair_random_stream():
    head = MeanPredictor()
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    api().auxiliary_forecast_loss(head, torch.ones(2, 96, 7), batch(), epoch=11)
    assert torch.equal(before, torch.get_rng_state())


def test_auxiliary_initialization_is_independent_of_repair_rng_and_final_head():
    from tests.integration.test_e23_stage1 import _protocol_settings
    protocol, _ = _protocol_settings()
    torch.manual_seed(12)
    rng = torch.get_rng_state().clone()
    auxiliary = api().build_auxiliary_predictor(protocol, device=torch.device('cpu'))
    assert torch.equal(rng, torch.get_rng_state())
    from scripts.run_author_aligned_p4 import _forecast_head
    torch.manual_seed(2021)
    final = _forecast_head(protocol)
    assert sum(p.numel() for p in auxiliary.parameters()) == sum(p.numel() for p in final.parameters())
    assert not torch.equal(auxiliary.core.W_P.weight, final.core.W_P.weight)


def test_stage1_auxiliary_is_discarded_and_manifest_tampering_is_rejected(monkeypatch, tmp_path):
    import json
    from integrations.e23.train import train_repair, load_stage1_selection, train_forecast
    from tests.integration.test_e23_stage1 import _protocol_settings, _batch
    import integrations.e23.train as train_module
    protocol, settings = _protocol_settings()
    settings = replace(settings, stage1_epochs=11)
    monkeypatch.setattr(train_module, 'iter_p4_batches',
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    selected = train_repair(protocol, settings, output=tmp_path/'aux', device=torch.device('cpu'),
                            epochs=11, auxiliary_forecast='forecast_mse_v1')
    record = json.loads(selected.read_text())
    assert record['stage1_aux_forecast'] == 'forecast_mse_v1'
    assert record['stage1_aux_repair_gradient_batches'] == 1
    payload = load_stage1_selection(selected, protocol, settings, device=torch.device('cpu'))
    assert 'auxiliary_predictor_state' in payload
    second = train_forecast(protocol, settings, stage1_manifest=selected, output=tmp_path/'forecast',
                            device=torch.device('cpu'), epochs=1, forecast_loss='mse', selection_metric='mae')
    final = json.loads(second.read_text())
    assert final['loss'] == 'forecast_mse_only'
    assert final['predictor_initial_sha256'] != record['stage1_aux_predictor_initial_sha256']
    final_state = torch.load(final['checkpoint'], weights_only=False)['model_state']
    assert not any('auxiliary' in k for k in final_state)
    record['stage1_aux_forecast_weight'] = .5
    selected.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='auxiliary'):
        load_stage1_selection(selected, protocol, settings, device=torch.device('cpu'))


def test_stage1_warmup_preserves_original_repair_training(monkeypatch, tmp_path):
    import json
    from integrations.e23.train import train_repair
    from tests.integration.test_e23_stage1 import _protocol_settings, _batch
    import integrations.e23.train as train_module
    protocol, settings = _protocol_settings()
    monkeypatch.setattr(train_module, 'iter_p4_batches',
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    states=[]
    for mode in ('none', 'forecast_mse_v1'):
        selected=train_repair(protocol, settings, output=tmp_path/mode, device=torch.device('cpu'),
                              epochs=2, auxiliary_forecast=mode)
        states.append(torch.load(json.loads(selected.read_text())['checkpoint'], weights_only=False)['model_state'])
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


def test_author_t1_auxiliary_supervision_keeps_original_imputer_checkpoint(monkeypatch, tmp_path):
    import json
    from tests.integration.test_e23_stage1 import _protocol_settings, _batch
    from scripts import run_author_aligned_p4 as runner
    from integrations.forecast_compare.author_profiles import load_author_profiles, build_author_model, ROOT
    protocol, _ = _protocol_settings()
    cfg=load_author_profiles()['models']['T1']
    training={**cfg['training'], 'epochs': 11, 'batch_size': 2, 'precision': 'float32', 'patience': None}
    monkeypatch.setattr(runner, 'iter_p4_batches', lambda _protocol, split, **kwargs: iter((_batch(split),)))
    model=build_author_model('T1', cfg, ROOT/cfg['root'], torch.device('cpu'))
    selected=runner._train_stage('T1', model, protocol, None, cfg, training, 'repair', tmp_path,
                                 torch.device('cpu'), {}, auxiliary_forecast='forecast_mse_v1')
    record=json.loads(selected.read_text());payload=torch.load(record['checkpoint'], weights_only=False)
    assert record['stage1_aux_repair_gradient_batches']==1
    assert record['stage1_aux_predictor_discarded'] is True
    assert 'auxiliary_predictor_state' in payload
    model.load_state_dict(payload['model_state'], strict=True)
    assert not any('auxiliary' in k for k in payload['model_state'])
