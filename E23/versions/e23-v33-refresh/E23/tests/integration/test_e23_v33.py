from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch

from integrations.e23 import train
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.round_supervision import supervised_round_objective, decode_repair_rounds
from tests.integration.test_e23_stage1 import _protocol_settings, _batch

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ('e23_multi_expert_p4_round_supervision_v33.yaml',
           'e23_multi_expert_p4_refreshed_rounds_v33.yaml')


@pytest.mark.parametrize('config', CONFIGS)
def test_round_supervision_reaches_every_state_and_clamps_observations(config):
    protocol, _ = _protocol_settings()
    settings = load_e23_settings(ROOT/'configs'/config, protocol=protocol)
    model = train.build_repair_model(protocol, settings, device=torch.device('cpu'))
    batch = _batch('train')
    starts = train._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    out = model(batch.normalized_observed, batch.observed_mask,
                batch.query_mean, batch.query_std, starts)
    for sequence in decode_repair_rounds(model, out, batch):
        assert torch.equal(sequence[batch.observed_mask], batch.normalized_observed[batch.observed_mask])
    for item in out.iterations:
        item.latents.retain_grad()
    retrieval = torch.tensor(.7, requires_grad=True)
    loss, parts = supervised_round_objective(model, out, batch, retrieval, retrieval_weight=.1)
    loss.backward()
    assert retrieval.grad.item() == pytest.approx(.1)
    assert all(i.latents.grad is not None and i.latents.grad.abs().sum() > 0 for i in out.iterations)
    assert torch.isfinite(loss)
    assert not model.decoder.weight.grad.isnan().any()


def test_original_predictor_initialization_and_structure_are_identical():
    protocol, _ = _protocol_settings()
    names = (*CONFIGS, 'e23_multi_expert_p4_width64_prediction_preservation_v32.yaml')
    models = []
    for name in names:
        settings = load_e23_settings(ROOT/'configs'/name, protocol=protocol)
        torch.manual_seed(2021)
        models.append(train.build_forecast_model(protocol, settings, device=torch.device('cpu')))
    assert len({parameter_hash(m.predictor) for m in models}) == 1
    assert len({str(m.predictor) for m in models}) == 1
    assert all(sum(p.numel() for p in m.predictor.parameters()) == 35427 for m in models)


def test_round_diagnostic_uses_original_predictor_padding(monkeypatch):
    from scripts import diagnose_e23_v33 as diagnostic
    protocol, _ = _protocol_settings()
    settings = load_e23_settings(ROOT/'configs'/CONFIGS[0], protocol=protocol)
    model = train.build_forecast_model(protocol, settings, device=torch.device('cpu')).eval()
    b = _batch('validation')
    monkeypatch.setattr(diagnostic, 'iter_p4_batches', lambda *args, **kwargs: iter((b,)))
    with torch.inference_mode():
        prediction, _ = model(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std,
            train._starts(b, stride=8, patches=12, device=torch.device('cpu')))
        expected = (prediction-b.forecast_target).double().square().mean().item()
    result = diagnostic.diagnose(model, protocol, settings, torch.device('cpu'), 'stage2')
    assert result['rounds'][-1]['forecast_mse'] == pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize('config', CONFIGS)
def test_two_stage_identity_and_pure_mse_roundtrip(monkeypatch, tmp_path, config):
    protocol, _ = _protocol_settings()
    settings = load_e23_settings(ROOT/'configs'/config, protocol=protocol)
    monkeypatch.setattr(train, 'iter_p4_batches', lambda p, split, **kw: iter((_batch(split),)))
    s1 = train.train_repair(protocol, settings, output=tmp_path/'stage1', device=torch.device('cpu'), epochs=2)
    r1 = json.loads(s1.read_text())
    assert r1['stage1_round_weights'] == [.2, .3, .5]
    assert r1['stage1_selection_metric_space'] == 'train_standardized'
    s2 = train.train_forecast(protocol, settings, stage1_manifest=s1, output=tmp_path/'stage2',
        device=torch.device('cpu'), epochs=1, selection_metric='mae', forecast_loss='mse')
    r2 = json.loads(s2.read_text())
    assert r2['loss'] == 'forecast_mse_only'
    assert r2['recovery_schedule'] == settings.recovery_schedule
    assert r2.get('repair_retention', 'none') == 'none'
    train.load_stage2_selection(s2, protocol, settings, device=torch.device('cpu'))
    with pytest.raises(ValueError, match='stage1_objective.*mismatch'):
        train.load_stage1_selection(s1, protocol, replace(settings, stage1_objective='global_mse_v1'), device=torch.device('cpu'))
