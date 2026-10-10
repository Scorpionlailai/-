import importlib
from dataclasses import replace

import pytest
import torch

from tests.integration.test_e23_stage1 import _batch, _protocol_settings


def api():
    try:
        return importlib.import_module('integrations.e23.prediction_preservation')
    except ModuleNotFoundError:
        pytest.fail('prediction preservation implementation is missing')


def probe_and_batch():
    protocol, _ = _protocol_settings()
    probe = api().build_probe(protocol, device=torch.device('cpu'), seed=2053)
    return probe, _batch('train')


def test_clean_probe_initialization_and_training_preserve_repair_rng():
    batch = _batch('train')
    protocol, _ = _protocol_settings()
    before = torch.get_rng_state().clone()
    probe = api().build_probe(protocol, device=torch.device('cpu'), seed=2053)
    assert torch.equal(before, torch.get_rng_state())
    old = [p.detach().clone() for p in probe.parameters()]
    optimizer = torch.optim.Adam(probe.parameters(), lr=.001)
    before = torch.get_rng_state().clone()
    loss = api().train_clean_probe_step(probe, optimizer, batch)
    assert loss >= 0 and torch.equal(before, torch.get_rng_state())
    assert any(not torch.equal(a,b) for a,b in zip(old, probe.parameters()))


def test_consistency_uses_same_masks_stats_and_detached_clean_target():
    probe, batch = probe_and_batch()
    api().freeze_probe(probe)
    clean = batch.normalized_target.detach().clone().requires_grad_(True)
    batch = replace(batch, normalized_target=clean)
    repaired = clean.detach().clone().requires_grad_(True)
    zero = api().consistency_loss(probe, repaired, batch)
    assert zero.item() == 0
    perturbed = (clean.detach() + (~batch.observed_mask).float()).requires_grad_(True)
    old = {k:v.clone() for k,v in probe.state_dict().items()}
    loss = api().consistency_loss(probe, perturbed, batch)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert perturbed.grad[~batch.observed_mask].abs().sum() > 0
    assert clean.grad is None and all(p.grad is None for p in probe.parameters())
    assert all(torch.equal(old[k],v) for k,v in probe.state_dict().items())
    from malr.data.patching import patchify_preserving_values
    patches = patchify_preserving_values(perturbed, batch.observed_mask, 16, 8)
    expected = probe(patches.values.masked_fill(~patches.validity,0),patches.effective_mask,
                     batch.query_mean,batch.query_std,patch_validity=patches.validity)
    torch.testing.assert_close(api().predict_sequence(probe,perturbed,batch), expected)


def test_consistency_requires_frozen_teacher_and_handles_small_scales():
    probe, batch = probe_and_batch()
    with pytest.raises(ValueError, match='frozen'):
        api().consistency_loss(probe,batch.normalized_observed,batch)
    api().freeze_probe(probe)
    batch = replace(batch, query_std=torch.full_like(batch.query_std, 1e-5))
    repaired = batch.normalized_observed.clone().requires_grad_(True)
    loss = api().consistency_loss(probe,repaired,batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(repaired.grad).all()


def test_preservation_identity_is_fixed_and_none_is_legacy():
    assert api().preservation_identity('none') == {}
    identity = api().preservation_identity('clean_forecast_v1')
    assert identity['stage1_preservation_weight'] == .1
    assert identity['stage1_preservation_warmup'] == 20
    assert identity['stage1_preservation_probe_seed'] == 2053
    with pytest.raises(ValueError): api().preservation_identity('typo')


def test_stage1_probe_lifecycle_handoff_and_tamper_rejection(monkeypatch, tmp_path):
    import json
    import integrations.e23.train as training
    from integrations.e23.forecast_sensitive import parameter_hash
    protocol, settings = _protocol_settings()
    settings = replace(settings, stage1_objective='global_mse_v1')
    monkeypatch.setattr(training, 'iter_p4_batches',
                        lambda _p, split, **kw: iter((_batch(split),)))
    selected = training.train_repair(protocol,settings,output=tmp_path/'repair',
        device=torch.device('cpu'),epochs=21,stage1_preservation='clean_forecast_v1')
    record = json.loads(selected.read_text())
    assert record['preservation_probe_training_batches'] == 20
    assert record['preservation_active_batches'] == 1
    assert record['preservation_active_windows'] == 2
    assert record['preservation_probe_frozen_sha256'] == record['preservation_probe_final_sha256']
    payload = training.load_stage1_selection(selected,protocol,settings,device=torch.device('cpu'))
    assert not any('probe' in k for k in payload['model_state'])
    stage2 = training.train_forecast(protocol,settings,stage1_manifest=selected,
        output=tmp_path/'forecast',device=torch.device('cpu'),epochs=1,selection_metric='mae')
    s2 = json.loads(stage2.read_text());assert s2['loss'] == 'forecast_mse_only'
    assert s2['stage1_preservation'] == 'clean_forecast_v1'
    training.load_stage2_selection(stage2,protocol,settings,device=torch.device('cpu'))
    torch.manual_seed(settings.seed)
    fresh = training.build_forecast_model(protocol,settings,device=torch.device('cpu'))
    assert s2['predictor_initial_sha256'] == parameter_hash(fresh.predictor)
    assert s2['predictor_initial_sha256'] != record['preservation_probe_final_sha256']
    s2.pop('stage1_preservation');stage2.write_text(json.dumps(s2))
    with pytest.raises(ValueError,match='preservation'):
        training.load_stage2_selection(stage2,protocol,settings,device=torch.device('cpu'))
    record['stage1_preservation_weight']=.2;selected.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='preservation'):
        training.load_stage1_selection(selected,protocol,settings,device=torch.device('cpu'))
