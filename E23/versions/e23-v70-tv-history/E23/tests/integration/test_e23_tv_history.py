from dataclasses import replace
from pathlib import Path
import torch
import yaml

from integrations.e23 import train as training
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_stage1 import _protocol_settings, _batch

ROOT = Path(__file__).resolve().parents[2]


def settings(tmp_path, protocol):
    document = yaml.safe_load((ROOT/'configs/e23_v69_signature_position_seed2023.yaml').read_text())
    document['training']['seed'] = protocol.training_seed
    document['model']['historical_tv_query'] = 'fully_hidden_v1'
    path = tmp_path/'tv-history.yaml'
    path.write_text(yaml.safe_dump(document))
    return load_e23_settings(path, protocol=protocol)


def test_tv_query_keeps_head_and_original_repair_initialization(tmp_path):
    # Catches moving original initialization or passing a different forecast head configuration.
    protocol, _ = _protocol_settings(); trial = settings(tmp_path, protocol)
    torch.manual_seed(2023)
    original = training.build_forecast_model(protocol, replace(trial, historical_tv_query='none'), device=torch.device('cpu'))
    original_rng = torch.get_rng_state()
    torch.manual_seed(2023)
    actual = training.build_forecast_model(protocol, trial, device=torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert parameter_hash(actual.predictor) == parameter_hash(original.predictor)
    assert sum(p.numel() for p in actual.predictor.parameters()) == 35427
    assert sum(p.numel() for p in actual.repair.parameters()) == 425389
    for name, value in original.repair.state_dict().items():
        assert torch.equal(actual.repair.state_dict()[name], value), name


def test_stage1_tv_alignment_trains_query_and_preserves_observed_values(tmp_path):
    # Catches dropping the new query from the optimizer or losing its gradient in forward.
    protocol, _ = _protocol_settings(); trial = settings(tmp_path, protocol)
    repair = training.build_repair_model(protocol, trial, device=torch.device('cpu'))
    base = _batch('train'); mask = base.observed_mask.clone(); mask[:, 24:56] = False
    batch = replace(base, observed_mask=mask, normalized_observed=base.normalized_target.masked_fill(~mask, 0))
    starts = training._starts(batch, stride=8, patches=12, device=torch.device('cpu'))
    out = repair(batch.normalized_observed, mask, batch.query_mean, batch.query_std, starts)
    assert torch.equal(out.explicit.normalized_sequence[mask], batch.normalized_observed[mask])
    loss = training._tv_query_loss(repair, batch, out)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert repair.historical.tv_query.project.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in repair.temporal.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in repair.variable.parameters())
    assert all(p.grad is None for p in repair.historical.retrieval.frozen_encoder.parameters())


def test_controller_queries_h_after_the_variable_preview(monkeypatch):
    # Catches computing H queries/candidates before the sequential TV previews.
    from tests.unit.test_e23_iterative import _parts
    from malr.models.e23.tv_history import TVHistoricalQuery
    controller, inputs, parts = _parts(1)
    h = parts['historical']; h.tv_query = TVHistoricalQuery(8, 8)
    seen = {}
    original = h.prepare_tv_candidates
    def capture(arg, cache):
        seen['query_state'] = arg.base_latents
        seen['query_mask'] = arg.patch_mask
        return original(arg, cache)
    monkeypatch.setattr(h, 'prepare_tv_candidates', capture)
    from_cache = h.from_cache
    def read(arg, *args, **kwargs):
        seen['read_state'] = arg.base_latents
        return from_cache(arg, *args, **kwargs)
    monkeypatch.setattr(h, 'from_cache', read)
    controller(inputs, **parts)
    assert 'query_state' in seen
    assert torch.equal(seen['query_state'], seen['read_state'])
    assert not torch.equal(seen['query_state'], inputs.base_latents)
    assert torch.equal(seen['query_mask'], inputs.patch_mask)
