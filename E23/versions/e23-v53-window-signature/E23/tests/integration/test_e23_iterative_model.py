import numpy as np
import torch

from malr.models.e23.model import MultiExpertRepairEncoder, E23ForecastModel
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel
from malr.prototypes.online_bank import OnlinePrototypeMemory
from tests.integration.test_e23_model import _inputs


def _model(rounds=3, variables=2):
    repair = MultiExpertRepairEncoder(variables, 24, 8, 4, d_model=8, retrieval_dim=8,
        historical_threshold=-1., variable_selection="window_context_topk_v1",
        recovery_schedule="iterative_tv_h_v1", recovery_rounds=rounds)
    core = PatchTSTCore(c_in=variables, seq_len=24, pred_len=6, patch_len=8, stride=4,
                       d_model=8, n_heads=2, d_ff=16, n_layers=1, dropout=0.)
    return E23ForecastModel(repair, MaskAwareForecastModel(core))


def test_iterative_model_preserves_observations_padding_and_shared_shapes():
    # Catches changed masks, latent accumulation on clean patches, and extra copies.
    torch.manual_seed(19)
    model = _model()
    values, mask, valid, starts, mean, std = _inputs()
    forecast, out = model(values * mask, mask, mean, std, starts, valid_mask=valid)
    assert forecast.shape == (2, 2, 6)
    assert len(out.iterations) == 3
    assert torch.equal(out.explicit.normalized_sequence[mask & valid], values[mask & valid])
    assert not out.explicit.sequence_validity[:, -1].any()
    clean = ~out.router.available[..., 1:].any(-1)
    assert torch.equal(out.router.latents[clean], out.base_latents[clean])
    one = _model(1)
    assert {k: v.shape for k, v in one.state_dict().items()} == {
           k: v.shape for k, v in model.state_dict().items()}
    assert not any("iterations.0" in key for key in model.state_dict())


def test_iterative_model_no_evidence_is_exact_base_and_finite():
    model = _model(variables=1)
    values, mask, valid, starts, mean, std = _inputs(variables=1)
    mask[:] = False
    _, out = model(values * mask, mask, mean, std, starts, valid_mask=valid)
    assert torch.equal(out.router.latents, out.base_latents)
    assert out.router.weights[..., 0].eq(1).all()
    assert torch.isfinite(out.explicit.normalized_sequence).all()


def test_iterative_model_three_round_loss_reaches_all_experts():
    model = _model()
    memory = OnlinePrototypeMemory(patch_len=8, patch_stride=4, train_end=64, variable_count=2)
    for variable in range(2):
        memory.stage_patch(variable, 0, np.linspace(-1, 1, 8), np.eye(8)[0], observed=True)
    model.repair.historical.refresh_snapshot(memory.publish_epoch(1))
    values, mask, valid, starts, mean, std = _inputs()
    forecast, out = model(values * mask, mask, mean, std, starts, valid_mask=valid)
    (forecast.square().mean() + out.explicit.normalized_sequence.square().mean()).backward()
    for parameter in (model.repair.historical.message_projector[0].weight,
                      model.repair.temporal.source_context_projection.weight,
                      model.repair.variable.source_context_projection.weight,
                      model.repair.router.shared_score[0].weight,
                      model.repair.decoder.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert torch.equal(values.grad[~valid], torch.zeros_like(values.grad[~valid]))
    assert out.iterations[0].latents.grad_fn is not None


def test_iterative_forward_rebuilds_caches_and_ignores_hidden_input_values(monkeypatch):
    model = _model()
    values, mask, valid, starts, mean, std = _inputs()
    calls = []
    original = model.repair.temporal.prepare_sources
    def capture(*args, **kwargs):
        cache = original(*args, **kwargs)
        calls.append(cache)
        return cache
    monkeypatch.setattr(model.repair.temporal, "prepare_sources", capture)
    _, first = model(values * mask, mask, mean, std, starts, valid_mask=valid)
    changed = values.detach().clone()
    changed[~(mask & valid)] = 1e5
    _, second = model(changed, mask, mean, std, starts, valid_mask=valid)
    assert len(calls) == 2 and calls[0] is not calls[1]
    torch.testing.assert_close(first.router.latents, second.router.latents, rtol=0, atol=0)
