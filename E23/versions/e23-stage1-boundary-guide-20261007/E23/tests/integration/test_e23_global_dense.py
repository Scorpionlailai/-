from dataclasses import replace

import torch

from integrations.e23 import train as tr
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_cross_variable_level import configuration, batch
from tests.integration.test_e23_recent_bands import run


def models(tmp_path):
    protocol, cfg = configuration(tmp_path)
    torch.manual_seed(2023)
    old = tr.build_forecast_model(protocol, cfg, device=torch.device('cpu')).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    new = tr.build_forecast_model(protocol, replace(cfg, recent_mean_projection='global_dense_time_v4'),
                                  device=torch.device('cpu')).eval()
    assert torch.equal(rng, torch.get_rng_state())
    return old, new, batch()


def test_zero_start_preserves_parent_function_rng_and_predictor(tmp_path):
    old, new, b = models(tmp_path)
    assert parameter_hash(old.predictor) == parameter_hash(new.predictor)
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    assert sum(p.numel() for p in new.repair.parameters()) - sum(p.numel() for p in old.repair.parameters()) == 79751
    with torch.no_grad():
        a, z = run(old, b), run(new, b)
    assert torch.equal(a.explicit.normalized_sequence, z.explicit.normalized_sequence)
    assert torch.equal(a.explicit.repaired_patches, z.explicit.repaired_patches)


def test_global_correction_preserves_visible_and_invalid_values(tmp_path):
    _, model, b = models(tmp_path)
    valid = torch.ones_like(b.observed_mask)
    valid[:, 93, 2] = False
    with torch.no_grad():
        a = run(model, b, valid=valid)
        model.repair.recent_mean_projector.global_dense_refiner.output.bias.fill_(.2)
        z = run(model, b, valid=valid)
    assert torch.equal(z.explicit.normalized_sequence[b.observed_mask & valid], b.normalized_observed[b.observed_mask & valid])
    assert torch.equal(z.explicit.normalized_sequence[~valid], a.explicit.normalized_sequence[~valid])
    hidden = ~b.observed_mask & valid
    expected = 2 * torch.tanh(torch.tensor(.1))
    delta = (z.explicit.normalized_sequence - a.explicit.normalized_sequence) * b.query_std[:, None]
    torch.testing.assert_close(delta[hidden], torch.full_like(delta[hidden], expected), atol=1e-5, rtol=1e-4)
    assert hidden[:, :64].any() and delta[:, :64][hidden[:, :64]].abs().min() > .1


def test_hidden_payload_and_empty_windows_cannot_drive_refinement(tmp_path):
    _, model, b = models(tmp_path)
    with torch.no_grad():
        ref = model.repair.recent_mean_projector.global_dense_refiner
        ref.output.weight.normal_(std=.01)
        a = run(model, b, torch.where(b.observed_mask, b.normalized_observed, 0))
        z = run(model, b, torch.where(b.observed_mask, b.normalized_observed, float('nan')))
        assert torch.equal(a.explicit.normalized_sequence, z.explicit.normalized_sequence)
        empty = replace(b, observed_mask=torch.zeros_like(b.observed_mask))
        a = run(model, empty)
        ref.output.bias.fill_(1)
        z = run(model, empty)
    assert torch.isfinite(a.explicit.normalized_sequence).all()
    assert torch.equal(a.explicit.normalized_sequence, z.explicit.normalized_sequence)


def test_gradients_reach_local_and_global_paths_without_predictor_training(tmp_path):
    _, model, b = models(tmp_path)
    model.train()
    opt = torch.optim.Adam(model.repair.parameters(), lr=.001)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = run(model, b)
        loss, _ = tr.stage1_objective(out.explicit.normalized_sequence, b.normalized_target,
            b.observed_mask, torch.tensor(0.), retrieval_weight=.1,
            loss_mode='global_mse_v1', query_std=b.query_std, structure_mode='recent_mean_v2')
        loss.backward()
        ref = model.repair.recent_mean_projector.global_dense_refiner
        assert ref.output.weight.grad.abs().sum() > 0
        if step:
            assert ref.temporal.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
            assert ref.depthwise.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.predictor.parameters())
        opt.step()


def test_coarse_conditioning_is_detached_but_direct_skip_has_gradient(tmp_path):
    _, model, b = models(tmp_path)
    with torch.no_grad():
        explicit = run(model, b).explicit
    ref = model.repair.recent_mean_projector.global_dense_refiner
    with torch.no_grad():
        ref.output.weight.normal_(std=.02)
    value = explicit.normalized_sequence.detach().clone().requires_grad_(True)
    ref(value, explicit, b.query_mean, b.query_std).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value.grad))


def test_fully_invalid_attention_window_remains_finite_and_unchanged(tmp_path):
    _, model, b = models(tmp_path)
    with torch.no_grad():
        explicit = run(model, b).explicit
        invalid = replace(explicit, sequence_validity=torch.zeros_like(explicit.sequence_validity),
            observed_sequence=torch.full_like(explicit.observed_sequence, float('nan')))
        values = torch.zeros_like(explicit.normalized_sequence)
        ref = model.repair.recent_mean_projector.global_dense_refiner
        ref.output.weight.normal_(std=.02)
        result = ref(values, invalid, b.query_mean, b.query_std)
    assert torch.isfinite(result).all()
    assert torch.equal(result, values)
