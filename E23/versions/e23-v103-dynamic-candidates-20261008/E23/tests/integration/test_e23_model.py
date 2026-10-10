import numpy as np
import pytest
import torch
from pathlib import Path

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model
from malr.migration import MigrationCatalog

from malr.models.e23.model import E23ForecastModel, MultiExpertRepairEncoder
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel
from malr.prototypes.online_bank import OnlinePrototypeMemory


def _model(variables=2):
    repair = MultiExpertRepairEncoder(
        variable_count=variables, window_length=24, patch_length=8,
        stride=4, d_model=8, retrieval_dim=8, historical_threshold=-1.0,
    )
    core = PatchTSTCore(
        c_in=variables, seq_len=24, pred_len=6, patch_len=8, stride=4,
        d_model=8, n_heads=2, d_ff=16, n_layers=1, dropout=0.0,
    )
    return E23ForecastModel(repair, MaskAwareForecastModel(core))


def _inputs(variables=2):
    values = torch.randn(2, 24, variables, requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, 8:12] = False
    valid = torch.ones_like(mask)
    valid[:, -1] = False
    mask[:, -1] = False
    starts = torch.arange(6).repeat(2, 1) * 4 + 64
    mean = torch.zeros(2, variables)
    std = torch.ones(2, variables)
    return values, mask, valid, starts, mean, std


def test_e23_repairs_only_missing_preserves_mask_and_padding_gradient():
    torch.manual_seed(7)
    model = _model()
    values, mask, valid, starts, mean, std = _inputs()
    forecast, output = model(values, mask, mean, std, starts, valid_mask=valid)
    explicit = output.explicit
    assert forecast.shape == (2, 2, 6)
    assert explicit.normalized_sequence.shape == values.shape
    assert torch.equal(explicit.sequence_observed, mask & valid)
    assert torch.equal(explicit.normalized_sequence[mask & valid], values[mask & valid])
    assert not explicit.sequence_validity[:, -1].any()
    assert torch.equal(explicit.repaired_patches[~explicit.validity_patches],
                       torch.zeros_like(explicit.repaired_patches[~explicit.validity_patches]))
    assert explicit.coverage.max() == 2
    assert torch.equal(output.router.weights.sum(-1), torch.ones_like(output.router.weights[..., 0]))
    (forecast.square().mean() + explicit.normalized_sequence.square().mean()).backward()
    assert torch.equal(values.grad[~valid], torch.zeros_like(values.grad[~valid]))
    assert model.repair.decoder.weight.grad is not None
    assert model.predictor.core.head.linear.weight.grad is not None
    with pytest.raises(TypeError, match="forecast_target"):
        model(values, mask, mean, std, starts, valid_mask=valid, forecast_target=forecast)


def test_e23_all_experts_empty_routes_exactly_to_base():
    model = _model(variables=1)
    values, mask, valid, starts, mean, std = _inputs(variables=1)
    mask[:] = False
    _, output = model(values, mask, mean, std, starts, valid_mask=valid)
    assert not output.router.available[..., 1:].any()
    assert torch.equal(output.router.weights[..., 0], torch.ones_like(output.router.weights[..., 0]))
    assert torch.equal(output.router.latents, output.base_latents)
    assert torch.isfinite(output.explicit.normalized_sequence).all()


def test_e23_enabled_experts_router_and_forecaster_receive_gradient():
    torch.manual_seed(10)
    model = _model()
    memory = OnlinePrototypeMemory(
        patch_len=8, patch_stride=4, train_end=64, variable_count=2,
    )
    for variable in range(2):
        memory.stage_patch(variable, 0, np.linspace(-1, 1, 8),
                           np.array([1., 0., 0., 0., 0., 0., 0., 0.]), observed=True)
    model.repair.historical.refresh_snapshot(memory.publish_epoch(1))
    values, mask, valid, starts, mean, std = _inputs()
    forecast, output = model(values, mask, mean, std, starts, valid_mask=valid)
    assert all(output.evidence[name].available.any() for name in
               ("historical", "variable", "temporal"))
    forecast.square().mean().backward()
    params = (
        model.repair.historical.message_projector[0].weight,
        model.repair.variable.message_projector[0].weight,
        model.repair.temporal.message_projector[0].weight,
        model.repair.router.shared_score[0].weight,
        model.repair.decoder.weight,
        model.predictor.core.head.linear.weight,
    )
    for parameter in params:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_width64_repair_keeps_width16_predictor_and_observed_clamp():
    root = Path(__file__).resolve().parents[2]
    catalog = MigrationCatalog.load(root.parent / "E22/migration/paper_catalog.jsonl", project="E23")
    protocol = load_e23_p4_protocol(
        catalog=catalog,
        protocol_config=root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=root.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(root / "configs/e23_multi_expert_p4_width64.yaml", protocol=protocol)
    model = build_forecast_model(protocol, settings, device=torch.device("cpu"))
    assert model.repair.d_model == 64
    assert model.repair.historical.key_encoder.retrieval_dim == 64
    assert model.predictor.core.d_model == 16
    values = torch.randn(2, 96, 7)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 8:16] = False
    starts = torch.arange(12).repeat(2, 1) * 8
    model.eval()
    with torch.inference_mode():
        forecast, output = model(
            values * observed, observed, torch.zeros(2, 7), torch.ones(2, 7), starts,
        )
    assert forecast.shape == (2, 7, 96)
    assert torch.equal(output.explicit.normalized_sequence[observed], values[observed])
    assert torch.equal(output.explicit.sequence_observed, observed)


def test_width64_forecast_builder_enables_patch_variable_mixer():
    root = Path(__file__).resolve().parents[2]
    catalog = MigrationCatalog.load(root.parent / "E22/migration/paper_catalog.jsonl", project="E23")
    protocol = load_e23_p4_protocol(
        catalog=catalog,
        protocol_config=root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=root.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(root / "configs/e23_multi_expert_p4_width64.yaml", protocol=protocol)
    model = build_forecast_model(
        protocol, settings, device=torch.device("cpu"),
        forecast_mixer="patch_attention_v1",
    )
    assert model.repair.d_model == 64
    assert model.predictor.core.d_model == 16
    assert model.predictor.variable_mixer is not None
    assert model.predictor.variable_mixer.attention.num_heads == 4
    with pytest.raises(ValueError, match="forecast mixer"):
        build_forecast_model(protocol, settings, device=torch.device("cpu"),
                             forecast_mixer="unsupported")


@pytest.mark.parametrize("mode", ("topk", "topp"))
def test_width128_repair_keeps_width16_predictor(mode):
    root = Path(__file__).resolve().parents[2]
    catalog = MigrationCatalog.load(root.parent / "E22/migration/paper_catalog.jsonl", project="E23")
    protocol = load_e23_p4_protocol(
        catalog=catalog,
        protocol_config=root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=root.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(root / f"configs/e23_multi_expert_p4_width128_{mode}.yaml",
                                 protocol=protocol)
    model = build_forecast_model(protocol, settings, device=torch.device("cpu"))
    assert model.repair.d_model == 128
    assert model.repair.historical.key_encoder.retrieval_dim == 128
    assert model.repair.variable.message_projector[-1].out_features == 128
    assert model.repair.decoder.in_features == 128
    assert model.predictor.core.d_model == 16
    assert model.repair.historical.topk == (8 if mode == "topp" else 4)
    assert model.repair.variable.topk == (6 if mode == "topp" else 4)
