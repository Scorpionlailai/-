import numpy as np
import pytest
import torch

from malr.models.e23.model import E23ForecastModel, MultiExpertRepairEncoder
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel
from malr.prototypes.online_bank import OnlinePrototypeMemory
from integrations.e23.train import build_forecast_model
from tests.integration.test_e23_stage1 import _protocol_settings


def _model(archive=None, bounded=False):
    repair = MultiExpertRepairEncoder(
        variable_count=2, window_length=24, patch_length=8, stride=4,
        d_model=8, retrieval_dim=8, historical_threshold=-1.0,
        historical_archive=archive,
        bounded_history=bounded,
    )
    core = PatchTSTCore(
        c_in=2, seq_len=24, pred_len=6, patch_len=8, stride=4,
        d_model=8, n_heads=2, d_ff=16, n_layers=1, dropout=0.0,
    )
    return E23ForecastModel(repair, MaskAwareForecastModel(core))


@pytest.mark.parametrize("bounded", [False, True])
def test_zero_gated_adapter_preserves_v3_and_trains_after_gate_opens(bounded):
    torch.manual_seed(71)
    archive = torch.arange(128, dtype=torch.float32).reshape(64, 2) / 10
    base = _model()
    adapted = _model(archive, bounded=bounded)
    missing, unexpected = adapted.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(name.startswith("repair.history_adapter.") for name in missing)
    assert not unexpected
    adapted.repair.historical.retrieval.load_state_dict(base.repair.historical.retrieval.state_dict())
    memory = OnlinePrototypeMemory(patch_len=8, patch_stride=4, train_end=64, variable_count=2)
    for variable in range(2):
        memory.stage_patch(variable, 0, archive[:8, variable].numpy(),
                           np.array([1., 0., 0., 0., 0., 0., 0., 0.]), observed=True)
    snapshot = memory.publish_epoch(1)
    base.repair.historical.refresh_snapshot(snapshot)
    adapted.repair.historical.refresh_snapshot(snapshot)
    values = torch.randn(2, 24, 2)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 8:12] = False
    starts = torch.arange(6).repeat(2, 1) * 4 + 32
    mean, std = torch.zeros(2, 2), torch.ones(2, 2)
    base.eval()
    adapted.eval()
    old_forecast, old_output = base(values * observed, observed, mean, std, starts)
    new_forecast, new_output = adapted(values * observed, observed, mean, std, starts)
    assert new_output.evidence["historical_variable"].available.any()
    assert torch.equal(new_forecast, old_forecast)
    assert torch.equal(new_output.explicit.repaired_patches, old_output.explicit.repaired_patches)
    new_forecast.square().mean().backward()
    assert adapted.repair.history_adapter.gates.grad.abs().sum() > 0
    adapted.zero_grad(set_to_none=True)
    with torch.no_grad():
        adapted.repair.history_adapter.gates.fill_(0.2)
    changed_forecast, _ = adapted(values * observed, observed, mean, std, starts)
    assert not torch.equal(changed_forecast, old_forecast)
    changed_forecast.square().mean().backward()
    expert = adapted.repair.history_adapter.expert
    assert expert.source_embedding.weight.grad.abs().sum() > 0
    assert expert.message_projector[0].weight.grad.abs().sum() > 0


def test_bounded_variant_builder_selects_only_bounded_adapter():
    protocol, settings = _protocol_settings()
    v1 = build_forecast_model(protocol, settings, device=torch.device("cpu"),
                              history_adapter="aligned_v1")
    v2 = build_forecast_model(protocol, settings, device=torch.device("cpu"),
                              history_adapter="aligned_bounded_v2")
    assert v1.repair.history_adapter.expert.bounded_values is False
    assert v2.repair.history_adapter.expert.bounded_values is True
    assert v1.state_dict().keys() == v2.state_dict().keys()
