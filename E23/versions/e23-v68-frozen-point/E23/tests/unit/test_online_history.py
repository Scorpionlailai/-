from __future__ import annotations

import torch

from malr.models.online_history import build_online_historical_candidates
from malr.models.online_stage1 import OnlineStage1Model
from malr.models.patchtst import PatchTSTCore
from malr.prototypes.online_bank import OnlinePrototypeMemory
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder, RetrievalEncoderEMA


def _fixture() -> tuple[OnlineStage1Model, dict[str, torch.Tensor]]:
    batch, variables, patches, patch_len = 1, 2, 12, 16
    clean = torch.randn(batch, variables, patches, patch_len)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[:, 0, 5, 4:12] = False
    rho = mask.to(torch.float32).mean(dim=-1)
    state = torch.where(
        rho == 1,
        torch.zeros_like(rho, dtype=torch.long),
        torch.ones_like(rho, dtype=torch.long),
    )
    core = PatchTSTCore(
        c_in=variables,
        seq_len=96,
        pred_len=96,
        patch_len=patch_len,
        stride=8,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
    )
    retrieval = RetrievalEncoderEMA(MaskAwareRetrievalEncoder(patch_len, retrieval_dim=8), decay=0.9)
    memory = OnlinePrototypeMemory(
        patch_len=patch_len,
        patch_stride=8,
        train_end=128,
        variable_count=variables,
        group_capacity=4,
    )
    return OnlineStage1Model(core, memory=memory, retrieval=retrieval, variable_count=variables), {
        "values": torch.where(mask, clean, torch.zeros_like(clean)),
        "mask": mask,
        "rho": rho,
        "state": state,
        "starts": torch.arange(patches).unsqueeze(0) * 8,
    }


def test_shared_history_returns_empty_finite_batch() -> None:
    """An empty published memory must produce a finite fixed-shape batch."""

    model, data = _fixture()
    batch = build_online_historical_candidates(
        model.core,
        model.memory,
        model.retrieval,
        data["values"],
        data["mask"],
        data["rho"],
        data["state"],
        data["starts"],
        torch.zeros(1, 2, 12, model.core.d_model),
        max_items=4,
        similarity_threshold=0.7,
    )
    assert batch.source_nodes.shape == (1, 2, 12, 4, model.core.d_model)
    assert not batch.eligible.any()
    assert torch.isfinite(batch.features).all()
