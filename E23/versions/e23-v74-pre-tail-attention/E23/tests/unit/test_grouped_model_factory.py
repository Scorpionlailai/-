from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from malr.cli import build_experiment_model
from malr.config import load_config
from malr.models.relations import GroupedHistoricalRelation
from malr.data.patching import PATCH_MISSING
from malr.prototypes.bank import build_grouped_prototype_bank
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder, retrieval_state_tuple


def test_factory_enables_configured_node_gate() -> None:
    config = load_config(Path("configs/stage1_sr010_grouped_memory_imputation.yaml"))
    config = replace(
        config,
        recovery=replace(config.recovery, node_gate_enabled=True),
        ablations=replace(config.ablations, use_historical=False),
    )

    model = build_experiment_model(config, "ETTh1", "ours")

    assert model.recovery is not None
    assert model.recovery.node_gate is not None


def test_factory_loads_grouped_bank_and_freezes_retrieval_encoder() -> None:
    config = load_config(Path("configs/stage1_sr010_grouped_memory_imputation.yaml"))
    n_vars = 7
    starts = np.array([0], dtype=np.int64)
    patches = np.zeros((n_vars, 1, 16), dtype=np.float64)
    keys = np.zeros((n_vars, 1, 16), dtype=np.float64)
    keys[..., 0] = 1.0
    encoder = MaskAwareRetrievalEncoder(16, 16, hidden_dim=8)
    bank = build_grouped_prototype_bank(
        patches,
        keys,
        starts,
        dataset="ETTh1",
        train_end=8640,
        source_checksum="source",
        config_sha256="config",
        clean_checkpoint_sha256="clean",
        patch_stride=8,
        train_mean=np.zeros(n_vars),
        train_std=np.ones(n_vars),
        retrieval_state=retrieval_state_tuple(encoder),
    )

    model = build_experiment_model(config, "ETTh1", "ours", bank=bank)

    assert model.recovery is not None
    assert isinstance(model.recovery.historical, GroupedHistoricalRelation)
    assert all(
        not parameter.requires_grad
        for parameter in model.recovery.historical.retrieval_encoder.parameters()
    )

    model.train()
    patch_count = model.core.patch_num
    values = torch.randn(1, n_vars, patch_count, 16)
    patch_mask = torch.ones_like(values, dtype=torch.bool)
    patch_mask[:, 0, 1] = False
    values = torch.where(patch_mask, values, torch.zeros_like(values))
    rho = patch_mask.float().mean(dim=-1)
    state = torch.zeros(1, n_vars, patch_count, dtype=torch.long)
    state[:, 0, 1] = PATCH_MISSING
    context_mask = torch.ones(1, 96, n_vars, dtype=torch.bool)
    context_mask[:, 8:24, 0] = False
    raw_context = torch.randn(1, 96, n_vars)
    raw_context = torch.where(context_mask, raw_context, torch.zeros_like(raw_context))
    output = model(
        values,
        patch_mask,
        rho,
        state,
        torch.zeros(1, n_vars),
        torch.ones(1, n_vars),
        raw_context=raw_context,
        context_mask=context_mask,
        train_std=torch.ones(n_vars),
        patch_starts=torch.arange(patch_count).view(1, -1) * 8,
    )
    output.forecast.sum().backward()
    assert model.core.W_P.weight.grad is not None
