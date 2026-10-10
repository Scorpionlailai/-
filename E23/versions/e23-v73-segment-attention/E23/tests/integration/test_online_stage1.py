from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from malr.models.online_history import HistoricalCandidateBatch
from malr.models.online_stage1 import OnlineStage1Model
from malr.models.patchtst import PatchTSTCore
from malr.prototypes.online_bank import OnlinePrototypeMemory
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder, RetrievalEncoderEMA


def _fixture() -> tuple[OnlineStage1Model, dict[str, torch.Tensor]]:
    batch, variables, patches, patch_len = 1, 2, 12, 16
    clean = torch.randn(batch, variables, patches, patch_len)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[:, 0, 5, 4:12] = False
    mask[:, 1, 6] = False
    rho = mask.to(torch.float32).mean(dim=-1)
    state = torch.where(
        rho == 0,
        torch.full_like(rho, 2, dtype=torch.long),
        torch.where(rho == 1, torch.zeros_like(rho, dtype=torch.long), torch.ones_like(rho, dtype=torch.long)),
    )
    values = torch.where(mask, clean, torch.zeros_like(clean))
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
    model = OnlineStage1Model(core, memory=memory, retrieval=retrieval, variable_count=variables)
    return model, {
        "clean": clean,
        "values": values,
        "mask": mask,
        "rho": rho,
        "state": state,
        "query_mean": torch.zeros(batch, variables),
        "query_std": torch.ones(batch, variables),
        "starts": torch.arange(patches).unsqueeze(0) * 8,
        "forecast_target": torch.randn(batch, variables, 96),
    }


def test_online_stage1_runs_from_empty_memory_and_publishes_after_epoch() -> None:
    model, data = _fixture()
    model.begin_epoch(1)
    before = model.online_state()["memory_hash"]
    output = model(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], patch_starts=data["starts"], epoch_index=0,
    )
    assert output.forecast.shape == (1, 2, 96)
    assert output.route_index.shape == (1, 2, 12)
    assert output.edge_features.shape[-1] == 11
    assert torch.equal(output.graph.residual, torch.zeros_like(output.graph.residual))
    model.stage_observed_batch(data["clean"], data["mask"], data["rho"], data["state"], data["starts"])
    model.end_epoch(1)
    assert model.memory.snapshot().member_count > 0
    assert model.online_state()["memory_hash"] != before


def test_online_stage1_validation_forward_is_read_only_and_returns_unified_loss() -> None:
    model, data = _fixture()
    model.begin_epoch(1)
    model.stage_observed_batch(data["clean"], data["mask"], data["rho"], data["state"], data["starts"])
    model.end_epoch(1)
    before = model.online_state()
    model.eval()
    output = model(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], patch_starts=data["starts"], epoch_index=1,
    )
    after = model.online_state()
    assert before["memory_hash"] == after["memory_hash"]
    assert before["ema_hash"] == after["ema_hash"]
    result = model.training_objective(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], data["forecast_target"],
        data["clean"], patch_starts=data["starts"], epoch_index=1,
    )
    assert result.loss.total.requires_grad
    assert output.graph.active_edges.shape[:3] == (1, 2, 12)


def test_observed_staging_avoids_per_patch_tensor_scalar_reads() -> None:
    model, data = _fixture()
    model.begin_epoch(1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        staged = model.stage_observed_batch(
            data["clean"],
            data["mask"],
            data["rho"],
            data["state"],
            data["starts"],
        )

    scalar_reads = sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    assert staged > 0
    assert scalar_reads <= 4


def test_empty_historical_memory_avoids_per_patch_tensor_scalar_reads() -> None:
    model, data = _fixture()
    base_latents = torch.zeros(1, 2, 12, model.core.d_model)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        _, _, eligible, candidate_count = model._historical_candidates(
            data["values"],
            data["mask"],
            data["rho"],
            data["state"],
            data["starts"],
            base_latents,
        )

    scalar_reads = sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    assert not eligible.any()
    assert not candidate_count.any()
    assert scalar_reads <= 1


def test_legacy_private_adapter_matches_shared_history() -> None:
    """The tuple compatibility adapter must expose the shared H batch exactly."""

    model, data = _fixture()
    model.begin_epoch(1)
    model.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    model.end_epoch(1)
    base = model.core.project_clean_patches(data["values"])
    old_tuple = model._historical_candidates(
        data["values"], data["mask"], data["rho"], data["state"], data["starts"], base
    )
    shared: HistoricalCandidateBatch = model._historical_candidate_batch(
        data["values"], data["mask"], data["rho"], data["state"], data["starts"], base
    )
    for old, new in zip(
        old_tuple,
        (shared.source_nodes, shared.features, shared.eligible, shared.candidate_count),
        strict=True,
    ):
        torch.testing.assert_close(old, new)


def test_load_online_state_restores_relation_prior() -> None:
    """The model loader must restore the full model-owned online state."""

    source, data = _fixture()
    source.begin_epoch(1)
    source.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    source.end_epoch(1)
    source.relation_prior.fill_(0.25)
    restored, _ = _fixture()

    restored.load_online_state(source.online_state())

    assert restored.memory.snapshot().epoch == source.memory.snapshot().epoch
    assert restored.memory.snapshot().member_count == source.memory.snapshot().member_count
    assert restored.retrieval.frozen_epoch == source.retrieval.frozen_epoch
    torch.testing.assert_close(restored.relation_prior, source.relation_prior)


def test_populated_historical_memory_transfers_query_metadata_in_batches() -> None:
    model, data = _fixture()
    model.begin_epoch(1)
    model.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    model.end_epoch(1)
    base_latents = torch.zeros(1, 2, 12, model.core.d_model)
    damaged_state = torch.ones_like(data["state"])
    causal_starts = torch.full_like(data["starts"], 120)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        _, _, eligible, candidate_count = model._historical_candidates(
            data["values"],
            data["mask"],
            data["rho"],
            damaged_state,
            causal_starts,
            base_latents,
        )

    scalar_reads = sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    assert eligible.any()
    assert candidate_count.any()
    assert scalar_reads <= 2


def test_historical_prototypes_are_projected_in_one_batch() -> None:
    model, data = _fixture()
    model.historical_similarity_threshold = -1.0
    model.begin_epoch(1)
    model.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    model.end_epoch(1)
    base_latents = torch.zeros(1, 2, 12, model.core.d_model)
    damaged_state = torch.ones_like(data["state"])
    causal_starts = torch.full_like(data["starts"], 120)
    projection_calls = 0

    def count_projection_calls(*_args: object) -> None:
        nonlocal projection_calls
        projection_calls += 1

    handle = model.core.W_P.register_forward_hook(count_projection_calls)
    try:
        _, _, eligible, _ = model._historical_candidates(
            data["values"],
            data["mask"],
            data["rho"],
            damaged_state,
            causal_starts,
            base_latents,
        )
    finally:
        handle.remove()

    assert eligible.sum() >= 24
    assert projection_calls == 1


def test_relation_prior_update_aggregates_edges_without_scalar_reads() -> None:
    model, data = _fixture()
    output = model(
        data["values"],
        data["mask"],
        data["rho"],
        data["state"],
        data["query_mean"],
        data["query_std"],
        patch_starts=data["starts"],
        epoch_index=0,
    )
    edge_gain_target = torch.ones_like(output.predicted_gain)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        model.update_relation_prior(output, edge_gain_target)

    scalar_reads = sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    assert scalar_reads <= 2


def test_epoch_end_rekeys_all_memory_members_in_one_encoder_batch() -> None:
    model, data = _fixture()
    model.begin_epoch(1)
    model.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    encoder_calls = 0

    def count_encoder_calls(*_args: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.retrieval.ema_encoder.register_forward_hook(count_encoder_calls)
    try:
        model.end_epoch(1)
    finally:
        handle.remove()

    assert model.memory.snapshot().member_count > 1
    assert encoder_calls == 1


def test_historical_candidate_query_batches_group_scoring() -> None:
    model, data = _fixture()
    model.memory.group_similarity_threshold = 1.0
    random = np.random.default_rng(3)
    for variable in range(2):
        for start in range(100):
            model.memory.stage_patch(
                variable,
                start,
                random.normal(size=16),
                random.normal(size=8),
                observed=True,
            )
    model.memory.publish_epoch(1)
    model.historical_similarity_threshold = -1.0
    batch_size = 64
    values = data["values"].expand(batch_size, -1, -1, -1).clone()
    mask = data["mask"].expand(batch_size, -1, -1, -1).clone()
    rho = data["rho"].expand(batch_size, -1, -1).clone()
    state = torch.ones_like(rho, dtype=torch.long)
    starts = torch.full((batch_size, 12), 120, dtype=torch.long)
    base_latents = torch.zeros(batch_size, 2, 12, model.core.d_model)

    started = perf_counter()
    _, _, eligible, _ = model._historical_candidates(
        values, mask, rho, state, starts, base_latents
    )
    elapsed = perf_counter() - started

    assert eligible.sum() == batch_size * 2 * 12
    assert elapsed < 0.75
