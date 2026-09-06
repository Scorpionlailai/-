from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from malr.models.filtering import EdgeConfidenceFilter
from malr.models.relations import (
    HistoricalRelation,
    TemporalRelation,
    VariableRelation,
    stable_masked_topk,
)
from malr.prototypes.bank import PrototypeBank, PrototypeRecord, source_checksum
from malr.prototypes.factors import compute_context_factors


def _edge_filter(
    d_model: int, auxiliary_dim: int, max_edges: int
) -> EdgeConfidenceFilter:
    return EdgeConfidenceFilter(
        d_model=d_model,
        auxiliary_dim=auxiliary_dim,
        max_edges=max_edges,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )


def test_missing_sources_are_excluded_and_unselected_weights_are_zero():
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    eligible = torch.tensor([[True, False, True, True]])
    weights, chosen = stable_masked_topk(
        scores, eligible, k=2, largest=True, normalize="softmax"
    )

    assert chosen.tolist() == [[0, 2]]
    assert weights[0, 1].item() == 0.0
    assert weights[0, 3].item() == 0.0
    torch.testing.assert_close(weights.sum(-1), torch.ones(1))


def test_topk_ties_are_broken_by_ascending_candidate_index():
    scores = torch.ones(1, 4)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    _, chosen = stable_masked_topk(scores, eligible, k=3)
    assert chosen.tolist() == [[0, 1, 2]]


def test_zero_candidate_relation_returns_zero_message():
    relation = VariableRelation(d_model=8, topk=4)
    message = relation(
        query=torch.randn(1, 1, 1, 8),
        sources=torch.randn(1, 1, 1, 8),
        rho=torch.zeros(1, 1, 1),
    )
    torch.testing.assert_close(message, torch.zeros_like(message))


def test_temporal_relation_can_select_a_distant_patch_and_excludes_rho_zero():
    relation = TemporalRelation(d_model=2, topk=1, bias=False)
    with torch.no_grad():
        relation.W_k.weight.copy_(torch.eye(2))
        relation.W_v.weight.copy_(torch.eye(2))
    query = torch.tensor([[[[1.0, 0.0]] * 5]])
    sources = torch.tensor(
        [[[[0.0, 0.0], [100.0, 0.0], [0.0, 1.0], [1.0, 0.0], [3.0, 0.0]]]]
    )
    rho = torch.tensor([[[1.0, 0.0, 1.0, 1.0, 1.0]]])

    result = relation.select(query, sources, rho)

    assert result.selected[0, 0, 0, 0].item() == 4
    torch.testing.assert_close(result.message[0, 0, 0], sources[0, 0, 4])


def test_temporal_filter_uses_all_visible_candidates_then_caps_edges() -> None:
    relation = TemporalRelation(d_model=2, topk=2, bias=False)
    query = torch.randn(1, 1, 4, 2)
    sources = torch.randn(1, 1, 4, 2)
    rho = torch.tensor([[[1.0, 0.0, 0.5, 1.0]]])
    damaged = torch.tensor([[[False, True, False, False]]])

    result = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=2, max_edges=2),
    )

    assert result.filter_probabilities is not None
    assert result.hard_gate is not None
    assert result.retained_count is not None
    assert result.pool_count is not None
    assert result.candidate_count[0, 0, 1].item() == 3
    assert result.pool_count[0, 0, 1].item() == 3
    assert result.retained_count[0, 0, 1].item() == 2
    assert result.message.shape == query.shape


def test_temporal_filter_initialization_matches_legacy_message() -> None:
    torch.manual_seed(31)
    relation = TemporalRelation(d_model=2, topk=2, bias=False)
    query = torch.randn(1, 1, 4, 2)
    sources = torch.randn(1, 1, 4, 2)
    rho = torch.tensor([[[1.0, 0.0, 0.5, 1.0]]])
    damaged = torch.tensor([[[False, True, False, False]]])

    legacy = relation.select(query, sources, rho, damaged)
    filtered = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=2, max_edges=2),
    )

    torch.testing.assert_close(filtered.selected, legacy.selected)
    torch.testing.assert_close(filtered.weights, legacy.weights)
    torch.testing.assert_close(filtered.message, legacy.message)


def test_variable_filter_receives_source_availability_and_identity() -> None:
    relation = VariableRelation(d_model=2, topk=1, bias=False)
    query = torch.randn(1, 3, 2, 2)
    sources = torch.randn(1, 3, 2, 2)
    rho = torch.tensor([[[1.0, 0.0], [0.5, 1.0], [1.0, 1.0]]])
    damaged = torch.tensor([[[False, True], [False, False], [False, False]]])

    result = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=5, max_edges=1),
        variable_embeddings=torch.randn(3, 2),
    )

    assert result.filter_probabilities is not None
    assert result.hard_gate is not None
    assert result.retained_count is not None
    assert result.pool_count is not None
    assert result.candidate_count[0, 0, 1].item() == 2
    assert result.pool_count[0, 0, 1].item() == 2
    assert result.retained_count[0, 0, 1].item() == 1
    assert result.keep_loss is not None


def test_variable_filter_initialization_matches_legacy_message() -> None:
    torch.manual_seed(37)
    relation = VariableRelation(d_model=2, topk=1, bias=False)
    query = torch.randn(1, 3, 2, 2)
    sources = torch.randn(1, 3, 2, 2)
    rho = torch.tensor([[[1.0, 0.0], [0.5, 1.0], [1.0, 1.0]]])
    damaged = torch.tensor([[[False, True], [False, False], [False, False]]])

    legacy = relation.select(query, sources, rho, damaged)
    filtered = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=5, max_edges=1),
        variable_embeddings=torch.randn(3, 2),
    )

    torch.testing.assert_close(filtered.selected, legacy.selected)
    torch.testing.assert_close(filtered.weights, legacy.weights)
    torch.testing.assert_close(filtered.message, legacy.message)


def _bank_with_centers(center_a: float, center_b: float) -> PrototypeBank:
    source = np.zeros((128, 1), dtype=np.float64)
    records = []
    for prototype_id, center in enumerate((center_a, center_b)):
        segment = np.arange(64, dtype=np.float64)
        segment[24:40] = center
        factors = compute_context_factors(segment, np.ones(64, bool), train_std=1.0)
        records.append(
            PrototypeRecord(
                variable_id=0,
                start=prototype_id * 64,
                raw_segment=segment,
                center_start=24,
                center_stop=40,
                factors=factors,
                occurrence_count=2,
                mean_profile_distance=0.0,
            )
        )
    return PrototypeBank(
        dataset="fixture",
        train_end=128,
        source_checksum=source_checksum(source),
        config_sha256="fixture",
        records=(tuple(records),),
        factor_mean=np.zeros((1, 7)),
        factor_std=np.ones((1, 7)),
    )


def _bank_with_many_centers(centers: Sequence[float]) -> PrototypeBank:
    source = np.zeros((max(128, 64 * len(centers)), 1), dtype=np.float64)
    records = []
    for prototype_id, center in enumerate(centers):
        segment = np.arange(64, dtype=np.float64)
        segment[24:40] = center
        factors = compute_context_factors(segment, np.ones(64, bool), train_std=1.0)
        records.append(
            PrototypeRecord(
                variable_id=0,
                start=prototype_id * 64,
                raw_segment=segment,
                center_start=24,
                center_stop=40,
                factors=factors,
                occurrence_count=2,
                mean_profile_distance=0.0,
            )
        )
    return PrototypeBank(
        dataset="fixture",
        train_end=source.shape[0],
        source_checksum=source_checksum(source),
        config_sha256="fixture",
        records=(tuple(records),),
        factor_mean=np.zeros((1, 7)),
        factor_std=np.ones((1, 7)),
    )


def test_historical_selection_does_not_read_prototype_centers():
    teacher = nn.Linear(16, 4, bias=False)
    first = HistoricalRelation(_bank_with_centers(10.0, 20.0), topk=2, d_model=4)
    second = HistoricalRelation(_bank_with_centers(-1e6, 1e6), topk=2, d_model=4)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)
    mean = torch.zeros((1, 1))
    std = torch.ones((1, 1))

    first_result = first.select(
        context,
        context_mask,
        state,
        mean,
        std,
        torch.ones(1),
        prototype_projection=teacher,
    )
    second_result = second.select(
        context,
        context_mask,
        state,
        mean,
        std,
        torch.ones(1),
        prototype_projection=teacher,
    )

    torch.testing.assert_close(first_result.selected, second_result.selected)
    torch.testing.assert_close(first_result.weights, second_result.weights)
    assert not torch.allclose(first_result.message, second_result.message)


def test_historical_message_uses_projection_supplied_at_selection_time():
    projection = nn.Linear(16, 1, bias=False)
    with torch.no_grad():
        projection.weight.fill_(1.0)
    relation = HistoricalRelation(_bank_with_centers(10.0, 20.0), topk=1, d_model=1)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)

    first = relation.select(
        context,
        context_mask,
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=projection,
    )
    with torch.no_grad():
        projection.weight.fill_(2.0)
    second = relation.select(
        context,
        context_mask,
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=projection,
    )

    torch.testing.assert_close(second.message, first.message * 2.0)


def test_historical_filter_prepools_and_preserves_bank_ids() -> None:
    bank = _bank_with_many_centers(tuple(float(value) for value in range(20)))
    relation = HistoricalRelation(
        bank,
        topk=4,
        candidate_pool=16,
        d_model=4,
    )
    result = relation.select(
        torch.arange(96, dtype=torch.float32).view(1, 96, 1),
        torch.ones(1, 96, 1, dtype=torch.bool),
        torch.ones((1, 1, 12), dtype=torch.long),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=nn.Linear(16, 4),
        filter_query=torch.randn(1, 1, 12, 4),
        edge_filter=_edge_filter(4, auxiliary_dim=3, max_edges=4),
    )

    assert result.selected.shape == (1, 1, 12, 4)
    assert torch.all((result.selected == -1) | (result.selected < 20))
    assert result.filter_probabilities is not None
    assert result.filter_probabilities.shape[-1] == 16
    assert result.pool_count is not None
    assert torch.all(result.pool_count == 16)


def test_historical_filter_can_reject_every_prepooled_prototype() -> None:
    relation = HistoricalRelation(
        _bank_with_many_centers(tuple(float(value) for value in range(20))),
        topk=4,
        candidate_pool=16,
        d_model=4,
    )
    edge_filter = _edge_filter(4, auxiliary_dim=3, max_edges=4)
    with torch.no_grad():
        edge_filter.network[-1].weight.zero_()
        edge_filter.network[-1].bias.fill_(-20.0)

    result = relation.select(
        torch.arange(96, dtype=torch.float32).view(1, 96, 1),
        torch.ones(1, 96, 1, dtype=torch.bool),
        torch.ones((1, 1, 12), dtype=torch.long),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=nn.Linear(16, 4),
        filter_query=torch.randn(1, 1, 12, 4),
        edge_filter=edge_filter,
    )

    assert result.retained_count is not None
    assert torch.count_nonzero(result.retained_count) == 0
    torch.testing.assert_close(result.message, torch.zeros_like(result.message))
    assert torch.isfinite(result.message).all()


def test_historical_filter_initialization_matches_legacy_message() -> None:
    bank = _bank_with_many_centers(tuple(float(value) for value in range(20)))
    legacy_relation = HistoricalRelation(bank, topk=4, d_model=4)
    filtered_relation = HistoricalRelation(
        bank,
        topk=4,
        candidate_pool=16,
        d_model=4,
    )
    projection = nn.Linear(16, 4)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)
    common = {
        "context": context,
        "context_mask": context_mask,
        "state": state,
        "query_mean": torch.zeros(1, 1),
        "query_std": torch.ones(1, 1),
        "train_std": torch.ones(1),
        "prototype_projection": projection,
    }

    legacy = legacy_relation.select(**common)
    filtered = filtered_relation.select(
        **common,
        filter_query=torch.randn(1, 1, 12, 4),
        edge_filter=_edge_filter(4, auxiliary_dim=3, max_edges=4),
    )

    torch.testing.assert_close(filtered.selected, legacy.selected)
    torch.testing.assert_close(filtered.weights, legacy.weights)
    torch.testing.assert_close(filtered.message, legacy.message)


def test_empty_historical_bank_returns_zero_message():
    source = np.zeros((64, 1), dtype=np.float64)
    bank = PrototypeBank(
        dataset="fixture",
        train_end=64,
        source_checksum=source_checksum(source),
        config_sha256="fixture",
        records=((),),
        factor_mean=np.zeros((1, 7)),
        factor_std=np.ones((1, 7)),
    )
    projection = nn.Linear(16, 3)
    relation = HistoricalRelation(bank, topk=4, d_model=3)
    message = relation(
        torch.zeros(1, 96, 1),
        torch.zeros(1, 96, 1, dtype=torch.bool),
        torch.ones(1, 1, 12, dtype=torch.long),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=projection,
    )
    torch.testing.assert_close(message, torch.zeros_like(message))


def test_historical_retrieval_does_not_transfer_each_query_to_cpu(monkeypatch):
    relation = HistoricalRelation(
        _bank_with_centers(10.0, 20.0), topk=2, d_model=4
    )
    projection = nn.Linear(16, 4)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)

    def reject_cpu_transfer(self, *args, **kwargs):
        raise AssertionError("historical queries must remain in batched tensor form")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_cpu_transfer)
    result = relation.select(
        context,
        context_mask,
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=projection,
    )

    assert result.message.shape == (1, 1, 12, 4)


def test_historical_retrieval_broadcasts_normalization_over_a_batch():
    relation = HistoricalRelation(
        _bank_with_centers(10.0, 20.0), topk=2, d_model=4
    )
    projection = nn.Linear(16, 4)
    context = torch.arange(192, dtype=torch.float32).view(2, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((2, 1, 12), dtype=torch.long)

    result = relation.select(
        context,
        context_mask,
        state,
        torch.tensor([[0.0], [100.0]]),
        torch.tensor([[1.0], [2.0]]),
        torch.ones(1),
        prototype_projection=projection,
    )

    assert result.message.shape == (2, 1, 12, 4)


def test_batched_query_factors_match_the_reference_definition():
    relation = HistoricalRelation(
        _bank_with_centers(10.0, 20.0), topk=2, d_model=4
    )
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    context_mask[:, 20:23] = False
    values, available = relation._batched_query_factors(
        context, context_mask, patch_count=12, train_std=torch.tensor([2.0])
    )

    patch_index = 5
    raw_start = patch_index * 8 - 24
    reference = compute_context_factors(
        context[0, raw_start : raw_start + 64, 0].numpy(),
        context_mask[0, raw_start : raw_start + 64, 0].numpy(),
        center_slice=slice(24, 40),
        train_std=2.0,
    )

    np.testing.assert_allclose(
        values[0, 0, patch_index].numpy(), reference.values, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(
        available[0, 0, patch_index].numpy(), reference.available
    )
