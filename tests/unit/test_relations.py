import numpy as np
import torch
from torch import nn

from malr.models.relations import (
    HistoricalRelation,
    TemporalRelation,
    VariableRelation,
    stable_masked_topk,
)
from malr.prototypes.bank import PrototypeBank, PrototypeRecord, source_checksum
from malr.prototypes.factors import compute_context_factors


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


def test_historical_selection_does_not_read_prototype_centers():
    teacher = nn.Linear(16, 4, bias=False)
    first = HistoricalRelation(_bank_with_centers(10.0, 20.0), teacher, topk=2)
    second = HistoricalRelation(_bank_with_centers(-1e6, 1e6), teacher, topk=2)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)
    mean = torch.zeros((1, 1))
    std = torch.ones((1, 1))

    first_result = first.select(context, context_mask, state, mean, std, torch.ones(1))
    second_result = second.select(context, context_mask, state, mean, std, torch.ones(1))

    torch.testing.assert_close(first_result.selected, second_result.selected)
    torch.testing.assert_close(first_result.weights, second_result.weights)
    assert not torch.allclose(first_result.message, second_result.message)


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
    relation = HistoricalRelation(bank, nn.Linear(16, 3), topk=4)
    message = relation(
        torch.zeros(1, 96, 1),
        torch.zeros(1, 96, 1, dtype=torch.bool),
        torch.ones(1, 1, 12, dtype=torch.long),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
    )
    torch.testing.assert_close(message, torch.zeros_like(message))


def test_historical_retrieval_does_not_transfer_each_query_to_cpu(monkeypatch):
    relation = HistoricalRelation(
        _bank_with_centers(10.0, 20.0), nn.Linear(16, 4), topk=2
    )
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
    )

    assert result.message.shape == (1, 1, 12, 4)


def test_historical_retrieval_broadcasts_normalization_over_a_batch():
    relation = HistoricalRelation(
        _bank_with_centers(10.0, 20.0), nn.Linear(16, 4), topk=2
    )
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
    )

    assert result.message.shape == (2, 1, 12, 4)


def test_batched_query_factors_match_the_reference_definition():
    relation = HistoricalRelation(
        _bank_with_centers(10.0, 20.0), nn.Linear(16, 4), topk=2
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
