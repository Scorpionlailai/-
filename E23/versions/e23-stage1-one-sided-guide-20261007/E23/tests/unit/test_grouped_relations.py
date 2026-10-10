import numpy as np
import torch
from torch import nn

from malr.models.relations import GroupedHistoricalRelation
from malr.prototypes.bank import build_grouped_prototype_bank


class FirstTwoEncoder(nn.Module):
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(values[..., :2], dim=-1)


class CountingFirstTwoEncoder(FirstTwoEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(values, mask)


def _relation() -> GroupedHistoricalRelation:
    starts = np.arange(3, dtype=np.int64)
    patches = np.array([[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    keys = np.array([[[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]]])
    bank = build_grouped_prototype_bank(
        patches,
        keys,
        starts,
        dataset="fixture",
        train_end=20,
        source_checksum="source",
        config_sha256="config",
        clean_checkpoint_sha256="clean",
        patch_stride=2,
        group_capacity=4,
        max_groups_per_variable=4,
    )
    return GroupedHistoricalRelation(
        bank,
        retrieval_encoder=FirstTwoEncoder(),
        topk=4,
        similarity_threshold=0.7,
        min_visible_points=2,
        overlap_exclusion=0,
        patch_len=4,
        patch_stride=2,
        d_model=4,
    )


def test_grouped_history_applies_same_mask_and_hard_threshold() -> None:
    relation = _relation()
    context = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])
    mask = torch.tensor([[[1], [1], [0], [0]]], dtype=torch.bool)
    state = torch.tensor([[[1, 0]]], dtype=torch.long)
    mean = torch.zeros(1, 1)
    std = torch.ones(1, 1)
    projection = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    result = relation.select(
        context,
        mask,
        state,
        mean,
        std,
        torch.ones(1),
        prototype_projection=projection,
        patch_starts=torch.tensor([[0, 2]]),
    )

    assert result.message.shape == (1, 1, 2, 4)
    assert result.selected.shape[-1] == 4
    assert result.selected[0, 0, 0, 0].item() == 0
    assert result.candidate_count[0, 0, 0].item() >= 1
    result.message.sum().backward()
    assert projection.weight.grad is not None


def test_grouped_history_skips_when_visible_points_are_insufficient() -> None:
    relation = _relation()
    context = torch.ones(1, 4, 1)
    mask = torch.tensor([[[1], [0], [0], [0]]], dtype=torch.bool)
    state = torch.tensor([[[1, 0]]], dtype=torch.long)
    result = relation.select(
        context,
        mask,
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=nn.Identity(),
        patch_starts=torch.tensor([[0, 2]]),
    )
    assert torch.count_nonzero(result.message) == 0
    assert result.candidate_count.sum().item() == 0


def test_grouped_history_batches_retrieval_encoder_calls() -> None:
    relation = _relation()
    encoder = CountingFirstTwoEncoder()
    relation.retrieval_encoder = encoder
    context = torch.tensor(
        [
            [[1.0], [0.0], [1.0], [0.0]],
            [[1.0], [0.0], [1.0], [0.0]],
        ]
    )
    mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones(2, 1, 2, dtype=torch.long)

    result = relation.select(
        context,
        mask,
        state,
        torch.zeros(2, 1),
        torch.ones(2, 1),
        torch.ones(1),
        prototype_projection=nn.Linear(4, 4, bias=False),
        patch_starts=torch.tensor([[0, 2], [4, 6]]),
    )

    assert result.message.shape == (2, 1, 2, 4)
    assert encoder.forward_calls == 2
