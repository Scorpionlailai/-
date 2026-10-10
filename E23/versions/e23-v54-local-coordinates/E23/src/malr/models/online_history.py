"""Read-only historical-candidate construction shared by online protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor
import torch

from malr.data.patching import PATCH_OBSERVED
from malr.prototypes.online_bank import OnlinePrototypeMemory
from malr.prototypes.retrieval import RetrievalEncoderEMA

from .patchtst import PatchTSTCore


@dataclass(frozen=True)
class HistoricalCandidateBatch:
    source_nodes: Tensor
    features: Tensor
    eligible: Tensor
    candidate_count: Tensor


def build_online_historical_candidates(
    core: PatchTSTCore,
    memory: OnlinePrototypeMemory,
    retrieval: RetrievalEncoderEMA,
    values: Tensor,
    mask: Tensor,
    rho: Tensor,
    state: Tensor,
    starts: Tensor,
    base_latents: Tensor,
    *,
    max_items: int,
    similarity_threshold: float,
    query_mean: Tensor | None = None,
    query_std: Tensor | None = None,
) -> HistoricalCandidateBatch:
    """Return causal H candidates without mutating online-memory state."""

    batch, variables, patches, d_model = base_latents.shape
    width = max_items
    source_nodes = base_latents.new_zeros(batch, variables, patches, width, d_model)
    features = base_latents.new_zeros(batch, variables, patches, width, 11)
    eligible = torch.zeros(batch, variables, patches, width, dtype=torch.bool, device=base_latents.device)
    candidate_count = torch.zeros(batch, variables, patches, dtype=torch.long, device=base_latents.device)
    if memory.snapshot().member_count == 0:
        return HistoricalCandidateBatch(source_nodes, features, eligible, candidate_count)
    flat_values = values.reshape(batch * variables * patches, -1)
    flat_mask = mask.reshape(batch * variables * patches, -1)
    query_keys = retrieval.encode_frozen(flat_values, flat_mask).reshape(batch, variables, patches, -1)
    query_keys_cpu = query_keys.detach().cpu().numpy()
    state_cpu = state.detach().cpu().numpy()
    starts_cpu = starts.detach().cpu().numpy()
    candidate_count_cpu = np.zeros((batch, variables, patches), dtype=np.int64)
    selected: list[tuple[int, int, int, int, float, np.ndarray]] = []
    damaged_indices = np.argwhere(state_cpu != PATCH_OBSERVED)
    if damaged_indices.size:
        query_starts = starts_cpu[damaged_indices[:, 0], damaged_indices[:, 2]]
        damaged_keys = query_keys_cpu[
            damaged_indices[:, 0],
            damaged_indices[:, 1],
            damaged_indices[:, 2],
        ]
        query_results = memory.query_batch(
            damaged_indices[:, 1],
            query_starts,
            damaged_keys,
            max_items=width,
        )
        for indices, key, members in zip(damaged_indices, damaged_keys, query_results, strict=True):
            b, variable, patch = (int(index) for index in indices)
            kept = [
                member
                for member in members
                if float(np.dot(key, member.clean_key)) >= similarity_threshold
            ]
            candidate_count_cpu[b, variable, patch] = len(kept)
            for slot, member in enumerate(kept):
                similarity = float(np.dot(key, member.clean_key))
                selected.append((b, variable, patch, slot, similarity, member.raw_patch))
    candidate_count = torch.as_tensor(candidate_count_cpu, device=values.device)
    if not selected:
        return HistoricalCandidateBatch(source_nodes, features, eligible, candidate_count)

    indices = torch.as_tensor(
        np.asarray([record[:4] for record in selected], dtype=np.int64),
        device=values.device,
    )
    batch_index, variable_index, patch_index, slot_index = indices.unbind(dim=1)
    raw_patches = torch.as_tensor(
        np.stack([record[5] for record in selected]),
        dtype=values.dtype,
        device=values.device,
    )
    if (query_mean is None) != (query_std is None):
        raise ValueError("historical query mean and std must be provided together")
    if query_mean is not None and query_std is not None:
        if query_mean.shape != (batch, variables) or query_std.shape != query_mean.shape:
            raise ValueError("historical query statistics have invalid shape")
        selected_mean = query_mean[batch_index, variable_index].unsqueeze(-1)
        selected_std = query_std[batch_index, variable_index].unsqueeze(-1)
        raw_patches = (raw_patches - selected_mean) / selected_std
    source_nodes[batch_index, variable_index, patch_index, slot_index] = core.W_P(
        raw_patches
    )
    selected_features = values.new_zeros(len(selected), 11)
    selected_features[:, 0] = torch.as_tensor(
        [record[4] for record in selected], dtype=values.dtype, device=values.device
    )
    selected_features[:, 1:3] = 1.0
    selected_features[:, 3] = rho[batch_index, variable_index, patch_index]
    selected_features[:, 4] = mask[batch_index, variable_index, patch_index].to(values.dtype).mean(dim=-1)
    selected_features[:, 6] = variable_index.to(values.dtype) / max(variables - 1, 1)
    features[batch_index, variable_index, patch_index, slot_index] = selected_features
    eligible[batch_index, variable_index, patch_index, slot_index] = True
    return HistoricalCandidateBatch(source_nodes, features, eligible, candidate_count)


__all__ = ["HistoricalCandidateBatch", "build_online_historical_candidates"]
