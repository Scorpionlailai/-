"""Device-resident read-only query view of published online prototype groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .online_bank import OnlinePrototypeSnapshot


@dataclass(frozen=True)
class TensorRetrieval:
    raw_patches: Tensor  # [B,C,N,K,P]
    key_similarity: Tensor  # [B,C,N,K]
    starts: Tensor  # [B,C,N,K]
    group_support: Tensor  # [B,C,N,K], published group size
    valid: Tensor  # [B,C,N,K]
    group_ids: Tensor  # [B,C,N,K]


class TensorPrototypeIndex:
    """Choose a causal group by its full center, then rank its causal members."""

    def __init__(
        self, *, device: torch.device, variable_count: int, patch_length: int,
        key_dim: int, epoch: int, centers: Tensor, group_variables: Tensor,
        group_first_end: Tensor, group_support: Tensor, group_ids: Tensor,
        member_keys: Tensor, member_raw: Tensor, member_starts: Tensor,
        member_groups: Tensor, group_member_keys: Tensor,
        group_member_raw: Tensor, group_member_starts: Tensor,
        group_member_valid: Tensor, group_ranges: tuple[tuple[int, int], ...],
        padded_centers: Tensor, padded_first_end: Tensor,
        padded_group_valid: Tensor, padded_group_index: Tensor,
        max_score_elements: int = 1_000_000,
    ) -> None:
        self.device = centers.device
        self.variable_count = variable_count
        self.patch_length = patch_length
        self.key_dim = key_dim
        self.epoch = epoch
        self.centers = centers
        self.group_variables = group_variables
        self.group_first_end = group_first_end
        self.group_support = group_support
        self.group_ids = group_ids
        self.member_keys = member_keys
        self.member_raw = member_raw
        self.member_starts = member_starts
        self.member_groups = member_groups
        self.group_member_keys = group_member_keys
        self.group_member_raw = group_member_raw
        self.group_member_starts = group_member_starts
        self.group_member_valid = group_member_valid
        self.group_ranges = group_ranges
        self.padded_centers = padded_centers
        self.padded_first_end = padded_first_end
        self.padded_group_valid = padded_group_valid
        self.padded_group_index = padded_group_index
        self.max_score_elements = max_score_elements

    @classmethod
    def from_snapshot(
        cls, snapshot: OnlinePrototypeSnapshot, *, device: torch.device,
    ) -> TensorPrototypeIndex:
        centers: list[np.ndarray] = []
        group_variables: list[int] = []
        first_ends: list[int] = []
        support: list[int] = []
        group_ids: list[int] = []
        keys: list[np.ndarray] = []
        raw: list[np.ndarray] = []
        starts: list[int] = []
        member_groups: list[int] = []
        grouped_members: list[list] = []
        patch_length = key_dim = 0
        for variable, variable_groups in enumerate(snapshot.groups):
            for group in variable_groups:
                if patch_length == 0:
                    patch_length = group.members[0].raw_patch.size
                    key_dim = group.members[0].clean_key.size
                if any(member.raw_patch.size != patch_length or member.clean_key.size != key_dim
                       for member in group.members):
                    raise ValueError("prototype snapshot has inconsistent patch/key width")
                group_index = len(centers)
                centers.append(group.center)
                group_variables.append(variable)
                first_ends.append(min(member.start + patch_length for member in group.members))
                support.append(len(group.members))
                group_ids.append(group.group_id)
                # Stable score ties must use the earliest member start.
                sorted_members = sorted(group.members, key=lambda item: item.start)
                grouped_members.append(sorted_members)
                for member in sorted_members:
                    keys.append(member.clean_key)
                    raw.append(member.raw_patch)
                    starts.append(member.start)
                    member_groups.append(group_index)

        def vectors(items: list[np.ndarray], width: int) -> Tensor:
            array = np.stack(items) if items else np.empty((0, width), dtype=np.float64)
            return torch.as_tensor(array, dtype=torch.float64, device=device).clone()

        def integers(items: list[int]) -> Tensor:
            return torch.tensor(items, dtype=torch.long, device=device)

        slots = max((len(members) for members in grouped_members), default=0)
        grouped_keys = np.zeros((len(centers), slots, key_dim), dtype=np.float64)
        grouped_raw = np.zeros((len(centers), slots, patch_length), dtype=np.float64)
        grouped_starts = np.zeros((len(centers), slots), dtype=np.int64)
        grouped_valid = np.zeros((len(centers), slots), dtype=np.bool_)
        for group_index, members in enumerate(grouped_members):
            count = len(members)
            grouped_keys[group_index, :count] = np.stack([member.clean_key for member in members])
            grouped_raw[group_index, :count] = np.stack([member.raw_patch for member in members])
            grouped_starts[group_index, :count] = [member.start for member in members]
            grouped_valid[group_index, :count] = True

        group_ranges = []
        cursor = 0
        for variable_groups in snapshot.groups:
            group_ranges.append((cursor, cursor + len(variable_groups)))
            cursor += len(variable_groups)

        max_groups = max((end - first for first, end in group_ranges), default=0)
        padded_centers = torch.zeros((len(snapshot.groups), max_groups, key_dim),
                                     dtype=torch.float64, device=device)
        padded_first_end = torch.zeros((len(snapshot.groups), max_groups),
                                       dtype=torch.long, device=device)
        padded_group_valid = torch.zeros((len(snapshot.groups), max_groups),
                                         dtype=torch.bool, device=device)
        padded_group_index = torch.zeros((len(snapshot.groups), max_groups),
                                         dtype=torch.long, device=device)
        center_tensor = vectors(centers, key_dim)
        first_end_tensor = integers(first_ends)
        for variable, (first, end) in enumerate(group_ranges):
            if first == end:
                continue
            count = end - first
            padded_centers[variable, :count] = center_tensor[first:end]
            padded_first_end[variable, :count] = first_end_tensor[first:end]
            padded_group_valid[variable, :count] = True
            padded_group_index[variable, :count] = torch.arange(first, end, device=device)

        return cls(
            device=device, variable_count=len(snapshot.groups), patch_length=patch_length,
            key_dim=key_dim, epoch=snapshot.epoch,
            centers=center_tensor, group_variables=integers(group_variables),
            group_first_end=first_end_tensor, group_support=integers(support),
            group_ids=integers(group_ids), member_keys=vectors(keys, key_dim),
            member_raw=vectors(raw, patch_length), member_starts=integers(starts),
            member_groups=integers(member_groups),
            group_member_keys=torch.as_tensor(grouped_keys, dtype=torch.float64, device=device).clone(),
            group_member_raw=torch.as_tensor(grouped_raw, dtype=torch.float64, device=device).clone(),
            group_member_starts=torch.as_tensor(grouped_starts, dtype=torch.long, device=device).clone(),
            group_member_valid=torch.as_tensor(grouped_valid, dtype=torch.bool, device=device).clone(),
            group_ranges=tuple(group_ranges),
            padded_centers=padded_centers, padded_first_end=padded_first_end,
            padded_group_valid=padded_group_valid, padded_group_index=padded_group_index,
        )

    def query(
        self, keys: Tensor, starts: Tensor, damaged: Tensor, *, topk: int = 4,
    ) -> TensorRetrieval:
        if keys.ndim != 4:
            raise ValueError("query keys must have shape [B,C,N,D]")
        batch, variables, patches, key_dim = keys.shape
        if variables > self.variable_count or (self.key_dim and key_dim != self.key_dim):
            raise ValueError("query variable/key dimension does not match index")
        if starts.shape == (batch, patches):
            starts = starts[:, None, :].expand(batch, variables, patches)
        if starts.shape != (batch, variables, patches) or damaged.shape != starts.shape:
            raise ValueError("query starts/damaged shape mismatch")
        if topk < 1 or keys.device != self.device or starts.device != self.device or damaged.device != self.device:
            raise ValueError("topk must be positive and all query tensors must be on index device")
        if not torch.isfinite(keys).all() or (starts < 0).any():
            raise ValueError("query keys/starts must be finite and nonnegative")
        shape = (batch, variables, patches, topk)
        result_raw = keys.new_zeros((*shape, self.patch_length))
        result_score = keys.new_zeros(shape)
        result_starts = torch.zeros(shape, dtype=torch.long, device=self.device)
        result_support = torch.zeros_like(result_starts)
        result_groups = torch.full_like(result_starts, -1)
        result_valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        if self.member_keys.numel() == 0 or not damaged.any():
            return TensorRetrieval(result_raw, result_score, result_starts,
                                   result_support, result_valid, result_groups)

        double_keys = keys.to(torch.float64)
        norms = torch.linalg.vector_norm(double_keys, dim=-1)
        if ((norms <= 1e-12) & damaged).any():
            raise ValueError("damaged query key has zero norm")
        normalized = double_keys / norms.clamp_min(1e-12)[..., None]
        queries = normalized.permute(1, 0, 2, 3).reshape(variables, batch * patches, key_dim)
        query_starts = starts.permute(1, 0, 2).reshape(variables, batch * patches)
        query_damaged = damaged.permute(1, 0, 2).reshape(variables, batch * patches)
        group_width = self.padded_centers.shape[1]
        member_width = self.group_member_keys.shape[1]
        take = min(topk, member_width)
        chunk_size = max(1, min(8192, self.max_score_elements // max(
            variables * group_width, variables * member_width, 1,
        )))
        variable_shape = (variables, batch * patches, topk)
        raw_by_variable = keys.new_zeros((*variable_shape, self.patch_length))
        score_by_variable = keys.new_zeros(variable_shape)
        starts_by_variable = torch.zeros(variable_shape, dtype=torch.long, device=self.device)
        support_by_variable = torch.zeros_like(starts_by_variable)
        groups_by_variable = torch.full_like(starts_by_variable, -1)
        valid_by_variable = torch.zeros(variable_shape, dtype=torch.bool, device=self.device)
        for first in range(0, batch * patches, chunk_size):
            last = min(first + chunk_size, batch * patches)
            q = queries[:, first:last]
            q_starts = query_starts[:, first:last]
            group_score = torch.bmm(q, self.padded_centers[:variables].transpose(1, 2))
            causal_group = (
                self.padded_group_valid[:variables, None, :]
                & (self.padded_first_end[:variables, None, :] <= q_starts[..., None])
                & query_damaged[:, first:last, None]
            )
            group_score = group_score.masked_fill(~causal_group, -torch.inf)
            best_local = group_score.argmax(dim=-1)
            has_group = causal_group.any(dim=-1)
            best_global = self.padded_group_index[:variables].gather(1, best_local)

            chosen_keys = self.group_member_keys[best_global]
            chosen_starts = self.group_member_starts[best_global]
            chosen_valid = self.group_member_valid[best_global]
            member_score = (q[..., None, :] * chosen_keys).sum(dim=-1)
            eligible = (
                chosen_valid
                & (chosen_starts + self.patch_length <= q_starts[..., None])
                & has_group[..., None]
            )
            member_score = member_score.masked_fill(~eligible, -torch.inf)
            order = torch.argsort(member_score, dim=-1, descending=True, stable=True)[..., :take]
            selected_valid = eligible.gather(-1, order)
            chosen_raw = self.group_member_raw[best_global]
            valid_by_variable[:, first:last, :take] = selected_valid
            raw_by_variable[:, first:last, :take] = chosen_raw.gather(
                2, order[..., None].expand(-1, -1, -1, self.patch_length),
            ).to(keys.dtype) * selected_valid[..., None]
            score_by_variable[:, first:last, :take] = member_score.gather(
                -1, order,
            ).masked_fill(~selected_valid, 0).to(keys.dtype)
            starts_by_variable[:, first:last, :take] = chosen_starts.gather(
                -1, order,
            ).masked_fill(~selected_valid, 0)
            support_by_variable[:, first:last, :take] = self.group_support[best_global][
                ..., None,
            ] * selected_valid
            groups_by_variable[:, first:last, :take] = self.group_ids[best_global][
                ..., None,
            ].masked_fill(~selected_valid, -1)

        def restore(value: Tensor) -> Tensor:
            return value.reshape(variables, batch, patches, *value.shape[2:]).transpose(0, 1).contiguous()

        return TensorRetrieval(
            restore(raw_by_variable), restore(score_by_variable),
            restore(starts_by_variable), restore(support_by_variable),
            restore(valid_by_variable), restore(groups_by_variable),
        )

    def query_two_groups(
        self, keys: Tensor, starts: Tensor, damaged: Tensor, *, member_topk: int = 4,
    ) -> TensorRetrieval:
        """Return four causal members from each of the two best causal groups."""
        if keys.ndim != 4:
            raise ValueError("query keys must have shape [B,C,N,D]")
        batch, variables, patches, key_dim = keys.shape
        if variables > self.variable_count or (self.key_dim and key_dim != self.key_dim):
            raise ValueError("query variable/key dimension does not match index")
        if starts.shape == (batch, patches):
            starts = starts[:, None, :].expand(batch, variables, patches)
        if starts.shape != (batch, variables, patches) or damaged.shape != starts.shape:
            raise ValueError("query starts/damaged shape mismatch")
        if (member_topk < 1 or keys.device != self.device or starts.device != self.device
                or damaged.device != self.device):
            raise ValueError("member_topk must be positive and query tensors must be on index device")
        if not torch.isfinite(keys).all() or (starts < 0).any():
            raise ValueError("query keys/starts must be finite and nonnegative")
        slots = 2 * member_topk
        shape = (batch, variables, patches, slots)
        result_raw = keys.new_zeros((*shape, self.patch_length))
        result_score = keys.new_zeros(shape)
        result_starts = torch.zeros(shape, dtype=torch.long, device=self.device)
        result_support = torch.zeros_like(result_starts)
        result_groups = torch.full_like(result_starts, -1)
        result_valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        if self.member_keys.numel() == 0 or not damaged.any():
            return TensorRetrieval(result_raw, result_score, result_starts,
                                   result_support, result_valid, result_groups)

        double_keys = keys.to(torch.float64)
        norms = torch.linalg.vector_norm(double_keys, dim=-1)
        if ((norms <= 1e-12) & damaged).any():
            raise ValueError("damaged query key has zero norm")
        normalized = double_keys / norms.clamp_min(1e-12)[..., None]
        queries = normalized.permute(1, 0, 2, 3).reshape(variables, batch * patches, key_dim)
        query_starts = starts.permute(1, 0, 2).reshape(variables, batch * patches)
        query_damaged = damaged.permute(1, 0, 2).reshape(variables, batch * patches)
        group_width = self.padded_centers.shape[1]
        member_width = self.group_member_keys.shape[1]
        take = min(member_topk, member_width)
        chunk_size = max(1, min(8192, self.max_score_elements // max(
            variables * group_width, 2 * variables * member_width, 1,
        )))
        variable_shape = (variables, batch * patches, 2, member_topk)
        raw = keys.new_zeros((*variable_shape, self.patch_length))
        score = keys.new_zeros(variable_shape)
        member_starts = torch.zeros(variable_shape, dtype=torch.long, device=self.device)
        support = torch.zeros_like(member_starts)
        groups = torch.full_like(member_starts, -1)
        valid = torch.zeros(variable_shape, dtype=torch.bool, device=self.device)
        variable_ix = torch.arange(variables, device=self.device)[:, None, None]
        for first in range(0, batch * patches, chunk_size):
            last = min(first + chunk_size, batch * patches)
            q = queries[:, first:last]
            q_starts = query_starts[:, first:last]
            group_score = torch.bmm(q, self.padded_centers[:variables].transpose(1, 2))
            causal_group = (
                self.padded_group_valid[:variables, None, :]
                & (self.padded_first_end[:variables, None, :] <= q_starts[..., None])
                & query_damaged[:, first:last, None]
            )
            group_score = group_score.masked_fill(~causal_group, -torch.inf)
            group_order = torch.argsort(group_score, dim=-1, descending=True, stable=True)[..., :2]
            group_valid = causal_group.gather(-1, group_order)
            if group_order.shape[-1] < 2:
                group_valid = torch.cat((group_valid, torch.zeros_like(group_valid)), dim=-1)
                group_order = torch.cat((group_order, torch.zeros_like(group_order)), dim=-1)
            global_groups = self.padded_group_index[variable_ix, group_order]
            chosen_keys = self.group_member_keys[global_groups]
            chosen_starts = self.group_member_starts[global_groups]
            eligible = (
                self.group_member_valid[global_groups]
                & (chosen_starts + self.patch_length <= q_starts[..., None, None])
                & group_valid[..., None]
            )
            member_score = (q[..., None, None, :] * chosen_keys).sum(-1)
            member_score = member_score.masked_fill(~eligible, -torch.inf)
            order = torch.argsort(member_score, dim=-1, descending=True, stable=True)[..., :take]
            selected_valid = eligible.gather(-1, order)
            valid[:, first:last, :, :take] = selected_valid
            raw[:, first:last, :, :take] = self.group_member_raw[global_groups].gather(
                3, order[..., None].expand(-1, -1, -1, -1, self.patch_length),
            ).to(keys.dtype) * selected_valid[..., None]
            score[:, first:last, :, :take] = member_score.gather(
                -1, order,
            ).masked_fill(~selected_valid, 0).to(keys.dtype)
            member_starts[:, first:last, :, :take] = chosen_starts.gather(
                -1, order,
            ).masked_fill(~selected_valid, 0)
            support[:, first:last, :, :take] = self.group_support[global_groups][
                ..., None,
            ] * selected_valid
            groups[:, first:last, :, :take] = self.group_ids[global_groups][
                ..., None,
            ].masked_fill(~selected_valid, -1)

        def restore(value: Tensor) -> Tensor:
            return (value.reshape(variables, batch, patches, slots, *value.shape[4:])
                    .transpose(0, 1).contiguous())

        return TensorRetrieval(restore(raw), restore(score), restore(member_starts),
                               restore(support), restore(valid), restore(groups))


__all__ = ["TensorPrototypeIndex", "TensorRetrieval"]
