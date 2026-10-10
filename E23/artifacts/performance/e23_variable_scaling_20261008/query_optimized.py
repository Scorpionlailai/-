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
        del chosen_keys
        eligible = (
            chosen_valid
            & (chosen_starts + self.patch_length <= q_starts[..., None])
            & has_group[..., None]
        )
        member_score = member_score.masked_fill(~eligible, -torch.inf)
        order = torch.argsort(member_score, dim=-1, descending=True, stable=True)[..., :take]
        selected_valid = eligible.gather(-1, order)
        selected_raw = self.group_member_raw[best_global[..., None], order]
        valid_by_variable[:, first:last, :take] = selected_valid
        raw_by_variable[:, first:last, :take] = selected_raw.to(keys.dtype) * selected_valid[..., None]
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
