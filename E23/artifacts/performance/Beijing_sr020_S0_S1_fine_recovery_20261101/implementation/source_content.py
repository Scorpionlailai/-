"""Exact A source candidate branches, excluding segment-only pooling/quality paths.
Generated before registration from sealed source; numeric candidate definitions retained.
"""
from math import sqrt
import torch
from integrations.v125_ablation.model.contracts import CandidateEvidence,ExpertEvidence
from integrations.v125_ablation.model.historical_messages import empty_historical


def temporal_content(self, inputs, cache, *, round_embedding=None, source_context_allowed=None):
    values, mask, validity = (inputs.patch_values, inputs.patch_mask, inputs.patch_validity)
    base = inputs.base_latents
    if values.ndim != 4:
        raise ValueError('patch values must have shape [B,C,N,P]')
    batch, variables, patches, length = values.shape
    if length != self.patch_length or mask.shape != values.shape or validity.shape != values.shape:
        raise ValueError('patch value/mask shape mismatch')
    if base.shape != (batch, variables, patches, self.d_model):
        raise ValueError('base latent shape mismatch')
    if inputs.damaged.shape != (batch, variables, patches):
        raise ValueError('damaged shape mismatch')
    observed, visible, source_encoded = (cache.observed, cache.visible, cache.source_encoded)
    if source_encoded.shape != base.shape or observed.shape != values.shape:
        raise ValueError('temporal cache shape mismatch')
    if self.contextual:
        source_encoded = source_encoded + self.source_context_projection(self.source_context_norm(base))
    query_base = base if round_embedding is None else base + round_embedding
    query = self.query_projection(query_base)
    key = self.key_projection(source_encoded)
    similarity = torch.einsum('bcnd,bcmd->bcnm', query, key) / sqrt(self.d_model)
    if source_context_allowed is not None:
        if source_context_allowed.shape != base.shape[:-1]:
            raise ValueError('temporal source context mask shape mismatch')
        original_key = self.key_projection(cache.source_encoded)
        original_similarity = torch.einsum('bcnd,bcmd->bcnm', query, original_key) / sqrt(self.d_model)
        similarity = torch.where(source_context_allowed[..., None], similarity, original_similarity)
    positions = torch.arange(patches, device=values.device)
    separation = positions[:, None] - positions[None, :]
    source_eligible = observed.any(dim=-1)
    eligible = source_eligible[:, :, None, :] & inputs.damaged[..., None].to(torch.bool)
    signed = torch.tensor((1, -1), device=values.device)[:, None, None] * separation
    lower = torch.tensor((0, *self.boundaries), device=values.device)
    upper = torch.tensor((*self.boundaries, patches), device=values.device)
    band = (signed[:, None] > lower[None, :, None, None]) & (signed[:, None] <= upper[None, :, None, None])
    band = band.reshape(-1, patches, patches)
    local_flags = torch.tensor((True, *(False for _ in self.boundaries)) * 2, device=values.device, dtype=torch.bool)
    allowed = eligible[:, :, None] & band[None, None]
    local_rank = -signed[:, None].expand(-1, len(lower), -1, -1).reshape(-1, patches, patches).to(similarity.dtype)
    rank = torch.where(local_flags[None, None, :, None, None], local_rank[None, None], similarity[:, :, None]).masked_fill(~allowed, -torch.inf)
    source_indices = rank.argmax(dim=-1).permute(0, 1, 3, 2)
    candidate_valid = allowed.any(dim=-1).permute(0, 1, 3, 2)
    available = candidate_valid.any(dim=-1)
    count = candidate_valid.sum(dim=-1)
    b_index = torch.arange(batch, device=values.device)[:, None, None, None]
    c_index = torch.arange(variables, device=values.device)[None, :, None, None]
    source = source_encoded[b_index, c_index, source_indices]
    if source_context_allowed is not None:
        original_source = cache.source_encoded[b_index, c_index, source_indices]
        source = torch.where(source_context_allowed[..., None, None], source, original_source)
    source_values = visible[b_index, c_index, source_indices]
    source_mask = observed[b_index, c_index, source_indices]
    target_mask = observed[..., None, :]
    target_values = visible[..., None, :]
    common = source_mask & target_mask
    error = ((source_values - target_values).abs() * common).sum(-1)
    error = error / common.sum(-1).clamp_min(1)
    consistency = torch.exp(-error) * common.any(dim=-1)
    fraction = source_mask.float().mean(dim=-1)
    target_fraction = observed.float().mean(dim=-1)[..., None].expand_as(fraction)
    distance = (source_indices - positions[None, None, :, None]).abs()
    relative_distance = distance.to(values.dtype) / max(patches - 1, 1)
    is_local = local_flags.to(values.dtype)
    is_local = is_local.view(1, 1, 1, -1).expand_as(fraction)
    candidates = torch.cat((base[..., None, :].expand_as(source), source, fraction[..., None], relative_distance[..., None], consistency[..., None], is_local[..., None]), dim=-1)
    candidate_messages = self.message_projector(candidates)
    selected_score = similarity.gather(dim=-1, index=source_indices)
    selected_score = selected_score + 0.5 * fraction - 0.05 * relative_distance
    selected_score = selected_score.masked_fill(~candidate_valid, -1000000000.0)
    weights = torch.softmax(selected_score, dim=-1) * candidate_valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-08)
    quality_features = torch.stack((fraction, relative_distance, consistency, target_fraction, is_local), dim=-1)
    return ExpertEvidence(torch.zeros_like(base),torch.zeros_like(base),available,
        {'candidate_valid':candidate_valid.detach(),'candidate_count':count.detach(),'source_indices':source_indices.detach()},
        CandidateEvidence(candidate_messages,quality_features,weights,candidate_valid))


def variable_content(self, inputs, cache, *, round_embedding=None, source_context_allowed=None):
    if cache.empty_evidence is not None:
        return cache.empty_evidence
    base, values = (inputs.base_latents, inputs.patch_values)
    batch, variables, patches, _ = base.shape
    k, o = (self.topk, len(self.offsets))
    if cache.visible_values.shape != (batch, variables, patches, k, o, self.patch_length):
        raise ValueError('variable cache shape mismatch')
    visible_values, source_observed = (cache.visible_values, cache.source_observed)
    lag, source_fraction, consistency = (cache.lag, cache.source_fraction, cache.consistency)
    candidate_valid, available = (cache.valid, cache.available)
    candidate_score = cache.logits
    source_state = None
    if self.contextual:
        batch_ix = torch.arange(batch, device=base.device)[:, None, None, None, None]
        source_state = self.context_norm(base[batch_ix, cache.source_ix, cache.patch_ix])
        target_state = self.context_norm(base)
        if round_embedding is not None:
            target_state = target_state + round_embedding
        query = self.context_query(target_state)[:, :, :, None, None, :]
        key = self.context_key(source_state)
        match = (query * key).sum(-1) / sqrt(self.d_model)
        adjustment = 0.5 * torch.tanh(match)
        if source_context_allowed is not None:
            if source_context_allowed.shape != base.shape[:-1]:
                raise ValueError('variable source context mask shape mismatch')
            adjustment = adjustment * source_context_allowed[..., None, None]
        candidate_score = candidate_score + adjustment
        candidate_score = candidate_score.masked_fill(~candidate_valid, -1000000000.0)
    weights = torch.softmax(candidate_score.flatten(-2), dim=-1).reshape_as(candidate_score)
    weights = weights * candidate_valid
    weights = weights / weights.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-08)
    expanded_base = base[:, :, :, None, None, :].expand(batch, variables, patches, k, o, self.d_model)
    features = torch.cat((visible_values, source_observed.to(values.dtype), expanded_base, lag[..., None], source_fraction[..., None], consistency[..., None]), dim=-1)
    candidate_messages = self.message_projector(features)
    if self.calibration is not None:
        batch_ix = torch.arange(batch, device=base.device)[:, None, None, None, None]
        source_mean = inputs.query_mean[batch_ix, cache.source_ix]
        source_std = inputs.query_std[batch_ix, cache.source_ix]
        raw_source = torch.where(source_observed, torch.where(source_observed, visible_values, 0) * source_std[..., None] + source_mean[..., None], 0)
        source_identity = self.source_embedding(cache.source_ix)
        target_identity = self.target_embedding.weight[None, :, None, None, None, :].expand_as(source_identity)
        target_mean = inputs.query_mean[:, :, None, None, None].expand_as(source_mean)
        target_std = inputs.query_std[:, :, None, None, None].expand_as(source_std)
        statistics = torch.stack((source_mean, source_std.clamp_min(1e-06).log(), target_mean, target_std.clamp_min(1e-06).log()), -1)
        conditioning = torch.cat((source_identity, target_identity, raw_source, source_observed.to(base.dtype), statistics), -1)
        correction = self.calibration(conditioning).masked_fill(~candidate_valid[..., None], 0)
        candidate_messages = candidate_messages + correction
    if source_state is not None:
        source_message = self.source_context_projection(source_state)
        if source_context_allowed is not None:
            source_message = source_message * source_context_allowed[..., None, None, None]
        candidate_messages = candidate_messages + source_message
    return ExpertEvidence(torch.zeros_like(base),torch.zeros_like(base),available,
        cache.provenance,CandidateEvidence(candidate_messages.flatten(-3,-2),cache.raw_quality.flatten(-3,-2),weights.flatten(-2),candidate_valid.flatten(-2)))


def historical_content(self, inputs, cache, *, round_embedding=None):
    base = inputs.base_latents
    batch, variables, patches, _ = base.shape
    count = getattr(self, 'topk', cache.normalized_raw.shape[-2])
    if cache.normalized_raw.shape != (batch, variables, patches, count, self.patch_length):
        raise ValueError('historical cache shape mismatch')
    if cache.empty:
        return empty_historical(base, self.patch_length, self.d_model, count)
    valid, available = (cache.valid, cache.available)
    normalized_raw, raw_quality = (cache.normalized_raw, cache.raw_quality)
    logits = cache.base_logits
    if self.contextual:
        state = self.context_norm(base)
        if round_embedding is not None:
            state = state + round_embedding
        query = self.context_query(state)
        key = self.context_key(normalized_raw)
        compatibility = (query[..., None, :] * key).sum(-1) / sqrt(self.d_model)
        logits = logits + 0.5 * torch.tanh(compatibility)
    logits = logits.masked_fill(~valid, -1000000000.0)
    weights = torch.softmax(logits, -1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-08)
    message_input = torch.cat((normalized_raw, base[..., None, :].expand(batch, variables, patches, count, self.d_model), raw_quality), dim=-1)
    candidate_messages = self.message_projector(message_input)
    return ExpertEvidence(torch.zeros_like(base),torch.zeros_like(base),available,
        cache.provenance,CandidateEvidence(candidate_messages,raw_quality,weights,valid))

