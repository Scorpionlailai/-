"""Directed cross-variable relation messages with train-fitted gain statistics."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .candidates_v2 import VariableCandidateBatch
from .relation_outputs import RelationBranchOutput, WeightedAggregationPlan


def signed_lag_bucket(lag: Tensor, band: Tensor, band_count: int) -> Tensor:
    """Map a lag and its absolute-distance band to a signed embedding bucket."""

    if type(band_count) is not int or band_count <= 0:
        raise ValueError("band_count must be a positive integer")
    if lag.shape != band.shape:
        raise ValueError("lag and band must have the same shape")
    if lag.is_floating_point() and not torch.isfinite(lag).all():
        raise ValueError("lag must be finite")
    if band.is_floating_point() and not torch.isfinite(band).all():
        raise ValueError("band must be finite")
    if band.is_floating_point() or band.dtype == torch.bool:
        raise TypeError("band must have an integer dtype")
    if band.numel() and (band.lt(0).any() or band.ge(band_count).any()):
        raise ValueError("band must lie in [0, band_count)")
    sync = torch.full_like(band, band_count)
    negative = band
    positive = band_count + 1 + band
    return torch.where(lag < 0, negative, torch.where(lag > 0, positive, sync))


class VariableRelationStatistics(nn.Module):
    """Train-only EMA gain statistics for ordered variable and lag relations."""

    def __init__(
        self,
        variable_count: int,
        signed_bucket_count: int,
        decay: float = 0.9,
        min_count: int = 128,
    ) -> None:
        super().__init__()
        if type(variable_count) is not int or variable_count <= 0:
            raise ValueError("variable_count must be a positive integer")
        if type(signed_bucket_count) is not int or signed_bucket_count <= 0:
            raise ValueError("signed_bucket_count must be a positive integer")
        if not isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must be finite and lie in [0, 1)")
        if type(min_count) is not int or min_count <= 0:
            raise ValueError("min_count must be a positive integer")
        self.variable_count = variable_count
        self.signed_bucket_count = signed_bucket_count
        self.decay = float(decay)
        self.min_count = min_count
        shape = (variable_count, variable_count, signed_bucket_count)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("second_moment", torch.zeros(shape))
        self.register_buffer("count", torch.zeros(shape, dtype=torch.long))

    def _validate_indices(self, source: Tensor, target: Tensor, bucket: Tensor) -> None:
        if source.shape != target.shape or source.shape != bucket.shape:
            raise ValueError("source, target, and bucket must share a shape")
        for name, value, upper in (
            ("source", source, self.variable_count),
            ("target", target, self.variable_count),
            ("bucket", bucket, self.signed_bucket_count),
        ):
            if value.is_floating_point() or value.dtype == torch.bool:
                raise TypeError(f"{name} must have an integer dtype")
            if value.numel() and (value.lt(0).any() or value.ge(upper).any()):
                raise ValueError(f"{name} indices are out of range")

    @torch.no_grad()
    def update(
        self,
        source_variable: Tensor,
        target_variable: Tensor,
        signed_bucket: Tensor,
        gain: Tensor,
        eligible: Tensor,
    ) -> None:
        """Incorporate explicitly admitted training gains, grouped by relation."""

        self._validate_indices(source_variable, target_variable, signed_bucket)
        if gain.shape != source_variable.shape or eligible.shape != source_variable.shape:
            raise ValueError("gain and eligible must match relation index shape")
        if not gain.is_floating_point():
            raise TypeError("gain must be floating-point")
        if not torch.isfinite(gain).all():
            raise ValueError("gain must be finite")
        if eligible.dtype != torch.bool:
            raise TypeError("eligible must be bool")
        device = self.mean.device
        source = source_variable.to(device=device, dtype=torch.long).reshape(-1)
        target = target_variable.to(device=device, dtype=torch.long).reshape(-1)
        bucket = signed_bucket.to(device=device, dtype=torch.long).reshape(-1)
        values = gain.detach().to(device=device, dtype=self.mean.dtype).reshape(-1)
        take = eligible.to(device=device, dtype=torch.bool).reshape(-1)
        if not take.any():
            return
        flat = ((source * self.variable_count + target) * self.signed_bucket_count + bucket)
        flat = flat[take]
        values = values[take]
        total = self.mean.numel()
        observations = torch.zeros(total, device=device, dtype=self.mean.dtype)
        gain_sum = torch.zeros_like(observations)
        second_sum = torch.zeros_like(observations)
        observations.scatter_add_(0, flat, torch.ones_like(values))
        gain_sum.scatter_add_(0, flat, values)
        second_sum.scatter_add_(0, flat, values.square())
        present = observations.gt(0)
        batch_mean = gain_sum / observations.clamp_min(1.0)
        batch_second = second_sum / observations.clamp_min(1.0)
        old_count = self.count.reshape(-1)
        old_mean = self.mean.reshape(-1)
        old_second = self.second_moment.reshape(-1)
        initialized = old_count.gt(0)
        new_mean = torch.where(
            initialized,
            self.decay * old_mean + (1.0 - self.decay) * batch_mean,
            batch_mean,
        )
        new_second = torch.where(
            initialized,
            self.decay * old_second + (1.0 - self.decay) * batch_second,
            batch_second,
        )
        old_mean[present] = new_mean[present]
        old_second[present] = new_second[present]
        old_count[present] += observations[present].to(dtype=old_count.dtype)

    def features(
        self, source_variable: Tensor, target_variable: Tensor, signed_bucket: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return read-only mean, standard deviation, and stability indicators."""

        self._validate_indices(source_variable, target_variable, signed_bucket)
        device = self.mean.device
        source = source_variable.to(device=device, dtype=torch.long)
        target = target_variable.to(device=device, dtype=torch.long)
        bucket = signed_bucket.to(device=device, dtype=torch.long)
        mean = self.mean[source, target, bucket]
        second = self.second_moment[source, target, bucket]
        count = self.count[source, target, bucket]
        std = (second - mean.square()).clamp_min(0.0).sqrt()
        stable = count.ge(self.min_count)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("relation statistics must be finite")
        output_device = source_variable.device
        return mean.to(output_device), std.to(output_device), stable.to(output_device)


def _masked_softmax(score: Tensor, selected: Tensor) -> Tensor:
    """Return exact-zero weights when no candidate was selected."""

    masked = score.masked_fill(~selected, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    numerator = torch.where(selected, torch.exp(masked - maximum), torch.zeros_like(score))
    return numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(score.dtype).tiny
    )


def _select_one_per_source(
    source_variable: Tensor,
    eligible: Tensor,
    gain: Tensor,
    uncertainty: Tensor,
    variable_count: int,
    topk: int,
    gain_threshold: float,
) -> Tensor:
    """Rank by gain minus uncertainty, then keep at most one lag per source."""

    score = gain - uncertainty
    admissible = eligible & gain.gt(gain_threshold)
    selected_by_source = torch.zeros_like(admissible)
    if source_variable.shape[-1] == 0:
        return selected_by_source
    for source in range(variable_count):
        group = admissible & source_variable.eq(source)
        best = score.masked_fill(~group, -torch.inf).argmax(dim=-1, keepdim=True)
        best_mask = torch.zeros_like(group).scatter(-1, best, True)
        selected_by_source |= best_mask & group.any(dim=-1, keepdim=True)
    ranked = score.masked_fill(~selected_by_source, -torch.inf)
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    ranked_selected = selected_by_source.gather(-1, order)
    rank = torch.arange(order.shape[-1], device=order.device).view(
        *((1,) * (order.ndim - 1)), -1
    )
    kept = ranked_selected & rank.lt(topk)
    return torch.zeros_like(selected_by_source).scatter(-1, order, kept)


class DirectedVariableRelation(nn.Module):
    """Translate selected ordered source-variable latents into target space."""

    def __init__(
        self,
        d_model: int,
        feature_dim: int,
        variable_count: int,
        signed_bucket_count: int,
        topk: int = 4,
        candidate_gain_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if type(variable_count) is not int or variable_count <= 1:
            raise ValueError("variable_count must be an integer greater than one")
        if type(signed_bucket_count) is not int or signed_bucket_count < 3 or signed_bucket_count % 2 != 1:
            raise ValueError("signed_bucket_count must be an odd integer of at least three")
        if type(topk) is not int or topk <= 0:
            raise ValueError("topk must be a positive integer")
        if not isfinite(candidate_gain_threshold):
            raise ValueError("candidate_gain_threshold must be finite")
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.variable_count = variable_count
        self.signed_bucket_count = signed_bucket_count
        self.lag_band_count = (signed_bucket_count - 1) // 2
        self.topk = topk
        self.candidate_gain_threshold = float(candidate_gain_threshold)
        hidden = max(d_model, feature_dim)
        self.source_embedding = nn.Embedding(variable_count, d_model)
        self.target_embedding = nn.Embedding(variable_count, d_model)
        self.lag_embedding = nn.Embedding(signed_bucket_count, d_model)
        self.translator = nn.Sequential(
            nn.Linear(4 * d_model + feature_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        edge_width = 2 * d_model + feature_dim + 3
        self.gain_head = nn.Sequential(
            nn.Linear(edge_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(edge_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.constant_(self.gain_head[-1].bias, 1.0)

    def _validate(
        self,
        nodes: Tensor,
        candidates: VariableCandidateBatch,
        statistics: VariableRelationStatistics,
    ) -> None:
        if not isinstance(statistics, VariableRelationStatistics):
            raise TypeError("statistics must be VariableRelationStatistics")
        if statistics.variable_count != self.variable_count:
            raise ValueError("statistics and relation variable counts must agree")
        if statistics.signed_bucket_count != self.signed_bucket_count:
            raise ValueError("statistics and relation signed bucket counts must agree")
        if nodes.ndim != 4 or nodes.shape[1] != self.variable_count or nodes.shape[-1] != self.d_model:
            raise ValueError("nodes must have shape [B,C,N,D]")
        if not nodes.is_floating_point() or not torch.isfinite(nodes).all():
            raise ValueError("nodes must be finite floating-point tensors")
        expected = (*nodes.shape[:-1], candidates.eligible.shape[-1])
        if candidates.eligible.shape != expected:
            raise ValueError("candidate tensors must have shape [B,C,N,K]")
        if candidates.eligible.dtype != torch.bool:
            raise TypeError("candidate eligibility must be bool")
        if candidates.features.shape != (*expected, self.feature_dim):
            raise ValueError("features must have shape [B,C,N,K,feature_dim]")
        if not candidates.features.is_floating_point() or not torch.isfinite(candidates.features).all():
            raise ValueError("candidate features must be finite floating-point tensors")
        if candidates.candidate_count.shape != nodes.shape[:-1]:
            raise ValueError("candidate_count must have shape [B,C,N]")
        metadata = (
            ("source_variable", candidates.source_variable, self.variable_count),
            ("source_patch", candidates.source_patch, nodes.shape[2]),
            ("lag_band", candidates.lag_band, self.lag_band_count),
        )
        if candidates.lag.shape != expected or candidates.is_synchronous.shape != expected:
            raise ValueError("candidate metadata must have shape [B,C,N,K]")
        if candidates.is_synchronous.dtype != torch.bool:
            raise TypeError("is_synchronous must be bool")
        for name, value, upper in metadata:
            if value.shape != expected:
                raise ValueError("candidate metadata must have shape [B,C,N,K]")
            if value.is_floating_point() or value.dtype == torch.bool:
                raise TypeError(f"{name} must have an integer dtype")
            if value.numel() and (value.lt(0).any() or value.ge(upper).any()):
                raise ValueError(f"{name} is out of range")
        if candidates.lag.is_floating_point() and not torch.isfinite(candidates.lag).all():
            raise ValueError("lag must be finite")

    def project_candidate_messages(
        self, nodes: Tensor, candidates: VariableCandidateBatch
    ) -> Tensor:
        """Translate each ordered variable candidate into target space."""

        source_variable = candidates.source_variable.to(device=nodes.device, dtype=torch.long)
        source_patch = candidates.source_patch.to(device=nodes.device, dtype=torch.long)
        source = nodes[
            torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1, 1, 1),
            source_variable,
            source_patch,
        ]
        signed_bucket = signed_lag_bucket(
            candidates.lag.to(device=nodes.device),
            candidates.lag_band.to(device=nodes.device, dtype=torch.long),
            self.lag_band_count,
        ).to(dtype=torch.long)
        target_variable = torch.arange(nodes.shape[1], device=nodes.device).view(1, -1, 1, 1)
        target_variable = target_variable.expand_as(source_variable)
        features = candidates.features.to(device=nodes.device, dtype=nodes.dtype)
        source_identity = self.source_embedding(source_variable)
        target_identity = self.target_embedding(target_variable)
        lag_identity = self.lag_embedding(signed_bucket)
        return self.translator(
            torch.cat((source, source_identity, target_identity, lag_identity, features), dim=-1)
        )

    def replay_message(
        self, candidate_messages: Tensor, plan: WeightedAggregationPlan
    ) -> Tensor:
        """Aggregate projected candidate messages with a previously computed plan."""

        message = (plan.coefficients.unsqueeze(-1) * candidate_messages).sum(dim=-2)
        return torch.where(
            plan.available.unsqueeze(-1), message, torch.zeros_like(message)
        )

    def forward_with_plan(
        self,
        nodes: Tensor,
        candidates: VariableCandidateBatch,
        statistics: VariableRelationStatistics,
    ) -> tuple[RelationBranchOutput, WeightedAggregationPlan]:
        self._validate(nodes, candidates, statistics)
        source_variable = candidates.source_variable.to(device=nodes.device, dtype=torch.long)
        signed_bucket = signed_lag_bucket(
            candidates.lag.to(device=nodes.device),
            candidates.lag_band.to(device=nodes.device, dtype=torch.long),
            self.lag_band_count,
        ).to(dtype=torch.long)
        target_variable = torch.arange(nodes.shape[1], device=nodes.device).view(1, -1, 1, 1)
        target_variable = target_variable.expand_as(source_variable)
        features = candidates.features.to(device=nodes.device, dtype=nodes.dtype)
        candidate_messages = self.project_candidate_messages(nodes, candidates)
        mean, std, stable = statistics.features(source_variable, target_variable, signed_bucket)
        mean = mean.to(device=nodes.device, dtype=nodes.dtype)
        std = std.to(device=nodes.device, dtype=nodes.dtype)
        stable = stable.to(device=nodes.device, dtype=nodes.dtype)
        edge_input = torch.cat(
            (
                nodes.unsqueeze(-2).expand_as(candidate_messages),
                candidate_messages,
                features,
                mean.unsqueeze(-1),
                std.unsqueeze(-1),
                stable.unsqueeze(-1),
            ),
            dim=-1,
        )
        candidate_gain = self.gain_head(edge_input).squeeze(-1)
        candidate_uncertainty = F.softplus(self.uncertainty_head(edge_input).squeeze(-1))
        candidate_uncertainty = candidate_uncertainty + torch.finfo(nodes.dtype).eps
        eligible = candidates.eligible.to(device=nodes.device, dtype=torch.bool)
        selected = _select_one_per_source(
            source_variable,
            eligible,
            candidate_gain,
            candidate_uncertainty,
            self.variable_count,
            self.topk,
            self.candidate_gain_threshold,
        )
        score = candidate_gain - candidate_uncertainty
        weights = _masked_softmax(score, selected)
        available = selected.any(dim=-1)
        plan = WeightedAggregationPlan(weights, selected, available)
        message = self.replay_message(candidate_messages, plan)
        gain = (weights * candidate_gain).sum(dim=-1)
        uncertainty = (weights * candidate_uncertainty).sum(dim=-1)
        gain = torch.where(available, gain, torch.zeros_like(gain))
        uncertainty = torch.where(available, uncertainty, torch.ones_like(uncertainty))
        output = RelationBranchOutput(
            message=message,
            predicted_gain=gain,
            uncertainty=uncertainty,
            available=available,
            candidate_messages=candidate_messages,
            candidate_predicted_gain=candidate_gain,
            candidate_uncertainty=candidate_uncertainty,
            candidate_eligible=eligible,
            candidate_selected=selected,
            diagnostics={
                "signed_bucket": signed_bucket,
                "statistics_mean": mean,
                "statistics_std": std,
                "statistics_stable": stable,
            },
        )
        return output, plan

    def forward(
        self,
        nodes: Tensor,
        candidates: VariableCandidateBatch,
        statistics: VariableRelationStatistics,
    ) -> RelationBranchOutput:
        output, _ = self.forward_with_plan(nodes, candidates, statistics)
        return output


__all__ = [
    "DirectedVariableRelation",
    "VariableRelationStatistics",
    "signed_lag_bucket",
]
