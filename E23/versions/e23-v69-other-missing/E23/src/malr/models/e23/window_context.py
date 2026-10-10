"""Visible-only temporal context, shared across variables, before H/V/T."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .window_signature import WindowSignatureContext
from .coordinates import POLICIES, repair_scale


class _TemporalContextBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.short = nn.Conv1d(width, width, 7, padding=3*dilation,
                               dilation=dilation, groups=width)
        self.long = nn.Conv1d(width, width, 31, padding=15*dilation,
                              dilation=dilation, groups=width)
        self.channel = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.GELU(),
                                     nn.Conv1d(2*width, width, 1))

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        normalized = self.norm(x.transpose(1, 2)).transpose(1, 2)
        update = self.channel(F.gelu((self.short(normalized)+self.long(normalized))/2))
        return (x + .1*update).masked_fill(~valid, 0)


class WindowContextEncoder(nn.Module):
    """Read original visible lookback values and produce a patch latent delta.

    Variables are processed independently with shared weights. Global train
    standardization avoids amplifying nearly constant visible windows. The zero
    output start preserves the original repair model's initial function.
    """
    def __init__(self, width: int, window_length: int, patch_length: int,
                 stride: int, *, variable_context: str = "none", variable_count: int = 0, coordinates: str = "global_v1") -> None:
        super().__init__()
        if (width, window_length, patch_length, stride) != (64, 96, 16, 8):
            raise ValueError("window context requires width64, window96, patch16/stride8")
        if coordinates not in POLICIES:raise ValueError("unsupported repair coordinates")
        self.coordinates = coordinates
        self.width, self.window_length = width, window_length
        self.patch_length, self.stride = patch_length, stride
        self.patch_count = 1+(window_length+stride-patch_length)//stride
        self.input = nn.Sequential(nn.Conv1d(5, width, 1), nn.GELU())
        self.blocks = nn.ModuleList([_TemporalContextBlock(width, 1),
                                     _TemporalContextBlock(width, 2)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.variable_context = None
        if variable_context not in ("none", "signature_attention_v1", "signature_plain_v2", "signature_plain_pos_v3"):
            raise ValueError("unsupported front variable context")
        if variable_context != "none":
            with torch.random.fork_rng(devices=[]):
                self.variable_context = WindowSignatureContext(width,variable_count,statistics_conditioning=variable_context == "signature_attention_v1",learned_position=variable_context == "signature_plain_pos_v3")

    def features(self, values: Tensor, observed: Tensor, query_mean: Tensor,
                 query_std: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, T, C = values.shape
        if T != self.window_length or observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("window context requires original boolean lookback mask")
        if query_mean.shape != (B, C) or query_std.shape != (B, C):
            raise ValueError("window context normalization shape mismatch")
        valid = torch.ones_like(observed) if valid_mask is None else valid_mask
        if valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("window context validity shape mismatch")
        observed = observed & valid
        # Mask before arithmetic so hidden NaN/Inf/large placeholders cannot leak.
        safe = torch.where(observed, values, 0)
        visible = torch.where(observed, safe*query_std[:, None]+query_mean[:, None], 0)
        if self.coordinates == "robust_local_v1":
            visible = torch.where(observed,safe*query_std[:,None]/repair_scale(query_std)[:,None],0)
        mask = observed.transpose(1, 2)
        index = torch.arange(T, device=values.device)[None, None].expand(B, C, T)
        left = torch.where(mask, index, -1).cummax(-1).values
        right = torch.where(mask, index, T).flip(-1).cummin(-1).values.flip(-1)
        left_gap = torch.where(left >= 0, index-left, T).to(values.dtype)/T
        right_gap = torch.where(right < T, right-index, T).to(values.dtype)/T
        position = torch.linspace(-1, 1, T, device=values.device, dtype=values.dtype)
        position = position[None, None].expand(B, C, T)
        features = torch.stack((visible.transpose(1, 2), mask.to(values.dtype),
                                position, left_gap, right_gap), dim=2)
        validity = valid.transpose(1, 2).reshape(B*C, 1, T)
        return features.reshape(B*C, 5, T).masked_fill(~validity, 0), validity

    def pool(self, points: Tensor, patch_validity: Tensor) -> Tensor:
        B, C, N, P = patch_validity.shape
        if (N, P) != (self.patch_count, self.patch_length):
            raise ValueError("window context patch validity mismatch")
        if points.shape != (B*C, self.width, self.window_length):
            raise ValueError("window context point feature shape mismatch")
        end = (N-1)*self.stride+P
        patches = F.pad(points, (0, end-self.window_length)).unfold(-1, P, self.stride)
        patches = patches.reshape(B, C, self.width, N, P)
        weights = patch_validity[:, :, None].to(points.dtype)
        pooled = (patches*weights).sum(-1)/weights.sum(-1).clamp_min(1)
        return pooled.permute(0, 1, 3, 2)

    def forward(self, values: Tensor, observed: Tensor, query_mean: Tensor,
                query_std: Tensor, patch_validity: Tensor, *,
                valid_mask: Tensor | None = None, return_points: bool = False):
        features, valid = self.features(values, observed, query_mean, query_std, valid_mask)
        points = self.input(features).masked_fill(~valid, 0)
        for block in self.blocks:
            points = block(points, valid)
        if self.variable_context is not None:
            signature_mean, signature_std = query_mean, query_std
            if self.coordinates == "robust_local_v1":
                center = query_mean.mean(-1,keepdim=True)
                spread = torch.sqrt(((query_mean-center).square()+query_std.square()).mean(-1,keepdim=True)+.01)
                signature_mean = (query_mean-center)/spread
                signature_std = query_std/spread
            points = self.variable_context(points,observed,valid,signature_mean,signature_std)
        delta = self.output(self.output_norm(self.pool(points, patch_validity)))
        return (delta, points, features) if return_points else delta


__all__ = ["WindowContextEncoder"]
