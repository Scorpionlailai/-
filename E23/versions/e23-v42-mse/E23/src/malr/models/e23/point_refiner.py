"""Point residual repair with full history context and a fixed output scope."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.patching import patchify_preserving_values
from malr.models.explicit_repair import ExplicitRepairBatch
from .contracts import RouterOutput


class _PointResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2*dilation,
                                   dilation=dilation, groups=width)
        self.channel = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.GELU(),
                                     nn.Conv1d(2*width, width, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.channel(F.gelu(self.depthwise(self.norm(x))))


class PointResidualRefiner(nn.Module):
    """Share one point network across variables; modify true missing points only.

    Both scopes encode all 96 history points. Scope affects only the final
    numerical correction, keeping parameter count and point-network work equal.
    """
    def __init__(self, width: int, window_length: int, patch_length: int,
                 stride: int, span: int) -> None:
        super().__init__()
        if width != 64 or window_length != 96 or patch_length != 16 or stride != 8:
            raise ValueError('point refiner requires width64, 96-point P4 geometry')
        if type(span) is not int or span not in (32, 96):
            raise ValueError('point refiner span must be 32 or 96')
        self.width, self.window_length = width, window_length
        self.patch_length, self.stride, self.span = patch_length, stride, span
        self.latent_norm = nn.LayerNorm(width)
        self.input = nn.Sequential(nn.Conv1d(width+8, width, 1), nn.GELU())
        self.blocks = nn.Sequential(_PointResidualBlock(width, 1),
                                    _PointResidualBlock(width, 3))
        self.output = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _points(self, patches: Tensor, validity: Tensor, coverage: Tensor) -> Tensor:
        """Average patch features over valid original-time copies only."""
        B, C, N, D = patches.shape
        P = self.patch_length
        values = patches.permute(0, 1, 3, 2)[..., None].expand(B, C, D, N, P)
        values = values * validity[:, :, None].to(values.dtype)
        columns = values.permute(0, 1, 2, 4, 3).reshape(B*C, D*P, N)
        size = self.window_length + self.stride
        folded = F.fold(columns, output_size=(1, size), kernel_size=(1, P),
                        stride=(1, self.stride)).squeeze(2)[..., :self.window_length]
        counts = coverage.permute(0, 2, 1).reshape(B*C, 1, self.window_length)
        return folded / counts.clamp_min(1).to(folded.dtype)

    def forward(self, explicit: ExplicitRepairBatch, router: RouterOutput,
                query_mean: Tensor, query_std: Tensor) -> ExplicitRepairBatch:
        B, T, C = explicit.normalized_sequence.shape
        if T != self.window_length or router.latents.shape[:2] != (B, C):
            raise ValueError('point refiner sequence/router geometry mismatch')
        latent = self._points(self.latent_norm(router.latents), explicit.validity_patches,
                              explicit.coverage)
        weights = self._points(router.weights, explicit.validity_patches, explicit.coverage)
        sequence = explicit.normalized_sequence.permute(0, 2, 1).reshape(B*C, 1, T)
        observed = explicit.observed_sequence.permute(0, 2, 1).reshape(B*C, 1, T)
        mask = explicit.sequence_observed.permute(0, 2, 1).reshape(B*C, 1, T).to(sequence.dtype)
        position = torch.linspace(-1, 0, T, device=sequence.device,
                                  dtype=sequence.dtype)[None, None].expand(B*C, 1, T)
        correction = self.output(self.blocks(self.input(torch.cat(
            (latent, weights, sequence, observed, mask, position), dim=1))))
        correction = correction.reshape(B, C, T).permute(0, 2, 1)
        scope = torch.arange(T, device=sequence.device)[None, :, None] >= T-self.span
        eligible = explicit.sequence_missing & scope
        repaired = torch.where(eligible, explicit.normalized_sequence+correction,
                                explicit.normalized_sequence)
        patches = patchify_preserving_values(repaired, explicit.sequence_observed,
                    self.patch_length, self.stride, validity=explicit.sequence_validity)
        return replace(explicit, normalized_sequence=repaired,
                       standardized_sequence=repaired*query_std[:, None]+query_mean[:, None],
                       repaired_patches=patches.values.masked_fill(~explicit.validity_patches, 0))


__all__ = ['PointResidualRefiner']
