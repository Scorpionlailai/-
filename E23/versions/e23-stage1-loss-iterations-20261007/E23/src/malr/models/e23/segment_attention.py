"""Read full lookback repair features and write global residuals to long gaps."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from malr.data.patching import patchify_preserving_values


def gap_geometry(hidden):
    """Window-clipped run length and [-1, 1] phase; invalid points break runs."""
    index = torch.arange(hidden.shape[-1], device=hidden.device)
    start = hidden & ~F.pad(hidden[..., :-1], (1, 0), value=False)
    end = hidden & ~F.pad(hidden[..., 1:], (0, 1), value=False)
    left = torch.where(start, index, 0).cummax(-1).values
    right = torch.where(end, index, hidden.shape[-1]-1).flip(-1).cummin(-1).values.flip(-1)
    length = torch.where(hidden, right-left+1, 0)
    phase = torch.where(hidden, 2*(index-left)/(length-1).clamp_min(1)-1, 0)
    return length, phase


class _GapAttentionBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=.1, bias=False, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(2*width, width), nn.Dropout(.1))

    def forward(self, x, valid):
        safe = torch.where(valid[..., None], x, 0)
        normalized = self.norm1(safe)
        normalized = torch.where(valid[..., None], normalized, 0)
        # Bias-free attention makes this always-valid null key/value exactly zero.
        source = torch.cat((normalized, torch.zeros_like(normalized[:, :1])), 1)
        source_valid = F.pad(valid, (0, 1), value=True)
        update, _ = self.attention(normalized, source, source,
                                   key_padding_mask=~source_valid, need_weights=False)
        x = safe + update
        x = x + self.ffn(self.norm2(x))
        return torch.where(valid[..., None], x, 0)


class SegmentAttentionReadout(nn.Module):
    def __init__(self, variables, width=64):
        super().__init__()
        if (variables, width) != (7, 64):
            raise ValueError('segment attention requires ETTh1 width64')
        self.variables, self.width = variables, width
        self.input = nn.Sequential(nn.Linear(143, width), nn.GELU(), nn.LayerNorm(width))
        self.position = nn.Parameter(torch.zeros(1, 96, width))
        self.blocks = nn.ModuleList([_GapAttentionBlock(width), _GapAttentionBlock(width)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _points(self, latents, validity, coverage):
        B, C, N, D = latents.shape
        raw = latents.permute(0, 1, 3, 2)[..., None].expand(B, C, D, N, 16)
        values = torch.where(validity[:, :, None], raw, 0)
        columns = values.permute(0, 1, 2, 4, 3).reshape(B*C, D*16, N)
        folded = F.fold(columns, (1, 104), (1, 16), stride=(1, 8)).squeeze(2)[..., :96]
        return folded/coverage.transpose(1, 2).reshape(B*C, 1, 96).clamp_min(1)

    def forward(self, explicit, router, points, visible_features, mean, std):
        sequence = explicit.normalized_sequence
        B, T, C = sequence.shape
        if (T, C) != (96, self.variables) or points.shape != (B*C, self.width, T) or visible_features.shape != (B*C, 5, T):
            raise ValueError('segment attention point geometry mismatch')
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError('segment attention needs positive finite original std')
        valid = explicit.sequence_validity
        observed = explicit.sequence_observed & valid
        hidden = explicit.sequence_missing & valid
        valid_rows = valid.transpose(1, 2).reshape(B*C, T)
        length, phase = gap_geometry(hidden.transpose(1, 2))
        scope = hidden & length.transpose(1, 2).ge(16)
        coarse = torch.where(valid, sequence, 0)*std[:, None]+mean[:, None]
        coarse = torch.where(valid, coarse, 0).transpose(1, 2).reshape(B*C, T, 1)
        visible = torch.where(observed, explicit.observed_sequence, 0)*std[:, None]+mean[:, None]
        visible = torch.where(observed, visible, 0).transpose(1, 2).reshape(B*C, T, 1)
        numeric = torch.cat((visible, visible_features[:, 1:].transpose(1, 2)), -1)
        numeric = torch.where(valid_rows[..., None], numeric, 0)
        point_context = torch.where(valid_rows[..., None], points.transpose(1, 2), 0)
        latent = self._points(router.latents, explicit.validity_patches, explicit.coverage).transpose(1, 2)
        identity = torch.eye(C, device=sequence.device, dtype=sequence.dtype)[None, :, None].expand(B, C, T, C).reshape(B*C, T, C)
        geometry = torch.stack((length.to(sequence.dtype)/96, phase.to(sequence.dtype)), -1).reshape(B*C, T, 2)
        features = torch.cat((point_context, latent, coarse, numeric, identity, geometry), -1)
        features = torch.where(valid_rows[..., None], features, 0)
        x = self.input(features) + self.position
        for block in self.blocks:
            x = block(x, valid_rows)
        delta = self.output(x).squeeze(-1).reshape(B, C, T).transpose(1, 2)
        # Global standardized residuals retain their units even for tiny local std.
        correction = torch.where(scope, delta, 0)/std[:, None]
        repaired = torch.where(scope, sequence+correction, sequence)
        patches = patchify_preserving_values(repaired, explicit.sequence_observed, 16, 8,
                                             validity=explicit.sequence_validity)
        return replace(explicit, normalized_sequence=repaired,
                       standardized_sequence=repaired*std[:, None]+mean[:, None],
                       repaired_patches=patches.values.masked_fill(~explicit.validity_patches, 0))
