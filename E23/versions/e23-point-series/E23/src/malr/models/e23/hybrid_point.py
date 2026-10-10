"""Dense history residuals with shared temporal CNN and channel-wise variable attention."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from malr.data.patching import patchify_preserving_values


class MultiScaleProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.large = nn.Conv1d(width, width, 31, padding=15, groups=width)
        self.small = nn.Conv1d(width, width, 5, padding=2, groups=width, bias=False)

    def forward(self, x):
        return self.large(x) + self.small(x)


class DenseRecoveryBlock(nn.Module):
    """CNN control and attention variant have identical parameters and initialization."""
    def __init__(self, width, attention):
        super().__init__()
        self.attention = attention
        self.query = MultiScaleProjection(width)
        self.key = MultiScaleProjection(width)
        self.value = MultiScaleProjection(width)
        self.projection = nn.Conv1d(width, width, 1)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.GELU(),
                                 nn.Conv1d(2*width, width, 1))
        self.scale1 = nn.Parameter(torch.full((1, width, 1), .001))
        self.scale2 = nn.Parameter(torch.full((1, width, 1), .001))

    def forward(self, x, observed):
        b, c, d, t = x.shape
        flat = x.reshape(b*c, d, t)
        q, k, v = (module(flat) for module in (self.query, self.key, self.value))
        if self.attention:
            q, k, v = (a.reshape(b, c, d, t).transpose(1, 2) for a in (q, k, v))
            reliable = observed.any(-1)
            bias = observed.float().mean(-1).clamp_min(.01).log()[:, None, None, :]
            bias = bias.masked_fill(~reliable[:, None, None, :], -1e9)
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                dropout_p=.1 if self.training else 0., scale=t**-.5)
            context = context * reliable.any(-1)[:, None, None, None]
            context = context.transpose(1, 2).reshape(b*c, d, t)
        else:
            context = (q+k+v)/3
        update = self.norm1(self.projection(context).transpose(1, 2)).transpose(1, 2)
        flat = flat + self.scale1*update
        update = self.norm2(self.ffn(flat).transpose(1, 2)).transpose(1, 2)
        return (flat+self.scale2*update).reshape(b, c, d, t)


class HybridPointRefiner(nn.Module):
    def __init__(self, width, window_length, patch_length, stride, span, *, attention):
        super().__init__()
        if (width, window_length, span) != (64, 96, 32) or patch_length not in (4, 8, 16) or stride != patch_length:
            raise ValueError('hybrid recovery requires nonoverlap4/8/16, width64, tail32')
        self.width, self.window_length = width, window_length
        self.patch_length, self.stride, self.span = patch_length, stride, span
        self.latent_norm = nn.LayerNorm(width)
        self.input = nn.Sequential(nn.Conv1d(width+8, width, 1), nn.GELU())
        self.blocks = nn.ModuleList([DenseRecoveryBlock(width, attention) for _ in range(2)])
        self.output = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, explicit, router, query_mean, query_std, *, specialist_selected=None):
        b, t, c = explicit.normalized_sequence.shape
        p = self.patch_length
        if t != 96 or router.latents.shape != (b, c, 96//p, self.width):
            raise ValueError('hybrid point geometry mismatch')
        latent = self.latent_norm(router.latents).repeat_interleave(p, 2).permute(0, 1, 3, 2)
        weights = router.weights.repeat_interleave(p, 2).permute(0, 1, 3, 2)
        coarse = explicit.standardized_sequence.transpose(1, 2)[:, :, None]
        mask = explicit.sequence_observed.transpose(1, 2)
        visible = (explicit.observed_sequence*query_std[:, None]+query_mean[:, None]).transpose(1, 2)
        visible = visible.masked_fill(~mask, 0)[:, :, None]
        position = torch.linspace(-1, 0, t, device=coarse.device, dtype=coarse.dtype)[None,None,None].expand(b,c,1,t)
        features = torch.cat((latent, weights, coarse, visible, mask[:, :, None].to(coarse.dtype), position), 2)
        features = self.input(features.reshape(b*c, self.width+8, t)).reshape(b,c,self.width,t)
        for block in self.blocks:
            features = block(features, mask)
        correction = self.output(features.reshape(b*c,self.width,t)).reshape(b,c,t).transpose(1,2)
        eligible = explicit.sequence_missing & (torch.arange(t,device=coarse.device)[None,:,None] >= t-self.span)
        if specialist_selected is not None:
            singleton = specialist_selected.repeat_interleave(p, 2).transpose(1,2)
            eligible = eligible & ~singleton
        # Keep original visible coordinates exact; convert only a global-scale residual.
        repaired = torch.where(eligible, explicit.normalized_sequence+correction/query_std[:,None],
                               explicit.normalized_sequence)
        patches = patchify_preserving_values(repaired, explicit.sequence_observed,p,self.stride,
                                             validity=explicit.sequence_validity)
        return replace(explicit, normalized_sequence=repaired,
                       standardized_sequence=repaired*query_std[:,None]+query_mean[:,None],
                       repaired_patches=patches.values.masked_fill(~explicit.validity_patches,0))
