"""Refine coarse imputation in global coordinates using visible history only."""
import torch
from torch import nn

DENSE_EXTRA_PARAMETERS = 80647


class GlobalDenseRefiner(nn.Module):
    def __init__(self, variables=7):
        super().__init__()
        if type(variables) is not int or variables < 2:
            raise ValueError("dense refinement requires at least two variables")
        self.variables = variables
        self.input = nn.Linear(5*variables, 64)
        self.input_norm = nn.LayerNorm(64)
        self.depthwise = nn.Conv1d(64, 64, 7, padding=3, groups=64)
        self.channel_mix = nn.Conv1d(64, 64, 1)
        self.position = nn.Parameter(torch.empty(1, 96, 64))
        nn.init.normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=.1,
            activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(64, variables)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.parameter_count = sum(p.numel() for p in self.parameters())
        assert self.parameter_count == 77952 + 385*variables

    def forward(self, normalized, explicit, mean, std):
        if normalized.shape[-1] != self.variables:
            raise ValueError("dense refinement variable count mismatch")
        valid = explicit.sequence_validity
        observed = explicit.sequence_observed & valid
        coarse = torch.where(valid, normalized * std[:, None] + mean[:, None], 0)
        safe = torch.where(observed, explicit.observed_sequence, 0)
        visible = torch.where(observed, safe * std[:, None] + mean[:, None], 0)
        # Coarse values are a fixed conditioning input. The final residual still
        # trains the underlying repair through the direct normalized skip path.
        index = torch.arange(normalized.shape[1], device=normalized.device)[None, :, None]
        left = torch.cummax(torch.where(observed, index, -1), dim=1).values
        right = torch.cummin(torch.where(observed, index, normalized.shape[1]).flip(1), dim=1).values.flip(1)
        left_value = visible.gather(1, left.clamp_min(0))
        right_value = visible.gather(1, right.clamp_max(normalized.shape[1]-1))
        span = (right-left).clamp_min(1)
        both = (left >= 0) & (right < normalized.shape[1]) & (right > left) & valid
        linear = left_value + (right_value-left_value)*(index-left)/span
        has_left, has_right = left >= 0, right < normalized.shape[1]
        one = (has_left ^ has_right) & valid & ~observed
        distance = torch.where(has_left, index-left, right-index).clamp_min(0)
        guide = torch.where(both, linear, torch.where(has_left, left_value, right_value))
        support = (both.to(coarse.dtype)*torch.exp(-(span-1).to(coarse.dtype)/16)
                   + one.to(coarse.dtype)*torch.exp(-distance.to(coarse.dtype)/16))
        suggested_delta = torch.where(both | one, guide-coarse.detach(), 0)*support
        features = torch.cat((coarse.detach(), visible, observed.to(coarse.dtype), suggested_delta, support), -1)
        x = self.input(features)
        local = self.channel_mix(torch.nn.functional.gelu(
            self.depthwise(self.input_norm(x).transpose(1, 2)))).transpose(1, 2)
        x = x + local + self.position
        token_valid = valid.any(-1)
        padding = ~token_valid
        # A sentinel avoids all-masked attention NaNs in fully invalid windows.
        safe_padding = padding.clone()
        safe_padding[:, 0] &= token_valid.any(-1)
        x = self.temporal(x, src_key_padding_mask=safe_padding)
        delta = 2 * torch.tanh(self.output(x) / 2)
        hidden = valid & ~observed
        has_source = observed.flatten(1).any(-1)[:, None, None]
        delta = torch.where(hidden & has_source, delta, 0)
        return normalized + delta / std[:, None]
