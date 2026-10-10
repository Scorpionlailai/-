"""Refine coarse imputation in global coordinates using visible history only."""
import torch
from torch import nn

DENSE_EXTRA_PARAMETERS = 79751


class GlobalDenseRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(21, 64)
        self.input_norm = nn.LayerNorm(64)
        self.depthwise = nn.Conv1d(64, 64, 7, padding=3, groups=64)
        self.channel_mix = nn.Conv1d(64, 64, 1)
        self.position = nn.Parameter(torch.empty(1, 96, 64))
        nn.init.normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=.1,
            activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(64, 7)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        assert sum(p.numel() for p in self.parameters()) == DENSE_EXTRA_PARAMETERS

    def forward(self, normalized, explicit, mean, std):
        valid = explicit.sequence_validity
        observed = explicit.sequence_observed & valid
        coarse = torch.where(valid, normalized * std[:, None] + mean[:, None], 0)
        safe = torch.where(observed, explicit.observed_sequence, 0)
        visible = torch.where(observed, safe * std[:, None] + mean[:, None], 0)
        # Coarse values are a fixed conditioning input. The final residual still
        # trains the underlying repair through the direct normalized skip path.
        features = torch.cat((coarse.detach(), visible, observed.to(coarse.dtype)), -1)
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
