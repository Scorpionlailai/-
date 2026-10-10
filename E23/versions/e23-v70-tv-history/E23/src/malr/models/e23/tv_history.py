"""Map the observed-window TV state into the published history key space."""
import torch
from torch import nn
from torch.nn import functional as F


def fully_hidden_with_context(inputs):
    observed = inputs.patch_mask.bool() & inputs.patch_validity.bool()
    valid = inputs.patch_validity.bool().any(-1)
    context = observed.flatten(1).any(-1)[:, None, None]
    return inputs.damaged.bool() & valid & ~observed.any(-1) & context


class TVHistoricalQuery(nn.Module):
    def __init__(self, width, retrieval_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.project = nn.Linear(width + 3, retrieval_dim, bias=False)

    def forward(self, inputs):
        observed = inputs.patch_mask.bool() & inputs.patch_validity.bool()
        valid = inputs.patch_validity.bool()
        fraction = observed.sum((-1, -2)) / valid.sum((-1, -2)).clamp_min(1)
        shape = inputs.base_latents.shape[:-1]
        mean = inputs.query_mean[..., None].expand(shape)
        log_std = inputs.query_std.log()[..., None].expand(shape)
        fraction = fraction[..., None].expand(shape)
        state = torch.cat((self.norm(inputs.base_latents),
                           torch.stack((mean, log_std, fraction), -1)), -1)
        return F.normalize(self.project(state), dim=-1, eps=1e-8)
