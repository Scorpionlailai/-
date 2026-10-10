"""Keep repaired values, original visibility and observed-only RevON statistics."""
from dataclasses import replace

import torch
from torch import nn

from integrations.e23.train import _starts
from integrations.forecast_compare.coifnet_p4 import coifnet_inputs


def repaired_coifnet_output(core, repaired, visible_values, mask, timestamps):
    """Use the upstream layers without letting RevON erase supplied repairs."""
    if not core.revin or not core.m_cfg.is_RevON or not core.m_cfg.use_head:
        raise ValueError("this minimal adapter requires the locked author configuration")
    # RevON's sum assumes missing inputs are zero. Its usual 'norm' output
    # also masks away repairs, so compute statistics and normalize separately.
    core.revin_layer(visible_values * mask, "norm", mask)
    normalized = core.revin_layer(repaired, "norm-fore")
    hidden = core.share_module(normalized, mask, timestamps)
    output = core.aux_head(hidden.permute(0, 2, 1)).permute(0, 2, 1)
    return core.revin_layer(output, "denorm").contiguous()


class FrozenRepairCoIFNet(nn.Module):
    def __init__(self, repair, coifnet, memory, *, repair_batch_size=128):
        super().__init__()
        self.repair = repair.eval().requires_grad_(False)
        self.coifnet = coifnet
        self.memory = memory
        self.repair_batch_size = repair_batch_size

    def train(self, mode=True):
        super().train(mode)
        self.repair.eval()
        return self

    @torch.no_grad()
    def repaired_history(self, batch):
        pieces = []
        for start in range(0, len(batch.refs), self.repair_batch_size):
            end = start + self.repair_batch_size
            part = replace(batch, **{name: getattr(batch, name)[start:end]
                for name in ("normalized_observed", "normalized_target", "observed_mask",
                             "query_mean", "query_std", "forecast_target")},
                refs=batch.refs[start:end])
            starts = _starts(part, stride=self.repair.stride,
                             patches=self.repair.patch_count, device=part.query_mean.device)
            output = self.repair(part.normalized_observed, part.observed_mask,
                                 part.query_mean, part.query_std, starts)
            normalized = output.explicit.normalized_sequence
            if not torch.equal(normalized[part.observed_mask], part.normalized_observed[part.observed_mask]):
                raise RuntimeError("repair changed visible values")
            complete = normalized * part.query_std[:, None] + part.query_mean[:, None]
            visible = part.normalized_observed * part.query_std[:, None] + part.query_mean[:, None]
            pieces.append(torch.where(part.observed_mask, visible, complete))
        repaired = torch.cat(pieces)
        if not torch.isfinite(repaired).all():
            raise FloatingPointError("nonfinite repair input")
        return repaired

    def forward(self, batch, timestamp_table, *, repaired=None):
        if repaired is None:
            repaired = self.repaired_history(batch)
        visible, mask, timestamps = coifnet_inputs(batch, timestamp_table)
        output = repaired_coifnet_output(self.coifnet, repaired, visible, mask, timestamps)
        return output, repaired
