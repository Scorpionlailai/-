"""Frozen official T1 repair, followed by the unchanged E23 forecast interface."""

import torch

from malr.data.patching import patchify_preserving_values
from .t1_p4 import T1ForecastModel, t1_inputs


class FrozenT1ForecastModel(T1ForecastModel):
    def __init__(self, imputer, predictor):
        super().__init__(imputer, predictor)
        self.imputer.requires_grad_(False)
        self.imputer.eval()

    def train(self, mode=True):
        super().train(mode)
        # Freeze BatchNorm running statistics and all stochastic repair layers.
        self.imputer.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            values, mask = t1_inputs(batch)
            candidate = self.imputer(values, mask=mask).float()
            repaired = torch.where(batch.observed_mask, values, candidate)
            normalized = (repaired - batch.query_mean[:, None, :]) / batch.query_std[:, None, :]
            normalized = torch.where(batch.observed_mask, batch.normalized_observed, normalized)
            patches = patchify_preserving_values(normalized, batch.observed_mask, 16, 8)
            # Keep T1's existing endpoint replication, as requested.
            predictor_values = patches.values
        return self.predictor(predictor_values, patches.effective_mask,
                              batch.query_mean, batch.query_std,
                              patch_validity=patches.validity)
