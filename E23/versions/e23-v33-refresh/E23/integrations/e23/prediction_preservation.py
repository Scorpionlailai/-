"""Stage1-only clean forecasting teacher and differentiable input consistency."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from malr.data.patching import patchify_preserving_values
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel


def preservation_identity(mode: str) -> dict:
    if mode == 'none':
        return {}
    if mode != 'clean_forecast_v1':
        raise ValueError('unsupported Stage1 preservation mode')
    return dict(stage1_preservation=mode, stage1_preservation_weight=.1,
                stage1_preservation_warmup=20, stage1_preservation_probe_seed=2053,
                stage1_preservation_probe_lr=.001,
                stage1_preservation_target='frozen_clean_forecast_same_mask_stats')


def _rng_devices(device):
    return [torch.cuda.current_device() if device.index is None else device.index] if device.type == 'cuda' else []


def build_probe(protocol, *, device: torch.device, seed: int):
    profile = protocol.config.model.for_dataset('ETTh1')
    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.random.default_generator.manual_seed(seed)
        core = PatchTSTCore(c_in=7, seq_len=96, pred_len=96, patch_len=16, stride=8,
                            d_model=profile.d_model, n_heads=profile.n_heads,
                            d_ff=profile.d_ff, n_layers=profile.encoder_layers,
                            dropout=profile.dropout, fc_dropout=profile.fc_dropout,
                            head_dropout=profile.head_dropout, attn_dropout=profile.attn_dropout)
        return MaskAwareForecastModel(core).to(device)


def predict_sequence(probe, sequence, batch):
    patches = patchify_preserving_values(sequence, batch.observed_mask, 16, 8)
    return probe(patches.values.masked_fill(~patches.validity, 0), patches.effective_mask,
                 batch.query_mean.detach(), batch.query_std.detach(),
                 patch_validity=patches.validity)


def train_clean_probe_step(probe, optimizer, batch) -> float:
    device = next(probe.parameters()).device
    with torch.random.fork_rng(devices=_rng_devices(device)):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_sequence(probe, batch.normalized_target.detach(), batch)
        loss = F.mse_loss(prediction, batch.forecast_target.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite clean probe loss')
        loss.backward()
        optimizer.step()
    return loss.detach().item()


def freeze_probe(probe):
    probe.eval().requires_grad_(False)
    probe.zero_grad(set_to_none=True)


def consistency_loss(probe, repaired, batch):
    if probe.training or any(p.requires_grad for p in probe.parameters()):
        raise ValueError('consistency teacher must be frozen and eval')
    with torch.no_grad():
        target = predict_sequence(probe, batch.normalized_target.detach(), batch)
    predicted = predict_sequence(probe, repaired, batch)
    loss = F.mse_loss(predicted, target)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite preservation loss')
    return loss


def checked_preservation(record, payload=None):
    """Bind the training-only identity without requiring teacher files at inference."""
    mode = record.get('stage1_preservation', 'none')
    identity = preservation_identity(mode)
    if any(record.get(k) != v for k,v in identity.items()):
        raise ValueError('Stage1 preservation identity mismatch')
    if identity:
        value = record.get('stage1_preservation_probe_state_sha256')
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('Stage1 preservation probe identity missing')
        identity['stage1_preservation_probe_state_sha256'] = value
    if payload is not None:
        if payload.get('stage1_preservation','none') != mode or any(payload.get(k) != v for k,v in identity.items()):
            raise ValueError('Stage1 preservation checkpoint mismatch')
    return identity
