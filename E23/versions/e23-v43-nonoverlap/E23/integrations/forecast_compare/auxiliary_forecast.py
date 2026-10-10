"""Training-only forecast supervision with an independently initialized head."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from malr.data.patching import patchify_preserving_values
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel


def auxiliary_identity(mode):
    if mode not in ('none', 'forecast_mse_v1'):
        raise ValueError('unsupported Stage1 auxiliary forecast mode')
    return {} if mode == 'none' else {
        'stage1_aux_forecast': mode,
        'stage1_aux_forecast_weight': .05,
        'stage1_aux_forecast_warmup': 10,
        'stage1_aux_predictor_seed': 2022,
        'stage1_aux_predictor_discarded': True,
    }


@contextmanager
def isolated_random_stream(device, seed):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def build_auxiliary_predictor(protocol, *, device):
    profile = protocol.config.model.for_dataset('ETTh1')
    with isolated_random_stream(device, 2022):
        core = PatchTSTCore(
            c_in=protocol.config.data.datasets['ETTh1'].n_vars,
            seq_len=protocol.config.data.seq_len, pred_len=protocol.config.data.pred_len,
            patch_len=protocol.config.patch.length, stride=protocol.config.patch.stride,
            d_model=profile.d_model, n_heads=profile.n_heads, d_ff=profile.d_ff,
            n_layers=profile.encoder_layers, dropout=profile.dropout,
            fc_dropout=profile.fc_dropout, head_dropout=profile.head_dropout,
            attn_dropout=profile.attn_dropout,
        )
        return MaskAwareForecastModel(core).to(device)


def predict_repaired(predictor, repaired_normalized, batch):
    if repaired_normalized.shape != batch.normalized_observed.shape:
        raise ValueError('auxiliary repaired sequence shape mismatch')
    values = torch.where(batch.observed_mask, batch.normalized_observed, repaired_normalized)
    patches = patchify_preserving_values(values, batch.observed_mask, length=16, stride=8)
    return predictor(patches.values, patches.effective_mask,
                     batch.query_mean, batch.query_std, patch_validity=patches.validity)


def auxiliary_forecast_loss(predictor, repaired_normalized, batch, *, epoch, batch_index=0):
    if epoch < 1:
        raise ValueError('auxiliary epoch must be positive')
    values = repaired_normalized.detach() if epoch <= 10 else repaired_normalized
    with isolated_random_stream(values.device, 2022 + epoch * 100000 + batch_index):
        prediction = predict_repaired(predictor, values, batch)
    if prediction.shape != batch.forecast_target.shape:
        raise ValueError('auxiliary forecast target shape mismatch')
    return F.mse_loss(prediction.float(), batch.forecast_target.float())


def verify_auxiliary_identity(record, payload):
    mode = record.get('stage1_aux_forecast', 'none')
    expected = auxiliary_identity(mode)
    if payload.get('stage1_aux_forecast', 'none') != mode or any(
        record.get(k) != v or payload.get(k) != v for k, v in expected.items()
    ):
        raise ValueError('Stage1 auxiliary forecast identity mismatch')


def t1_repaired_sequence(model, batch):
    from .t1_p4 import t1_inputs, t1_clean_history
    values, mask = t1_inputs(batch)
    candidate = model(values, mask=mask)
    repair_loss = (candidate - t1_clean_history(batch)).square()[~batch.observed_mask].mean()
    normalized = (candidate - batch.query_mean[:, None, :]) / batch.query_std[:, None, :]
    return repair_loss, torch.where(batch.observed_mask, batch.normalized_observed, normalized)
