"""Explicit construction and identity for a shared-prefix calibration model."""
from hashlib import sha256
import json
from math import isfinite
import torch

from malr.models.e23.calibration_ensemble import RecentMeanEnsembleProjector


KIND='shared_prefix_calibration_v1'


def ensemble_spec_sha256(spec):
    validate_ensemble_spec(spec)
    return sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_ensemble_spec(spec):
    if not isinstance(spec,dict) or set(spec)!={'kind','member_widths','weights','patch_len','stride','no_source_policy'}:
        raise ValueError('invalid calibration ensemble specification')
    if spec['kind']!=KIND or spec['patch_len']!=16 or spec['stride']!=8 or spec['no_source_policy']!='preserve_coarse':
        raise ValueError('unsupported calibration ensemble geometry or policy')
    widths,weights=spec['member_widths'],spec['weights']
    if not isinstance(widths,list) or not isinstance(weights,list) or len(widths)!=len(weights) or len(widths)<2:
        raise ValueError('invalid calibration ensemble members')
    if any(type(w) is not int or w not in (64,128,256) for w in widths):raise ValueError('invalid calibration width')
    if any(type(w) not in (int,float) or not isfinite(w) or w<=0 for w in weights) or abs(sum(weights)-1)>1e-8:
        raise ValueError('invalid calibration weights')


def build_calibration_repair(protocol,settings,spec,*,device):
    from .train import build_repair_model
    validate_ensemble_spec(spec)
    if settings.d_model!=64 or settings.repair_patch_length!=16 or settings.repair_stride!=8 or settings.recent_mean_projection!='global_dense_context_gate_v6':
        raise ValueError('ensemble requires the original contextual64 coarse repair')
    model=build_repair_model(protocol,settings,device=device)
    model.recent_mean_projector=RecentMeanEnsembleProjector(settings.variable_count,
        widths=spec['member_widths'],weights=spec['weights']).to(device)
    return model


def verify_ensemble_instance(model,spec):
    validate_ensemble_spec(spec)
    projector=model.recent_mean_projector
    if not isinstance(projector,RecentMeanEnsembleProjector) or len(projector.members)!=len(spec['weights']):
        raise ValueError('calibration ensemble instance mismatch')
    actual_widths=[member.global_dense_refiner.width for member in projector.members]
    expected_weights=torch.tensor(spec['weights'],dtype=torch.float64)
    if actual_widths!=spec['member_widths'] or not torch.equal(projector.weights.detach().cpu().double(),expected_weights):
        raise ValueError('calibration ensemble state differs from specification')
