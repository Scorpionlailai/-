"""Training-only position weights and repair retention in global data units."""
from __future__ import annotations
import torch
from torch import Tensor


def gains_to_weights(gains: Tensor, hidden: Tensor) -> Tensor:
    if hidden.ndim != 3 or hidden.shape[1:] != (96,7) or hidden.dtype != torch.bool:
        raise ValueError('hidden must be boolean [B,96,7]')
    if gains.ndim != 2 or gains.shape[1] != 42 or gains.shape[0] > hidden.shape[0]:
        raise ValueError('gains must be [subset,42]')
    if not torch.isfinite(gains).all():
        raise ValueError('nonfinite region gains')
    with torch.no_grad():
        weights=torch.ones(hidden.shape,device=hidden.device,dtype=gains.dtype)
        size=len(gains)
        positive=gains.detach().to(hidden.device).clamp_min(0).reshape(size,6,7).repeat_interleave(16,dim=1)
        selected=hidden[:size]
        count=selected.sum((1,2),keepdim=True).clamp_min(1)
        scale=(positive*selected).sum((1,2),keepdim=True)/count
        raw=1+(positive/scale.clamp_min(1e-6)).clamp(0,3)
        mean=(raw*selected).sum((1,2),keepdim=True)/count
        normalized=raw/mean.clamp_min(1)
        weights[:size]=torch.where(selected,normalized,torch.ones_like(normalized))
    return weights


def weighted_repair_loss(repaired: Tensor,target: Tensor,hidden: Tensor,
                         query_std: Tensor,weights: Tensor) -> Tensor:
    if repaired.shape!=target.shape or repaired.shape!=hidden.shape or repaired.shape!=weights.shape:
        raise ValueError('repair supervision shapes must match')
    if repaired.ndim!=3 or query_std.shape!=(repaired.shape[0],repaired.shape[2]):
        raise ValueError('query_std must be [B,C]')
    if hidden.dtype!=torch.bool or not torch.isfinite(query_std).all() or (query_std<=0).any():
        raise ValueError('boolean hidden and positive finite scales required')
    if not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('positive finite weights required')
    error=(repaired-target)*query_std.detach()[:,None,:]
    if not hidden.any():
        return repaired.sum()*0
    return (error.square()*weights.detach())[hidden].mean()


@torch.no_grad()
def region_oracle_gains(teacher,batch,*,subset: int=16) -> Tensor:
    """Restore train history regions under a fixed teacher; never update its state."""
    from .train import _starts
    from malr.data.patching import patchify_preserving_values
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('oracle teacher must be frozen and in evaluation mode')
    if type(subset)!=int or subset<1:
        raise ValueError('subset must be positive integer')
    size=min(subset,len(batch.normalized_observed))
    values=batch.normalized_observed[:size];mask=batch.observed_mask[:size]
    mean=batch.query_mean[:size];std=batch.query_std[:size]
    starts=_starts(batch,stride=teacher.repair.stride,patches=teacher.repair.patch_count,device=values.device)[:size]
    output=teacher.repair(values,mask,mean,std,starts)
    explicit=output.explicit
    repaired=explicit.repaired_patches
    clean=patchify_preserving_values(batch.normalized_target[:size],mask,16,8)
    clean_values=clean.values.masked_fill(~clean.validity,0)
    hidden=~explicit.observation_patches & explicit.validity_patches
    target=batch.forecast_target[:size]
    base=teacher.predictor(repaired,explicit.observation_patches,mean,std,patch_validity=explicit.validity_patches)
    base_mse=(base-target).square().mean((1,2))
    times=torch.arange(12,device=values.device)[:,None]*8+torch.arange(16,device=values.device)[None,:]
    gains=[]
    for band in range(6):
        selectors=torch.zeros(7,7,12,16,dtype=torch.bool,device=values.device)
        for variable in range(7):
            selectors[variable,variable]=(times>=band*16)&(times<(band+1)*16)
        selected=selectors[:,None]&hidden[None]
        variants=torch.where(selected,clean_values[None],repaired[None])
        pred=teacher.predictor(variants.reshape(7*size,7,12,16),explicit.observation_patches.repeat(7,1,1,1),mean.repeat(7,1),std.repeat(7,1),patch_validity=explicit.validity_patches.repeat(7,1,1,1)).reshape(7,size,7,96)
        gain=(base_mse[None]-(pred-target[None]).square().mean((2,3))).T
        active=(~mask[:,band*16:(band+1)*16]).any(1)
        gains.append(gain.masked_fill(~active,0))
    result=torch.cat(gains,dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite oracle region gains')
    return result


def snapshot_teacher(student):
    from copy import deepcopy
    return deepcopy(student).eval().requires_grad_(False)


def retention_identity(mode: str) -> dict:
    if mode not in ('none','uniform_v1','forecast_gain_v1'):
        raise ValueError('unsupported repair retention mode')
    if mode=='none':
        return {}
    identity=dict(repair_retention=mode,repair_retention_warmup=10,repair_retention_weight=.05)
    if mode=='forecast_gain_v1':
        identity.update(repair_retention_teacher='previous_epoch_final',repair_retention_subset=16,
                        repair_retention_regions=42,repair_retention_weights='positive_gain_floor1_cap4_mean1_v1')
    return identity


def retention_active(mode: str,epoch: int) -> bool:
    retention_identity(mode)
    return mode!='none' and epoch>10


def parameter_hash(module):
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(f'{name}:{array.dtype}:{array.shape}\n'.encode());digest.update(array.tobytes())
    return digest.hexdigest()


class RegionTrainingDiagnostics:
    """Aggregate signed train-only teacher gains and paired regional repair error."""
    def __init__(self):
        self.gain_sum=torch.zeros(42,dtype=torch.float64)
        self.squared_sum=torch.zeros(42,dtype=torch.float64)
        self.hidden_count=torch.zeros(42,dtype=torch.int64)
        self.scored_windows=0
        self.coverage=[]

    @torch.no_grad()
    def update(self,gains,repaired,batch):
        size=len(gains)
        self.gain_sum+=gains.detach().double().sum(0).cpu()
        error=(repaired[:size]-batch.normalized_target[:size])*batch.query_std[:size,None,:]
        for band in range(6):
            for c in range(7):
                hidden=~batch.observed_mask[:size,band*16:(band+1)*16,c]
                errors=error[:,band*16:(band+1)*16,c][hidden]
                i=band*7+c
                self.squared_sum[i]+=errors.square().double().sum().cpu()
                self.hidden_count[i]+=hidden.sum().cpu()
        self.scored_windows+=size
        self.coverage.append(dict(scored_windows=size,batch_windows=len(batch.normalized_observed)))

    def result(self):
        return dict(mean_signed_gain=(self.gain_sum/max(self.scored_windows,1)).tolist(),region_repair_mse=(self.squared_sum/self.hidden_count.clamp_min(1)).tolist(),region_hidden_coordinates=self.hidden_count.tolist(),scored_windows=self.scored_windows,scored_batches=len(self.coverage),per_batch_coverage=self.coverage,source='training only; teacher repair errors paired with signed intervention gains')
