"""Train H keys on matching raw coordinates while preserving clean diversity."""
import torch
from torch.nn import functional as F
from malr.data.patching import patchify_with_state,patchify_preserving_values
from malr.prototypes.retrieval import masked_pair_loss,masked_contrastive_loss


def clean_key_regularization(keys):
    # [B,C,N,D]; real clean full patches only. Enforce content variance per variable.
    B,C,N,D=keys.shape
    x=keys.permute(1,0,2,3).reshape(C,B*N,D)
    centered=x-x.mean(1,keepdim=True)
    std=torch.sqrt(centered.square().mean(1)+1e-6)
    variance=(F.relu(.1-std)/.1).square().mean()
    covariance=centered.transpose(-1,-2)@centered/max(B*N-1,1)
    offdiag=covariance-torch.diag_embed(torch.diagonal(covariance,dim1=-2,dim2=-1))
    return variance,offdiag.square().sum((-1,-2)).div(D).mean()


def global_diverse_key_loss(model,batch):
    P,S=model.patch_length,model.stride
    # Training targets use only the clean lookback; no forecast_target is read.
    raw=batch.normalized_target*batch.query_std[:,None]+batch.query_mean[:,None]
    observed=torch.where(batch.observed_mask,batch.normalized_observed*batch.query_std[:,None]+batch.query_mean[:,None],0)
    query=patchify_with_state(observed,batch.observed_mask,P,S)
    clean=patchify_preserving_values(raw,torch.ones_like(batch.observed_mask),P,S)
    valid=clean.validity.all(-1)
    eligible=query.rho.lt(1)&query.effective_mask.any(-1)&valid
    encoder=model.historical.key_encoder
    if not eligible.any():return encoder.network[0].weight.sum()*0
    q=encoder(query.values[eligible],query.effective_mask[eligible])
    k=encoder(clean.values[eligible],clean.validity[eligible])
    n=len(q);alignment=masked_pair_loss(q,k)+.5*masked_contrastive_loss(q,k,temperature=.1)/torch.log(torch.tensor(float(max(n,2)),device=q.device))
    # All clean full patches provide the anti-collapse gradient. Padded tail excluded.
    B,C,N,_=clean.values.shape
    full=encoder(clean.values[valid],clean.validity[valid]).reshape(B,C,N-1,encoder.retrieval_dim)
    variance,covariance=clean_key_regularization(full)
    return alignment+.5*variance+.05*covariance
