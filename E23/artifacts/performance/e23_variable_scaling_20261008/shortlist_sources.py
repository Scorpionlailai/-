"""Experimental bounded local variable scoring, using a learned-prior shortlist.

For C-1 <= cap the original selection is called exactly. For larger C this
changes source selection and needs accuracy testing; it is not a checkpoint-
equivalent optimization. Keep this module outside sealed training sources.
"""
from math import sqrt
import torch


def shortlisted_sources(self, inputs, *, epoch, explore, cap=16, original):
    values, mask, validity = inputs.patch_values, inputs.patch_mask, inputs.patch_validity
    B, C, N, P = values.shape
    if cap < self.topk:
        raise ValueError('shortlist must hold at least all selected sources')
    if C-1 <= cap:
        return original(self, inputs, epoch=epoch, explore=explore)
    if self.selection != 'window_context_topk_v1':
        raise ValueError('shortlist prototype supports original contextual V policy only')
    prior = (self.source_embedding.weight @ self.target_embedding.weight.T / sqrt(self.d_model)).T
    targets = torch.arange(C, device=values.device)
    global_rank = torch.argsort(prior.masked_fill(torch.eye(C,device=values.device,dtype=torch.bool),-torch.inf),dim=-1,descending=True,stable=True)
    shortlist = global_rank[:, :cap].clone()
    if explore:
        remaining = global_rank[:, cap:C-1]
        rotation = (epoch+targets) % remaining.shape[-1]
        shortlist[:, -1] = remaining.gather(1,rotation[:,None]).squeeze(1)
    patch = torch.arange(N,device=values.device)
    offsets = torch.tensor(self.offsets,device=values.device)
    raw = patch[:,None]+offsets[None,:]
    in_range = (raw >= 0) & (raw < N)
    local = raw.clamp(0,N-1)
    local_values = values[:,:,local].permute(0,2,1,3,4)
    local_valid = validity[:,:,local].permute(0,2,1,3,4) & in_range[None,:,None,:,None]
    local_observed = (mask & validity)[:,:,local].permute(0,2,1,3,4) & local_valid
    visible_fraction = local_observed.sum((-1,-2)).float()/local_valid.sum((-1,-2)).clamp_min(1)
    aligned = self.offsets.index(0)
    source_observed = local_observed[...,aligned,:][:,:,shortlist]
    target_observed = (mask & validity).permute(0,2,1,3)
    common = target_observed[:,:,:,None,:] & source_observed
    source_values = local_values[...,aligned,:][:,:,shortlist]
    target_values = values.permute(0,2,1,3)[:,:,:,None,:]
    difference = (source_values.masked_fill(~common,0)-target_values.masked_fill(~common,0)).abs()
    error = difference.sum(-1)/common.sum(-1).clamp_min(1)
    consistency = torch.exp(-error)*common.any(-1)
    selected_prior = prior.gather(1,shortlist)
    scores = (selected_prior[None,None]+.5*visible_fraction[:,:,shortlist]+.5*consistency).permute(0,2,1,3)
    has_source = local_observed.any((-1,-2))[:,:,shortlist].permute(0,2,1,3)
    scores = scores.masked_fill(~has_source,-1e9)
    # Original stable sorting breaks ties by source index. Preserve that rule
    # within the shortlist rather than by learned-prior shortlist position.
    source_order = torch.argsort(shortlist,dim=-1,stable=True)
    scores_in_source_order = scores.gather(-1,source_order[None,:,None,:].expand(B,C,N,cap))
    local_rank = torch.argsort(scores_in_source_order,dim=-1,descending=True,stable=True)
    ranked = source_order[None,:,None,:].expand(B,C,N,cap).gather(-1,local_rank)
    chosen = ranked[...,:self.topk].clone()
    if explore and cap > self.topk:
        choices=ranked[...,self.topk:]
        rotation=((epoch+targets[None,:,None])%choices.shape[-1]).expand(B,C,N)
        chosen[...,-1]=choices.gather(-1,rotation[...,None]).squeeze(-1)
    sources=shortlist[None,:,None,:].expand(B,C,N,cap).gather(-1,chosen)
    priors=prior[None,:,None,:].expand(B,C,N,C).gather(-1,sources)
    return sources,priors,torch.ones_like(sources,dtype=torch.bool)
