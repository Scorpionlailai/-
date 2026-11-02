"""Independent H/V/T parameters evaluated together along an explicit group axis."""
from dataclasses import dataclass, replace
from math import sqrt
from typing import Mapping

import torch
from torch import Tensor
from .contracts import ExpertInputs
from .group_ops import grouped_linear, grouped_layer_norm, grouped_mlp
from .router import EXPERTS


@dataclass(frozen=True)
class GroupedExpertEvidence:
    message: Tensor
    quality: Tensor
    available: Tensor
    provenance: Mapping[str, Tensor]


@dataclass(frozen=True)
class GroupedRoundOutput:
    latents: Tensor
    weights: Tensor
    deltas: Mapping[str, Tensor]
    preview_gates: Tensor
    available: Tensor
    evidence: Mapping[str, GroupedExpertEvidence]


def _observed_inputs(inputs):
    observed = inputs.patch_mask.bool() & inputs.patch_validity.bool()
    return replace(inputs, patch_values=inputs.patch_values.masked_fill(~observed,0))


def prepare_grouped_variable_sources(inputs, experts, *, epoch, explore):
    inputs = _observed_inputs(inputs)
    return tuple(expert.prepare_sources(inputs,epoch=epoch,explore=explore) for expert in experts)


def _weights(logits, valid):
    weights = torch.softmax(logits.masked_fill(~valid,-1e9),dim=-2) * valid
    return weights / weights.sum(-2,keepdim=True).clamp_min(1e-8)


def _finish(messages, raw_quality, weights, valid, encoders, provenance):
    available = valid.any(-2)
    message = (messages*weights[...,None]).sum(-3)
    quality = grouped_mlp((raw_quality*weights[...,None]).sum(-3),[e.quality_projector for e in encoders])
    return GroupedExpertEvidence(message.masked_fill(~available[...,None],0),
        quality.masked_fill(~available[...,None],0),available,
        {**provenance,'candidate_valid':valid.detach(),'weights':weights.detach(),'candidate_count':valid.sum(-2).detach()})


def grouped_historical(inputs, states, cache, encoders, embeddings):
    groups = len(encoders); width = states.shape[-1]; candidates = cache.normalized_raw.shape[-2]
    raw = cache.normalized_raw[...,None,:].expand(*cache.normalized_raw.shape[:-1],groups,encoders[0].patch_length)
    quality = cache.raw_quality[...,None,:].expand(*cache.raw_quality.shape[:-1],groups,5)
    valid = cache.valid[...,None].expand(*cache.valid.shape,groups)
    state = grouped_layer_norm(states,[e.context_norm for e in encoders])+embeddings
    query = grouped_linear(state,[e.context_query for e in encoders])
    key = grouped_linear(raw,[e.context_key for e in encoders])
    logits = cache.base_logits[...,None] + .5*torch.tanh((query[...,None,:,:]*key).sum(-1)/sqrt(width))
    weights = _weights(logits,valid)
    features = torch.cat((raw,states[...,None,:,:].expand(*raw.shape[:-1],width),quality),-1)
    messages = grouped_mlp(features,[e.message_projector for e in encoders])
    return _finish(messages,quality,weights,valid,encoders,{})


def grouped_variable(inputs, states, experts, caches, *, source_context_allowed, embeddings):
    if experts[0].topk == 0:
        return GroupedExpertEvidence(torch.zeros_like(states),torch.zeros_like(states),
            torch.zeros(states.shape[:-1],dtype=torch.bool,device=states.device),{})
    # Flatten each group's own K x offset axes before stacking G.
    def feature(name): return torch.stack([getattr(c,name).flatten(-3,-2) for c in caches],-2)
    def scalar(name): return torch.stack([getattr(c,name).flatten(-2) for c in caches],-1)
    raw, observed, quality = feature('visible_values'),feature('source_observed'),feature('raw_quality')
    valid, logits = scalar('valid'),scalar('logits')
    source_ix, patch_ix = scalar('source_ix'),scalar('patch_ix')
    b = torch.arange(states.shape[0],device=states.device)[:,None,None,None,None]
    g = torch.arange(len(experts),device=states.device)[None,None,None,None,:]
    normalized = grouped_layer_norm(states,[e.context_norm for e in experts])
    source = normalized[b,source_ix,patch_ix,g]
    query = grouped_linear(normalized+embeddings,[e.context_query for e in experts])
    key = grouped_linear(source,[e.context_key for e in experts])
    adjustment = .5*torch.tanh((query[...,None,:,:]*key).sum(-1)/sqrt(states.shape[-1]))
    if source_context_allowed is not None: adjustment = adjustment * source_context_allowed[...,None,None]
    weights = _weights(logits+adjustment,valid)
    width = states.shape[-1]
    features = torch.cat((raw,observed.to(raw.dtype),states[...,None,:,:].expand(*raw.shape[:-1],width),
        scalar('lag')[...,None],scalar('source_fraction')[...,None],scalar('consistency')[...,None]),-1)
    messages = grouped_mlp(features,[e.message_projector for e in experts])
    context = grouped_linear(source,[e.source_context_projection for e in experts])
    if source_context_allowed is not None: context = context * source_context_allowed[...,None,None,None]
    messages = messages + context
    return _finish(messages,quality,weights,valid,experts,{'source_variables':source_ix.detach(),'source_patches':patch_ix.detach()})


def grouped_temporal(inputs, states, experts, *, source_context_allowed, embeddings):
    b,c,n,groups,width = states.shape
    observed = inputs.patch_mask.bool() & inputs.patch_validity.bool()
    visible = inputs.patch_values.masked_fill(~observed,0)
    raw = torch.cat((visible,observed.to(visible.dtype)),-1)[...,None,:].expand(b,c,n,groups,2*visible.shape[-1])
    original = grouped_mlp(raw,[e.source_encoder for e in experts])
    source_states = original + grouped_linear(grouped_layer_norm(states,[e.source_context_norm for e in experts]),
                                              [e.source_context_projection for e in experts])
    query = grouped_linear(states+embeddings,[e.query_projection for e in experts])
    key = grouped_linear(source_states,[e.key_projection for e in experts])
    similarity = torch.einsum('bcngd,bcmgd->bcnmg',query,key)/sqrt(width)
    if source_context_allowed is not None:
        original_key = grouped_linear(original,[e.key_projection for e in experts])
        fallback = torch.einsum('bcngd,bcmgd->bcnmg',query,original_key)/sqrt(width)
        similarity = torch.where(source_context_allowed[...,None,None],similarity,fallback)
    positions = torch.arange(n,device=states.device)
    separation = positions[:,None]-positions[None,:]
    signed = torch.tensor((1,-1),device=states.device)[:,None,None]*separation
    boundaries = experts[0].boundaries
    lower = torch.tensor((0,*boundaries),device=states.device)
    upper = torch.tensor((*boundaries,n),device=states.device)
    band = ((signed[:,None]>lower[None,:,None,None]) & (signed[:,None]<=upper[None,:,None,None])).reshape(-1,n,n)
    slots = band.shape[0]
    local = torch.tensor((True,*(False for _ in boundaries))*2,device=states.device)
    eligible = observed.any(-1)[:,:,None,:] & inputs.damaged[...,None]
    allowed = (eligible[:,:,None] & band[None,None])[...,None].expand(b,c,slots,n,n,groups)
    local_rank = -signed[:,None].expand(-1,len(lower),-1,-1).reshape(slots,n,n).to(states.dtype)
    rank = torch.where(local[None,None,:,None,None,None],local_rank[None,None,:,:,:,None],similarity[:,:,None]).masked_fill(~allowed,-torch.inf)
    indices = rank.argmax(-2).permute(0,1,3,2,4)
    valid = allowed.any(-2).permute(0,1,3,2,4)
    bi = torch.arange(b,device=states.device)[:,None,None,None,None]
    ci = torch.arange(c,device=states.device)[None,:,None,None,None]
    gi = torch.arange(groups,device=states.device)[None,None,None,None,:]
    source = source_states[bi,ci,indices,gi]
    if source_context_allowed is not None:
        source = torch.where(source_context_allowed[...,None,None,None],source,original[bi,ci,indices,gi])
    source_values, source_mask = visible[bi,ci,indices], observed[bi,ci,indices]
    target_values, target_mask = visible[...,None,None,:],observed[...,None,None,:]
    common = source_mask & target_mask
    error = ((source_values-target_values).abs()*common).sum(-1)/common.sum(-1).clamp_min(1)
    consistency = torch.exp(-error)*common.any(-1)
    fraction = source_mask.float().mean(-1)
    distance = (indices-positions[None,None,:,None,None]).abs().to(states.dtype)/max(n-1,1)
    local_features = local.to(states.dtype)[None,None,None,:,None].expand_as(fraction)
    target_fraction = observed.float().mean(-1)[...,None,None].expand_as(fraction)
    features = torch.cat((states[...,None,:,:].expand_as(source),source,fraction[...,None],distance[...,None],consistency[...,None],local_features[...,None]),-1)
    messages = grouped_mlp(features,[e.message_projector for e in experts])
    selected_score = similarity.transpose(-1,-2).gather(-1,indices.transpose(-1,-2)).transpose(-1,-2)
    weights = _weights(selected_score+.5*fraction-.05*distance,valid)
    quality = torch.stack((fraction,distance,consistency,target_fraction,local_features),-1)
    return _finish(messages,quality,weights,valid,experts,{'source_indices':indices.detach()})


def _delta(states,name,item,routers):
    normalized = grouped_layer_norm(states,[r.base_norm for r in routers])
    scales = torch.stack([r.scales[name] for r in routers])[...,None]
    return scales*grouped_mlp(torch.cat((normalized,item.message),-1),[r.residual[name] for r in routers])


def _geometry(inputs,states):
    valid = inputs.patch_validity.bool(); observed = inputs.patch_mask.bool() & valid
    denominator = valid.sum(-1).clamp_min(1)
    ratio = observed.sum(-1).to(states.dtype)/denominator
    missing = ((~observed)&valid).sum(-1).to(states.dtype)/denominator
    starts = inputs.patch_starts.to(states.dtype)
    if starts.ndim == 2: starts = starts[:,None,:].expand_as(ratio)
    relative = starts / starts.amax(-1,keepdim=True).clamp_min(1)
    return torch.stack((ratio,missing,relative),-1)[...,None,:].expand(*states.shape[:-1],3)


def _none(states,inputs,routers,embeddings):
    conditioned = grouped_layer_norm(states,[r.base_norm for r in routers])+embeddings
    return grouped_linear(torch.cat((conditioned,_geometry(inputs,states)),-1),[r.none_score for r in routers]).squeeze(-1)+torch.stack([r.none_bias for r in routers])


def _score(states,inputs,name,item,delta,routers,embeddings):
    conditioned = grouped_layer_norm(states,[r.base_norm for r in routers])+embeddings
    kinds = torch.stack([r.type_embedding.weight[EXPERTS.index(name)] for r in routers]).expand_as(states)
    return grouped_mlp(torch.cat((conditioned,delta,item.quality,kinds,_geometry(inputs,states)),-1),[r.shared_score for r in routers]).squeeze(-1)


def grouped_tv_h(inputs,groups,h_cache,v_caches,allowed,source_context_allowed):
    width = inputs.base_latents.shape[-1]
    if allowed.shape != (*inputs.damaged.shape,len(groups),3): raise ValueError('group availability shape mismatch')
    if not torch.isfinite(inputs.base_latents).all() or not torch.isfinite(inputs.rho).all(): raise ValueError('nonfinite group input')
    states = inputs.base_latents[...,None,:].expand(*inputs.base_latents.shape[:-1],len(groups),width)
    embeddings = torch.stack([g.condition_embedding.weight[0] for g in groups])
    routers = [g.router for g in groups]
    t = grouped_temporal(inputs,states,[g.temporal for g in groups],source_context_allowed=source_context_allowed,embeddings=embeddings)
    t = replace(t,available=t.available & allowed[...,2])
    dt = _delta(states,'temporal',t,routers)
    gt = torch.sigmoid(_score(states,inputs,'temporal',t,dt,routers,embeddings)-_none(states,inputs,routers,embeddings)) * t.available * inputs.damaged[...,None]
    after_t = states + gt[...,None]*dt
    v = grouped_variable(inputs,after_t,[g.variable for g in groups],v_caches,source_context_allowed=source_context_allowed,embeddings=embeddings)
    v = replace(v,available=v.available & allowed[...,1])
    dv = _delta(after_t,'variable',v,routers)
    gv = torch.sigmoid(_score(after_t,inputs,'variable',v,dv,routers,embeddings)-_none(after_t,inputs,routers,embeddings)) * v.available * inputs.damaged[...,None]
    h = grouped_historical(inputs,after_t+gv[...,None]*dv,h_cache,[g.historical for g in groups],embeddings)
    h = replace(h,available=h.available & allowed[...,0])
    dh = _delta(after_t+gv[...,None]*dv,'historical',h,routers)
    evidence = dict(zip(EXPERTS,(h,v,t))); deltas = dict(zip(EXPERTS,(dh,dv,dt)))
    scores = torch.stack([_score(states,inputs,name,evidence[name],deltas[name],routers,embeddings) for name in EXPERTS],-1)
    available = torch.stack((torch.ones_like(h.available),*(evidence[name].available & inputs.damaged[...,None] for name in EXPERTS)),-1)
    logits = torch.cat((_none(states,inputs,routers,embeddings)[...,None],scores.masked_fill(~available[...,1:],-torch.inf)),-1)
    if not torch.isfinite(scores).all() or not torch.isfinite(logits[...,0]).all(): raise ValueError('nonfinite group score')
    weights = torch.softmax(logits,-1)
    latents = states
    for index,name in enumerate(EXPERTS,1): latents = latents + weights[...,index,None]*deltas[name]
    return GroupedRoundOutput(latents,weights,deltas,torch.stack((gt,gv),-1),available,evidence)
