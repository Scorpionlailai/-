"""Fine temporal context inside T, computed from original observations only."""
import torch
from torch import nn
from malr.models.explicit_repair import _aggregate_allow_empty


class PointTemporalBlock(nn.Module):
    def __init__(self,channels=32):
        super().__init__()
        self.norm=nn.LayerNorm(channels)
        self.convolutions=nn.ModuleList(nn.Conv1d(channels,channels,k,padding=k//2,
            groups=channels) for k in (3,9,25,49))
        self.mix=nn.Sequential(nn.Conv1d(4*channels,channels,1),nn.GELU())

    def forward(self,x):
        normalized=self.norm(x.transpose(1,2)).transpose(1,2)
        return x+self.mix(torch.cat([c(normalized) for c in self.convolutions],1))


class PointTemporalContext(nn.Module):
    def __init__(self,d_model,channels=32):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(2,channels,1),nn.GELU())
        self.blocks=nn.Sequential(PointTemporalBlock(channels),PointTemporalBlock(channels))
        self.projection=nn.Linear(channels,d_model)
        nn.init.zeros_(self.projection.weight); nn.init.zeros_(self.projection.bias)

    def forward(self,inputs):
        b,c,n,p=inputs.patch_values.shape; stride=p//2
        window_len=(n-1)*stride+p-stride
        original=inputs.patch_mask.to(torch.bool)&inputs.patch_validity.to(torch.bool)
        sequence,coverage=_aggregate_allow_empty(inputs.patch_values,original,
            window_len=window_len,patch_len=p,stride=stride)
        mask=coverage.gt(0)
        features=torch.stack((sequence.permute(0,2,1),mask.permute(0,2,1).to(sequence.dtype)),2)
        encoded=self.blocks(self.stem(features.reshape(b*c,2,window_len)))
        points=torch.arange(n,device=sequence.device)[:,None]*stride+torch.arange(p,device=sequence.device)[None]
        valid=points<window_len; indices=points.clamp_max(window_len-1)
        patches=encoded.transpose(1,2)[:,indices]
        pooled=(patches*valid[None,:,:,None]).sum(2)/valid.sum(-1)[None,:,None].clamp_min(1)
        return self.projection(pooled).reshape(b,c,n,-1)
