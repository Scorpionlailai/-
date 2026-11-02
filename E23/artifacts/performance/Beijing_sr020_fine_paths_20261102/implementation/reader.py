"""One missing-position-conditioned normalization across H/V/T and zero-valued null."""
from math import sqrt
import torch
from torch import nn

class UnifiedReader(nn.Module):
    def __init__(self):
        super().__init__();self.width=64;self.heads=8
        self.context_norm=nn.LayerNorm(64);self.latent_norm=nn.LayerNorm(64)
        self.query=nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Linear(64,64))
        self.content_norm=nn.LayerNorm(64)
        self.key=nn.Linear(64,64);self.value=nn.Linear(64,64,bias=False)
        self.quality=nn.ModuleDict({n:nn.Linear(q,8,bias=False) for n,q in [('historical',5),('variable',4),('temporal',5)]})
        self.type_bias=nn.Parameter(torch.zeros(3,8))
        self.output=nn.Linear(64,1,bias=False);nn.init.zeros_(self.output.weight)

    def forward(self,base,context,bundles,eligible):
        B,C,N,P,D=context.shape
        phase=torch.linspace(-1,1,P,device=base.device,dtype=base.dtype).expand(B,C,N,P)[...,None]
        anchor=self.latent_norm(base)[...,None,:].expand(B,C,N,P,D)
        query=self.query(torch.cat((self.context_norm(context),anchor,phase),-1))
        content=[];bias=[];valid=[];counts=[]
        for i,name in enumerate(('historical','variable','temporal')):
            item=bundles[name];a=item.valid
            raw=torch.where(a[...,None],item.content,0)
            r=torch.where(a[...,None],item.quality,0)
            content.append(self.content_norm(raw));valid.append(a)
            bias.append(self.quality[name](r)+self.type_bias[i]);counts.append(a.shape[-1])
        content=torch.cat(content,-2);allowed=torch.cat(valid,-1);K=content.shape[-2]
        score_bias=torch.cat(bias,-2).transpose(-1,-2)[...,None,:]
        q=query.reshape(B,C,N,P,8,8).transpose(-3,-2)
        k=self.key(content).reshape(B,C,N,K,8,8).transpose(-3,-2)
        v=self.value(content).reshape(B,C,N,K,8,8).transpose(-3,-2)
        score=torch.matmul(q,k.transpose(-1,-2))/sqrt(8)+score_bias
        score=score.masked_fill(~allowed[...,None,None,:],-torch.inf)
        null=torch.zeros_like(score[...,:1])
        alpha=torch.softmax(torch.cat((null,score),-1),-1)
        readout=torch.matmul(alpha[...,1:],v).transpose(-3,-2).reshape(B,C,N,P,64)
        correction=self.output(readout).squeeze(-1)
        correction=torch.where(eligible,correction,0)
        return correction,{'weights':alpha,'readout':readout,'query':query,'valid':allowed,'candidate_counts':tuple(counts)}
