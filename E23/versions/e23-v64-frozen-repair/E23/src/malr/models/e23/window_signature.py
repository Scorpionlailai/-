"""Whole-window variable interaction before retrieval and recovery messages.

The variable-attention geometry follows the public T1 CNN/attention backbone:
feature channels are attention heads and entire temporal signatures are vectors.
Here signatures come exclusively from the visible-only E23 window CNN. Positive
bounded affine conditioning and an unavailable-source/null mask are added.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class _SignatureProjection(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.large=nn.Conv1d(width,width,31,padding=15,groups=width)
        self.small=nn.Conv1d(width,width,5,padding=2,groups=width,bias=False)

    def forward(self,x):return self.large(x)+self.small(x)


class _SignatureBlock(nn.Module):
    def __init__(self,width,variables,*,statistics_conditioning=True):
        super().__init__()
        self.width,self.variables=width,variables
        self.query=_SignatureProjection(width)
        self.key=_SignatureProjection(width)
        self.value=_SignatureProjection(width)
        self.statistics = None
        if statistics_conditioning:
            self.statistics=nn.Sequential(nn.Linear(2*variables,32),nn.GELU(),nn.Linear(32,6*variables*width))
            nn.init.zeros_(self.statistics[-1].weight);nn.init.zeros_(self.statistics[-1].bias)
        self.project=nn.Conv1d(width,width,1)
        self.norm1=nn.LayerNorm((width,96))
        self.norm2=nn.LayerNorm((width,96))
        self.ffn=nn.Sequential(nn.Conv1d(width,2*width,1),nn.GELU(),nn.Dropout(.1),nn.Conv1d(2*width,width,1))
        self.scale1=nn.Parameter(torch.full((1,width,1),.001))
        self.scale2=nn.Parameter(torch.full((1,width,1),.001))
        # Exactly preserve the old point features, including point router queries.
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.ffn[-1].weight);nn.init.zeros_(self.ffn[-1].bias)

    def forward(self,x,observed,valid,mean,std):
        B,C,D,T=x.shape;flat=x.reshape(B*C,D,T)
        q,k,v=[m(flat).reshape(B,C,D,T).transpose(1,2) for m in [self.query,self.key,self.value]]
        if self.statistics is not None:
            stats=torch.cat((mean,std.clamp_min(1e-6).log()),-1)
            raw=self.statistics(stats).reshape(B,6,C,D).transpose(-1,-2)[...,None]
            scales=.25+3.75*torch.sigmoid(raw[:,:3]-math.log(4))
            offsets=2*torch.tanh(raw[:,3:]/2)
            q,k,v=[z*scales[:,i]+offsets[:,i] for i,z in enumerate([q,k,v])]
        q,k,v=[z.masked_fill(~valid[:,None,:,:],0) for z in [q,k,v]]
        reliable=(observed & valid).any(-1)
        fraction=(observed & valid).sum(-1).to(x.dtype)/valid.sum(-1).clamp_min(1)
        logits=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(T)
        logits=logits+fraction.clamp_min(1e-8).log()[:,None,None,:]
        logits=logits.masked_fill(~reliable[:,None,None,:],-torch.inf)
        null=torch.zeros(B,D,C,1,device=x.device,dtype=x.dtype)
        attention=torch.softmax(torch.cat((null,logits),-1),-1)
        attention=F.dropout(attention,p=.1,training=self.training)
        context=torch.matmul(attention[...,1:],v).transpose(1,2).reshape(B*C,D,T)
        update=self.norm1(self.project(context))
        flat=flat+self.scale1*update
        flat=flat+self.scale2*self.norm2(self.ffn(flat))
        return flat.reshape(B,C,D,T).masked_fill(~valid[:,:,None,:],0)


class WindowSignatureContext(nn.Module):
    def __init__(self,width,variables,*,statistics_conditioning=True):
        super().__init__()
        if width!=64 or variables<2:raise ValueError("window signature requires width64 and multiple variables")
        self.width,self.variables=width,variables
        self.blocks=nn.ModuleList([_SignatureBlock(width,variables,statistics_conditioning=statistics_conditioning) for _ in range(2)])

    def forward(self,points,observed,valid,mean,std):
        B,T,C=observed.shape
        if (T,C)!=(96,self.variables):raise ValueError("window signature geometry mismatch")
        x=points.reshape(B,C,self.width,T)
        original=observed.transpose(1,2);valid=valid.reshape(B,C,T)
        for block in self.blocks:x=block(x,original,valid,mean,std)
        return x.reshape(B*C,self.width,T)
