"""Diagnostic prototype: full-negative loss with bounded row storage, no training integration."""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx,queries,keys,temperature,rows):
        keys=keys.detach();ctx.save_for_backward(queries,keys);ctx.temperature=temperature;ctx.rows=rows
        n=len(queries);total=queries.new_zeros(())
        for start in range(0,n,rows):
            logits=(queries[start:start+rows]@keys.T)/temperature
            labels=torch.arange(start,min(start+rows,n),device=queries.device)
            total=total+F.cross_entropy(logits,labels,reduction='sum')
        return total/n
    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        queries,keys=ctx.saved_tensors;n=len(queries);gradient=torch.empty_like(queries)
        for start in range(0,n,ctx.rows):
            stop=min(start+ctx.rows,n)
            probabilities=((queries[start:stop]@keys.T)/ctx.temperature).softmax(-1)
            probabilities[torch.arange(stop-start,device=queries.device),torch.arange(start,stop,device=queries.device)]-=1
            gradient[start:stop]=(probabilities@keys)*(grad_output/(n*ctx.temperature))
        return gradient,None,None,None

def full_contrastive_rows(queries,keys,*,temperature=1.,rows=512):
    if queries.ndim!=2 or queries.shape!=keys.shape:raise ValueError('query/key geometry mismatch')
    if temperature<=0 or rows<1:raise ValueError('positive temperature and row count required')
    if not len(queries):return queries.sum()*0
    return _Rows.apply(queries,keys,temperature,rows)
