"""Small variable-specific modulation in numeric repair readouts."""
import torch
from torch import nn

class VariableReadoutConditioner(nn.Module):
    def __init__(self,variables,width):
        super().__init__()
        if min(variables,width)<1:raise ValueError("invalid variable readout geometry")
        self.scale=nn.Parameter(torch.zeros(variables,width))
        self.bias=nn.Parameter(torch.zeros(variables,width))

    def forward(self,x):
        # x: [B,C,...,D], fixed dataset channel ordering.
        if x.shape[1]!=self.scale.shape[0] or x.shape[-1]!=self.scale.shape[1]:
            raise ValueError("variable readout shape mismatch")
        shape=(1,self.scale.shape[0])+ (1,)*(x.ndim-3)+(self.scale.shape[1],)
        gamma=.5*torch.tanh(self.scale).reshape(shape)
        beta=.5*torch.tanh(self.bias).reshape(shape)
        return x*(1+gamma)+beta
