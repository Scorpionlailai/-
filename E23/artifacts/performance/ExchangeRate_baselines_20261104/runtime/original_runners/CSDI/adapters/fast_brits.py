"""TorchScript recurrence with the original RITS equations and parameters."""
from typing import List,Tuple
import torch
from torch import nn,Tensor
from torch.nn import functional as F

class Decay(nn.Module):
    def __init__(self,old):
        super().__init__();self.W=old.W;self.b=old.b;self.diag=old.diag
        self.register_buffer('m',old.m if old.diag else torch.ones_like(old.W))
    def forward(self,d:Tensor)->Tensor:
        return torch.exp(-F.relu(F.linear(d,self.W*self.m if self.diag else self.W,self.b)))

class Feature(nn.Module):
    def __init__(self,old):
        super().__init__();self.W=old.W;self.b=old.b;self.register_buffer('m',old.m)
    def forward(self,x:Tensor)->Tensor:return F.linear(x,self.W*self.m,self.b)

class ScriptRITS(nn.Module):
    def __init__(self,old):
        super().__init__();self.hid=old.rnn_hid_size;self.impute_weight=old.impute_weight
        self.rnn_cell=old.rnn_cell;self.temp_decay_h=Decay(old.temp_decay_h);self.temp_decay_x=Decay(old.temp_decay_x)
        self.hist_reg=old.hist_reg;self.feat_reg=Feature(old.feat_reg);self.weight_combine=old.weight_combine
        self.dropout=old.dropout;self.out=old.out
    def forward(self,values:Tensor,masks:Tensor)->Tuple[Tensor,Tensor]:
        h=values.new_zeros((values.size(0),self.hid));c=values.new_zeros((values.size(0),self.hid))
        delta=torch.ones_like(values)
        for t in range(1,values.size(1)):delta[:,t]=1+(1-masks[:,t])*delta[:,t-1]
        loss=values.new_zeros(());imputations:List[Tensor]=[]
        for t in range(96):
            x=values[:,t];m=masks[:,t];d=delta[:,t]
            gamma_h=self.temp_decay_h(d);gamma_x=self.temp_decay_x(d);h=h*gamma_h
            x_h=self.hist_reg(h);loss=loss+torch.sum(torch.abs(x-x_h)*m)/(torch.sum(m)+1e-5)
            x_c=m*x+(1-m)*x_h;z_h=self.feat_reg(x_c)
            loss=loss+torch.sum(torch.abs(x-z_h)*m)/(torch.sum(m)+1e-5)
            alpha=self.weight_combine(torch.cat([gamma_x,m],dim=1));c_h=alpha*z_h+(1-alpha)*x_h
            loss=loss+torch.sum(torch.abs(x-c_h)*m)/(torch.sum(m)+1e-5)
            c_c=m*x+(1-m)*c_h;h,c=self.rnn_cell(torch.cat([c_c,m],dim=1),(h,c));imputations.append(c_c.unsqueeze(1))
        # Keep the unused classifier's zero gradients, matching label_weight=0.
        loss=loss*self.impute_weight+self.out(h).sum()*0.
        return torch.cat(imputations,dim=1),loss

class FastBRITS(nn.Module):
    def __init__(self,old):
        super().__init__();self.model=old.model
        self.model.rits_f=torch.jit.script(ScriptRITS(self.model.rits_f))
        self.model.rits_b=torch.jit.script(ScriptRITS(self.model.rits_b))
    def _result(self,values,mask):
        values=torch.where(mask.bool(),values,torch.zeros_like(values))
        a,la=self.model.rits_f(values,mask);b,lb=self.model.rits_b(values.flip(1),mask.flip(1));b=b.flip(1)
        return (a+b)/2,la+lb+torch.abs(a-b).mean()*.1
    def forward(self,values,*,mask):return self._result(values,mask)[0]
    def original_loss(self,values,mask):return self._result(values,mask)[1]

def compile_brits(old):return FastBRITS(old)
