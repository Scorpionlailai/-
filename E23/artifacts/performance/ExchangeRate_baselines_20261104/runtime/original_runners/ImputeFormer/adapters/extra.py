from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

def execute(path,namespace,remove=()):
    text=path.read_text()
    for line in remove:
        assert line in text,line
        text=text.replace(line,'')
    exec(compile(text,str(path),'exec'),namespace)
    return namespace

def official_scheduler(optimizer):
    path=Path('/home/xyw/paper/ImputeFormer/experiments/imputeformer/scheduler.py')
    ns=execute(path,{})
    return ns['CosineSchedulerWithRestarts'](optimizer,num_warmup_steps=12,num_training_steps=300,min_factor=.1,linear_decay=.67,num_cycles=3)

class ImputeFormerAdapter(nn.Module):
    def __init__(self,cfg):
        super().__init__();root=Path(cfg['root'])/'experiments'
        u=execute(root/'tsl/nn/utils/utils.py',{},('from ..ops import expand_then_cat',))
        utils=SimpleNamespace(**u)
        dense=execute(root/'tsl/nn/base/dense.py',{'utils':utils},('from tsl.nn.utils import utils',))['Dense']
        mlp=execute(root/'tsl/nn/blocks/encoders/mlp.py',{'Dense':dense,'utils':utils},('from ...base.dense import Dense','from ...utils import utils'))['MLP']
        attention=execute(root/'imputeformer/models/Attention_layers.py',{})
        ns={'utils':utils,'MLP':mlp,**{k:attention[k] for k in ('AttentionLayer','SelfAttentionLayer','EmbeddedAttention')}}
        ns=execute(root/'imputeformer/models/ImputeFormer.py',ns,('from tsl.nn import utils','from tsl.nn.blocks.encoders import MLP','from .Attention_layers import AttentionLayer, SelfAttentionLayer, EmbeddedAttention'))
        self.core=ns['ImputeFormerModel'](num_nodes=8,input_dim=1,output_dim=1,windows=96,**cfg['model'])
    def forward(self,values,mask):
        return self.core(torch.where(mask.bool(),values,torch.zeros_like(values)).unsqueeze(-1),None,mask.unsqueeze(-1)).squeeze(-1)
    def original_loss(self,values,mask,clean):
        prediction=self(values,mask).float()
        l1=(prediction-clean).abs()[~mask.bool()].mean()
        # Official Freg with original_mask all true on this complete dataset.
        spectrum=torch.fft.fftn(prediction.unsqueeze(-1)).reshape(len(prediction),-1)
        return l1+.01*(spectrum.abs().sum(dim=1)/spectrum.numel()).mean()
