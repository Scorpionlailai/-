"""Dataset dimension adapters; preserve previous ETTh1 adapters and author files."""
import copy
import inspect
import os
from pathlib import Path
import sys

import torch
from torch import nn

from integrations.forecast_compare import author_profiles as author
from integrations.forecast_compare import saits_model
from integrations.forecast_compare.t1_frozen import FrozenT1ForecastModel
from integrations.forecast_compare.t1_p4 import t1_inputs
from integrations.s4m.overlay.malr_fair.data import build_s4m_sample
from integrations.s4m.search import _replace_argument

ROOT=Path(__file__).resolve().parents[2]


def build_author(name, profile, protocol, device):
    """Retain author's builders with explicit 7->21 task dimension substitutions."""
    if name=='S4M':return build_s4m(profile,protocol,device)
    namespace=dict(vars(author))
    brits=inspect.getsource(author._original_brits)
    assert brits.count('ast.Constant(7)')==1
    exec(compile(brits.replace('ast.Constant(7)','ast.Constant(21)'),__file__,'exec'),namespace)
    saits_namespace=dict(vars(saits_model),SAITS_SETTINGS={**saits_model.SAITS_SETTINGS,'d_feature':21})
    exec(compile(inspect.getsource(saits_model.build_official_saits),__file__,'exec'),saits_namespace)
    namespace['build_official_saits']=saits_namespace['build_official_saits']
    text=inspect.getsource(author.build_author_model)
    patches={'var_num=7':'var_num=21',"'channels': 7":"'channels': 21",
             '2, 7, [2, 3, 6, 7]':'2, 21, [2, 3, 6, 7]',
             "[cfg['patch_len']] * 7":"[cfg['patch_len']] * 21",
             'sampling_periods=[1] * 7, enc_in=7':'sampling_periods=[1] * 21, enc_in=21'}
    for old,new in patches.items():
        if text.count(old)!=1:raise ValueError('author dimension adapter no longer matches: '+old)
        text=text.replace(old,new)
    exec(compile(text,__file__,'exec'),namespace)
    cfg=copy.deepcopy(profile)
    if name=='T1':cfg['model']['enc_in']=21
    return namespace['build_author_model'](name,cfg,Path(cfg['root']),device)


def build_head(protocol):
    from malr.models.patchtst import PatchTSTCore
    from malr.models.repaired_forecast import MaskAwareForecastModel
    cfg=protocol.config.model.for_dataset('Weather')
    return MaskAwareForecastModel(PatchTSTCore(c_in=21,seq_len=96,pred_len=96,patch_len=16,stride=8,
        d_model=cfg.d_model,n_heads=cfg.n_heads,d_ff=cfg.d_ff,n_layers=cfg.encoder_layers,
        dropout=cfg.dropout,fc_dropout=cfg.fc_dropout,head_dropout=cfg.head_dropout,attn_dropout=cfg.attn_dropout))


def frozen_forecaster(imputer,head):
    # This wrapper is independent of T1's class: all three imputers expose values,mask.
    return FrozenT1ForecastModel(imputer,head)


class S4MWrapper(nn.Module):
    def __init__(self,core,args):
        super().__init__();self.core=core;self.args=args
    def train(self,mode=True):
        super().train(mode);self.core.mem_net.is_training=int(mode);self.core.is_training=int(mode)
        return self
    def tensors(self,batch):
        values,mask=t1_inputs(batch)
        samples=[build_s4m_sample(v,m,v_mask)
                 for v,m,v_mask in zip(values.detach().cpu().numpy(),batch.forecast_target.permute(0,2,1).detach().cpu().numpy(),batch.observed_mask.detach().cpu().numpy())]
        import numpy as np
        return [torch.from_numpy(np.stack([s[i] for s in samples])).to(values.device) for i in (0,1,6,7,8,9)]
    def warm(self,batch):
        self.core.mem_net.memory.init_bank();self.core.warmup(*self.tensors(batch))
    def forward(self,batch):return self.core(*self.tensors(batch))[:,-96:].permute(0,2,1)
    def memory_state(self):
        memory=self.core.mem_net.memory
        return {k:copy.deepcopy(getattr(memory,k)) for k in ('memory','repre','num_dic','count','idx_arr')}
    def restore_memory(self,state):
        # Author bank lives on CPU; map_location during checkpoint loading must
        # not move it away from the CPU queries used to update its clusters.
        def cpu(value):
            if torch.is_tensor(value):return value.detach().cpu()
            if isinstance(value,list):return [cpu(x) for x in value]
            return copy.deepcopy(value)
        for k,v in state.items():setattr(self.core.mem_net.memory,k,cpu(v))


def build_s4m(profile,protocol,device):
    if device.type!='cuda' or device.index not in (None,0):raise ValueError('S4M uses virtual cuda:0')
    from scripts import run_author_s4m_p4 as entry
    author_command,_verify_core=entry.author_command,entry._verify_core
    cfg=copy.deepcopy(profile)
    _verify_core(Path(cfg['root']))
    command=author_command(cfg,protocol.config_path,protocol.artifacts,protocol.artifacts/'unused_s4m_build',mode='build')
    _replace_argument(command,'--malr_dataset','Weather')
    # The bridge checks this against the immutable data config, independently of training RNG.
    _replace_argument(command,'--seed',str(protocol.config.training.seed))
    # The bridge's --seed validates the sealed data config (2021). Model RNG
    # belongs to this run's separately recorded profile seed (2023).
    invoke_source=inspect.getsource(entry._invoke)
    marker='text = source.read_text()'
    if invoke_source.count(marker)!=1:raise ValueError('S4M invocation seed adapter changed')
    invoke_source=invoke_source.replace(marker,"text = source.read_text().replace('fix_seed = args.seed', 'fix_seed = '+str(_weather_seed))")
    namespace={**vars(entry),'_weather_seed':profile['seed']}
    exec(compile(invoke_source,__file__,'exec'),namespace)
    _invoke=namespace['_invoke']
    original_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:exp,args=_invoke(command,build_only=True)
    finally:
        if original_visible is not None:os.environ['CUDA_VISIBLE_DEVICES']=original_visible
    return S4MWrapper(exp.model,args).to(device)
