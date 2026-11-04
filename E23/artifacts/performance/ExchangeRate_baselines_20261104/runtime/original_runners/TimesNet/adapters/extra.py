import sys,types
from pathlib import Path
import torch
from torch import nn
class TimesNetAdapter(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        root=Path(cfg['root']);sys.path.insert(0,str(root))
        source=root/'models/TimesNet.py'
        text=source.read_text().replace('torch.sum(mask == 1, dim=1)','torch.sum(mask == 1, dim=1).clamp_min(1)')
        namespace={'__name__':'exchange_official_timesnet','__file__':str(source)}
        exec(compile(text,str(source),'exec'),namespace)
        args=types.SimpleNamespace(task_name='imputation',seq_len=96,label_len=0,pred_len=0,enc_in=8,c_out=8,**cfg['model'])
        self.core=namespace['Model'](args)
    def forward(self,values,mask):
        return self.core(torch.where(mask.bool(),values,torch.zeros_like(values)),None,None,None,mask)
