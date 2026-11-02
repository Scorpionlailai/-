"""Frozen official T1 -> unchanged R3 numeric calibration interface."""
from pathlib import Path
import sys,json,importlib.util
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
import bootstrap
T1_SETTINGS=json.loads((ROOT/'T1_settings.json').read_text())
from malr.data.normalization import masked_window_normalize,fit_train_variable_stats
from malr.models.explicit_repair import ExplicitRepairBatch
from malr.data.patching import patchify_preserving_values
from single import PrefixBatch,apply_head
from r2_variants import make_calibrator
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero

class T1Repair(nn.Module):
 def __init__(self,data):
  super().__init__();self.data=data;self.config=dict(T1_SETTINGS,enc_in=data.n_vars)
  spec=importlib.util.spec_from_file_location('_explanation_official_t1',ROOT/'T1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  self.core=mod.Model(SimpleNamespace(**self.config))
 def forward(self,values,observed,starts):
  # Same original adapter boundary: T1 receives globally standardized observations.
  values=torch.where(observed,values,0.)
  result=self.core(values,mask=observed.to(values.dtype))
  return torch.where(observed,values,result)
 def loss(self,truth,observed,starts):
  pred=self(torch.where(observed,truth,0.),observed,starts)
  valid=torch.from_numpy(self.data.valid_for(starts)).to(truth.device)
  return mean_or_zero(hidden_error(pred,truth,observed,valid).square())

class FrozenT1(nn.Module):
 def __init__(self,repair):
  super().__init__();self.repair=repair;repair.requires_grad_(False).eval()
 def train(self,mode=True):
  super().train(False);return self
 @torch.no_grad()
 def extract(self,values,observed,starts):
  self.repair.eval();values=torch.where(observed,values,0.)
  coarse=self.repair(values,observed,starts)
  data=self.repair.data
  if not hasattr(data,'_reference_fallback'):
   data._reference_fallback=fit_train_variable_stats(torch.from_numpy(data.values[:data.bounds.train.end]),eps=0.)
  norm=masked_window_normalize(values,observed,data._reference_fallback,eps=1e-5)
  local=(coarse-norm.mean[:,None])/norm.std[:,None]
  # Match R3 geometric sequence/padding validity. Native label availability is
  # separate and remains excluded by valid_for in both losses and evaluation.
  repaired=patchify_preserving_values(local,observed,16,8)
  obs=patchify_preserving_values(norm.student,observed,16,8)
  coverage=torch.zeros_like(values,dtype=torch.long)
  for i in range(repaired.values.shape[2]):coverage[:,i*8:min(i*8+16,96)]+=1
  validity=torch.ones_like(observed,dtype=torch.bool)
  explicit=ExplicitRepairBatch(local,coarse,norm.student,observed,validity,~observed,repaired.values,obs.effective_mask,repaired.validity,coverage)
  # R3 Calibration.forward uses explicit/mean/std only; no R3 latent features,
  # candidates or front-end weights are supplied to the T1 arm.
  return PrefixBatch(values,observed,explicit,None,None,norm.mean,norm.std)

class CalibratedT1(nn.Module):
 def __init__(self,repair,seed=2023):
  super().__init__();self.frozen=FrozenT1(repair);self.head=make_calibrator(repair.data.n_vars,'H_GLOBAL',seed)
 def forward(self,values,observed,starts):return apply_head(self.head,self.frozen.extract(values,observed,starts))
