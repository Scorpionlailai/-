"""One mean/global-level + full-window calibration head after frozen repair."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn
import bootstrap
from integrations.v125_ablation.model.recent_mean import RecentMeanProjector

@dataclass
class PrefixBatch:
    values: torch.Tensor
    observed: torch.Tensor
    explicit: object
    router: object
    points: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

class FrozenRepair(nn.Module):
    def __init__(self,repair):
        super().__init__();self.repair=repair
        self.repair.requires_grad_(False).eval();self._points=None
        self._capture_handle=repair.core.front_context.register_forward_hook(self._capture)

    def _capture(self,module,args,output):
        if not isinstance(output,tuple) or len(output)!=3:raise RuntimeError('frozen frontend must return point features')
        self._points=output[1].detach()

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def extract(self,values,observed,starts):
        self.repair.eval();self._points=None
        values=torch.where(observed,values,0)
        result,norm,_=self.repair.output(values,observed,starts)
        if self._points is None:raise RuntimeError('missing frozen point features')
        return PrefixBatch(values,observed,result.explicit,SimpleNamespace(latents=result.router.latents),
            self._points,norm.mean,norm.std)


def make_head(variables,width,seed=2023):
    if width not in (64,128):raise ValueError('single candidate width must be64 or128')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return RecentMeanProjector(variables,width=64,mode='global_dense_context_gate_v6',global_dense_width=width)


def apply_head(head,batch):
    result,_=head(batch.explicit,batch.router,batch.points,batch.mean,batch.std)
    observed=batch.explicit.sequence_observed&batch.explicit.sequence_validity
    # Preserve the old final-stage all-unobserved-window fallback without ensembling.
    active=batch.explicit.sequence_validity&~observed&observed.flatten(1).any(-1)[:,None,None]
    local=torch.where(active,result.normalized_sequence,batch.explicit.normalized_sequence)
    values=local*batch.std[:,None]+batch.mean[:,None]
    return torch.where(batch.observed,batch.values,values)

class SingleBranchModel(nn.Module):
    def __init__(self,frozen,head):
        super().__init__();self.frozen=frozen;self.head=head

    def forward(self,values,observed,starts):
        return apply_head(self.head,self.frozen.extract(values,observed,starts))


def export_single(path,model,metadata):
    from pathlib import Path
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'retrieval':model.frozen.repair.core.historical.retrieval.state_dict(),**metadata},tmp)
    tmp.replace(path)


def load_single(path,data,device):
    from model import make_model
    state=torch.load(path,map_location=device,weights_only=False)
    repair=make_model(data,state['route'],state.get('seed',2023))
    result=SingleBranchModel(FrozenRepair(repair),make_head(data.n_vars,state['width'],state.get('seed',2023))).to(device)
    result.load_state_dict(state['model'],strict=True)
    result.frozen.repair.core.historical.retrieval.load_state_dict(state['retrieval'])
    result.frozen.repair.epoch=state['coarse_epoch']
    return result.eval(),state
