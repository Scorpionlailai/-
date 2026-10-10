"""Stage1-only short probes: objective and optimizer rate on one fixed architecture."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import copy
import hashlib
import json
import os
import sys
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE/'v119_v120_repair_loss_screen'
CODE=Path('/home/xyw/.cache/codex-worktrees/e23-v107-weather-context-gate-20261009')
sys.path[:0]=[str(CODE/'src'),str(CODE),str(BASE)]
from malr.models.e23.recent_mean import RecentMeanProjector
from malr.models.e23.global_dense import GlobalDenseRefiner
from torch import nn
from screen_v104_cpu import suffix,score

CANDIDATES=[('v119_A_small_mse',256,.02,.25),('v120_B_tail_mae_half',256,0.,.5)]


class WideDense(GlobalDenseRefiner):
    def __init__(self,width):
        super().__init__(21,contextual_gate=True)
        self.input=nn.Linear(105,width);self.input_norm=nn.LayerNorm(width)
        self.depthwise=nn.Conv1d(width,width,7,padding=3,groups=width)
        self.channel_mix=nn.Conv1d(width,width,1)
        self.position=nn.Parameter(torch.empty(1,96,width));nn.init.normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(width,4,2*width,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(width,21);self.guide_gate=nn.Linear(width,21)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.guide_gate.weight);nn.init.zeros_(self.guide_gate.bias)
        self.parameter_count=sum(p.numel() for p in self.parameters())



def status(state,**extra):
    r=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    p=ROOT/'status.tmp';p.write_text(json.dumps(r,indent=2));p.replace(ROOT/'status.json')
    print(json.dumps(r),flush=True)


def main():
    os.nice(10);torch.set_num_threads(4);started=perf_counter();status('running',phase='setup')
    cached=torch.load(BASE/'v97_cpu_screen_inputs.pt',map_location='cpu',weights_only=False)
    model=RecentMeanProjector(21,mode='global_dense_context_gate_v6')
    incompatible=model.load_state_dict(cached['initial_state'],strict=False)
    assert not incompatible.unexpected_keys and set(incompatible.missing_keys)=={
        'global_dense_refiner.guide_gate.weight','global_dense_refiner.guide_gate.bias'}
    initial=copy.deepcopy(model.state_dict())
    score_initial,_=score(model,cached['validation_rows']);assert score_initial==cached['metadata']['initial_metrics']
    design=dict(scope='throwaway CPU Stage1 candidate screen; repeated two1024-window banks; full fixed validation; no future or test',
        source_metadata=cached['metadata'],initial_output_exact=True,gate_architecture='same frozen v97 three-expert prefix and old mean layers; widen only global dense temporal calibration; original forecast head untouched',
        candidates=[dict(name=n,dense_width=w,full_mse=a,tail_mae=b,tail_mse=a*b) for n,w,a,b in CANDIDATES],objective='global MAE plus predefined small MSE or extra tail MAE, Adam1e-3',baseline_source=str(BASE/'v115_dense_capacity_screen/results.json'),
        epochs=20,updates_per_candidate=160,batch_size=128,seed=2023,
        selection='save validation MAE and MSE independently; repair is primary user goal',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'design.json').write_text(json.dumps(design,indent=2));results=[]
    for name,width,mse_weight,tail_weight in CANDIDATES:
        model=RecentMeanProjector(21,mode='global_dense_context_gate_v6')
        model.load_state_dict(initial,strict=True)
        torch.manual_seed(2023);model.global_dense_refiner=WideDense(width)
        first,_=score(model,cached['validation_rows']);assert first==score_initial
        model.requires_grad_(True)
        torch.set_rng_state(cached['initial_rng']);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0)
        choices={};history=[];candidate_start=perf_counter()
        for epoch in range(1,21):
            model.train()
            for row in cached['train_rows'][(epoch-1)%2]:
                optimizer.zero_grad(set_to_none=True);out,error=suffix(model,row)
                h=~row[-1]&out.sequence_validity;e=error[h];tail=error[:,-32:][h[:,-32:]]
                loss=e.abs().mean()+tail_weight*tail.abs().mean()+mse_weight*e.square().mean()+mse_weight*tail_weight*tail.square().mean()
                assert torch.isfinite(loss);loss.backward();optimizer.step()
            metrics,variables=score(model,cached['validation_rows']);history.append(dict(epoch=epoch,**metrics))
            for criterion in ('mse','mae'):
                if criterion not in choices or metrics['full_'+criterion]<choices[criterion]['metrics']['full_'+criterion]:
                    choices[criterion]=dict(epoch=epoch,metrics=metrics,per_variable=variables)
                    torch.save(dict(refiner_state=copy.deepcopy(model.state_dict()),**choices[criterion],source=design['scope']),
                               ROOT/(name+'_'+criterion+'_screen.pt'))
            (ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2))
            status('running',candidate=name,epoch=epoch,**metrics)
        results.append(dict(candidate=name,dense_width=width,trainable_parameters=sum(p.numel() for p in model.parameters()),mse_selected=choices['mse'],mae_selected=choices['mae'],seconds=perf_counter()-candidate_start))
        (ROOT/'results.json').write_text(json.dumps(dict(candidates=results,seconds=perf_counter()-started,scope=design['scope']),indent=2))
    status('completed',exit_code=0,seconds=perf_counter()-started)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',exit_code=1,error=repr(exc));raise
