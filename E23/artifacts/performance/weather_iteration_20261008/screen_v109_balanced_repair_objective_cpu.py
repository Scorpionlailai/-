"""Throwaway Stage1 screen of visible-scale-balanced MAE; no model/interface change."""
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
ROOT=BASE/'v109_balanced_repair_objective_screen'
CODE=Path('/home/xyw/.cache/codex-worktrees/e23-v107-weather-context-gate-20261009')
sys.path[:0]=[str(CODE/'src'),str(CODE),str(BASE)]
from malr.models.e23.recent_mean import RecentMeanProjector
from screen_v104_cpu import suffix,score

CANDIDATES=[
    ('v109_A_global_mae_plus_0p1_robust_mae',0.,1.,0.,.25,1e-3,.1),
    ('v109_B_global_mae_plus_1_robust_mae',0.,1.,0.,.25,1e-3,1.),
    ('v109_C_robust_mae_only',0.,0.,0.,0.,1e-3,1.),
    ('v109_D_robust_mae_plus_0p1_global_mse',.1,0.,.025,0.,1e-3,1.),
]


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
        source_metadata=cached['metadata'],initial_output_exact=True,gate_architecture='same as v107, no new architecture',
        candidates=[dict(name=n,mse=m,mae=a,tail_mse=tm,tail_mae=ta,lr=lr,robust_mae=robust,robust_scale_floor=.1) for n,m,a,tm,ta,lr,robust in CANDIDATES],
        epochs=20,updates_per_candidate=160,batch_size=128,seed=2023,
        selection='save validation MAE and MSE independently; repair is primary user goal',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'design.json').write_text(json.dumps(design,indent=2));results=[]
    for name,m,a,tm,ta,lr,robust in CANDIDATES:
        model.load_state_dict(initial,strict=True);model.requires_grad_(True)
        torch.set_rng_state(cached['initial_rng']);optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0)
        choices={};history=[];candidate_start=perf_counter()
        for epoch in range(1,21):
            model.train()
            for row in cached['train_rows'][(epoch-1)%2]:
                optimizer.zero_grad(set_to_none=True);out,error=suffix(model,row)
                h=~row[-1]&out.sequence_validity;e=error[h];tail=error[:,-32:][h[:,-32:]]
                # Scale depends only on original visible history, never target values.
                local_error=error / torch.sqrt(row[4].square()+.1**2)[:,None]
                le=local_error[h];lt=local_error[:,-32:][h[:,-32:]]
                loss=m*e.square().mean()+a*e.abs().mean()+tm*tail.square().mean()+ta*tail.abs().mean()+robust*(le.abs().mean()+.25*lt.abs().mean())
                assert torch.isfinite(loss);loss.backward();optimizer.step()
            metrics,variables=score(model,cached['validation_rows']);history.append(dict(epoch=epoch,**metrics))
            for criterion in ('mse','mae'):
                if criterion not in choices or metrics['full_'+criterion]<choices[criterion]['metrics']['full_'+criterion]:
                    choices[criterion]=dict(epoch=epoch,metrics=metrics,per_variable=variables)
                    torch.save(dict(refiner_state=copy.deepcopy(model.state_dict()),**choices[criterion],source=design['scope']),
                               ROOT/(name+'_'+criterion+'_screen.pt'))
            (ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2))
            status('running',candidate=name,epoch=epoch,**metrics)
        results.append(dict(candidate=name,lr=lr,robust_mae_weight=robust,mse_selected=choices['mse'],mae_selected=choices['mae'],seconds=perf_counter()-candidate_start))
        (ROOT/'results.json').write_text(json.dumps(dict(candidates=results,seconds=perf_counter()-started,scope=design['scope']),indent=2))
    status('completed',exit_code=0,seconds=perf_counter()-started)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',exit_code=1,error=repr(exc));raise
