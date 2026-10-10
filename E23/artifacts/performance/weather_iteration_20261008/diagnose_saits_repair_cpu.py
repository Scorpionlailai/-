"""Compare cached validation outputs with the selected official SAITS repair."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import csv
import hashlib
import importlib.util
import json
import os
import sys
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE/'saits_repair_trace_20261009'
CODE=Path('/home/xyw/.cache/codex-worktrees/e23-v107-weather-context-gate-20261009')
sys.path[:0]=[str(CODE/'src'),str(CODE)]
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.saits_model import SAITS_SETTINGS, SAITS_SOURCE_SHA256


def status(state,**kwargs):
    value=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kwargs)
    temp=ROOT/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(ROOT/'status.json')
    print(json.dumps(value),flush=True)


def main():
    os.nice(10);torch.set_num_threads(4);started=perf_counter();status('running',phase='setup')
    manifest=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json')
    dataset=json.loads(manifest.read_text());columns=dataset['columns']
    selection=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/SAITS/stage1/selection.json')
    chosen=json.loads(selection.read_text());checkpoint=Path(chosen['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==chosen['checkpoint_sha256']
    # Build only the official imputer; the Weather adapter also imports unrelated
    # forecast wrappers absent from this isolated E23 checkout.
    source=Path(chosen['profile']['root'])/'modeling/saits.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SAITS_SOURCE_SHA256
    sys.path.insert(0,chosen['profile']['root'])
    module=importlib.import_module('modeling.saits')
    assert Path(module.__file__).resolve()==source.resolve()
    cfg={**SAITS_SETTINGS,'d_feature':21}
    assert all(cfg[k]==v for k,v in chosen['profile']['model'].items())
    model=module.SAITS(**cfg,device=torch.device('cpu'))
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert all(k.startswith('model.') for k in weights['model_state'])
    model.load_state_dict({k[len('model.'):]:v for k,v in weights['model_state'].items()},strict=True)
    model.eval().requires_grad_(False)
    before=parameter_hash(model)
    cache=torch.load(BASE/'small_scope_trace_20261009/validation_outputs.pt',map_location='cpu',weights_only=False)
    assert cache['validation_mask_sha256']==chosen['validation_mask_sha256']
    totals={};outputs=[]
    with torch.no_grad():
        for i,(v97,target,mask,mean,std,t1) in enumerate(cache['batches'],1):
            true_global=target*std[:,None]+mean[:,None]
            visible=torch.where(mask,true_global,0)
            candidate=model.impute({'X':visible,'missing_mask':mask.float()})[0]
            saits=torch.where(mask,target,(candidate-mean[:,None])/std[:,None])
            groups=dict(all_hidden=~mask,small_std=~mask&std[:,None].lt(.01),ordinary_std=~mask&std[:,None].ge(.01),
                        tail32_hidden=~mask&torch.arange(96)[None,:,None].ge(64))
            for name,z in (('v97',v97),('T1',t1),('SAITS',saits)):
                error=((z-target)*std[:,None]).double()
                for group,scope in groups.items():
                    s=totals.setdefault((name,group),{k:torch.zeros(21,dtype=torch.float64) for k in ('sq','ab','n')})
                    s['sq']+=(error.square()*scope).sum((0,1));s['ab']+=(error.abs()*scope).sum((0,1));s['n']+=scope.sum((0,1))
            outputs.append(saits.cpu())
            if i==1 or i%5==0:status('running',phase='baseline_disaggregation',batch=i,batches=len(cache['batches']),seconds=perf_counter()-started)
    assert parameter_hash(model)==before
    rows=[];aggregate=[]
    for (name,group),s in totals.items():
        n=float(s['n'].sum());aggregate.append(dict(model=name,group=group,count=int(n),mse=float(s['sq'].sum()/n),mae=float(s['ab'].sum()/n)))
        for c,col in enumerate(columns):
            count=float(s['n'][c])
            if count:rows.append(dict(model=name,group=group,variable=col,count=int(count),mse=float(s['sq'][c]/count),mae=float(s['ab'][c]/count)))
    replay=next(x for x in aggregate if x['model']=='SAITS' and x['group']=='all_hidden')
    differences={metric:replay[metric]-chosen['best_validation'][metric] for metric in ('mse','mae')}
    with (ROOT/'per_variable.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    torch.save(dict(batches=outputs,validation_mask_sha256=cache['validation_mask_sha256'],source_checkpoint_sha256=chosen['checkpoint_sha256']),ROOT/'saits_validation_outputs.pt')
    result=dict(summary=aggregate,cpu_minus_original_cuda_validation=differences,weights_unchanged=True,
                source_checkpoint_sha256=chosen['checkpoint_sha256'],selected_epoch=chosen['best_epoch'],
                scope='fixed validation disaggregation, no training/future forecast/test targets',seconds=perf_counter()-started)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    assert max(abs(x) for x in differences.values())<5e-5,differences
    status('completed',exit_code=0,**result)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',exit_code=1,error=repr(exc));raise
