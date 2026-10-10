"""Validation-only check of visible-variable conditional guidance after v107."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import csv
import hashlib
import json
import os
import sys

BASE=Path(__file__).resolve().parent
RUN=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v107_weather_context_gate_20261009')
sys.path.insert(0,str(RUN))
from weather_common import protocol_for
import torch
from malr.models.e23.recent_mean import RecentMeanProjector
from screen_v104_cpu import suffix
from diagnose_joint_variable_covariance_cpu import coefficients,prediction

ROOT=BASE/'v116_conditional_after_v107_cpu_replay_audit'


def status(state,**extra):
    record=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    (ROOT/'status.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


def main():
    os.nice(10);torch.set_num_threads(4);started=perf_counter();status('running')
    selected=RUN/'E23-v107/sr010_bl8/stage1/runs/e23-stage1-mae.json'
    manifest=json.loads(selected.read_text());checkpoint=Path(manifest['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    refiner=RecentMeanProjector(21,mode='global_dense_context_gate_v6').eval()
    prefix='recent_mean_projector.'
    refiner.load_state_dict({k[len(prefix):]:v for k,v in payload['model_state'].items() if k.startswith(prefix)},strict=True)
    plan=json.loads((RUN/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0])
    endpoint=max(ref.target_start for ref in protocol.refs['train'])
    train=torch.from_numpy(protocol.standardized[:endpoint]).double()
    mean=train.mean(0);centered=train-mean;covariance=centered.T@centered/len(train)
    cached=torch.load(BASE/'v97_cpu_screen_inputs.pt',map_location='cpu',weights_only=False)
    candidates=[None,.0001,.001,.01]
    totals={str(c):{k:torch.zeros(21,dtype=torch.float64) for k in ('square','absolute','tail_square','tail_absolute','count','tail_count','used')} for c in candidates}
    with torch.no_grad():
        for row in cached['validation_rows']:
            explicit,_,_,query_mean,std,target_z,observed=row
            output,base_error=suffix(refiner,row)
            valid=output.sequence_validity;hidden=~observed&valid
            coarse=output.normalized_sequence*std[:,None]+query_mean[:,None]
            target=target_z*std[:,None]+query_mean[:,None]
            visible=torch.where(observed,explicit.observed_sequence*std[:,None]+query_mean[:,None],0)
            solved,uncertainty=coefficients(observed,covariance)
            proposed=prediction(visible,observed,mean[None,None],solved)
            assert torch.equal(proposed,prediction(torch.where(observed,visible,torch.full_like(visible,float('nan'))),observed,mean[None,None],solved))
            for c in candidates:
                active=torch.zeros_like(hidden) if c is None else hidden&(uncertainty<c)
                repaired=torch.where(active,proposed.to(coarse.dtype),coarse)
                assert torch.equal(repaired[observed],coarse[observed])
                error=base_error.double() if c is None else (repaired-target).double()
                t=totals[str(c)];e=error.masked_fill(~hidden,0)
                t['square']+=e.square().sum((0,1));t['absolute']+=e.abs().sum((0,1));t['count']+=hidden.sum((0,1))
                t['tail_square']+=e[:,-32:].square().sum((0,1));t['tail_absolute']+=e[:,-32:].abs().sum((0,1));t['tail_count']+=hidden[:,-32:].sum((0,1));t['used']+=active.sum((0,1))
    results=[];variables=[]
    names=list(protocol.config.data.datasets['Weather'].value_columns) if hasattr(protocol.config.data.datasets['Weather'],'value_columns') else list(range(21))
    for c in candidates:
        t=totals[str(c)];metrics={}
        for name,numerator,denom in [('full_mse','square','count'),('full_mae','absolute','count'),('tail_mse','tail_square','tail_count'),('tail_mae','tail_absolute','tail_count')]:
            metrics[name]=(t[numerator].sum()/t[denom].sum()).item()
        results.append(dict(posterior_variance_ratio_threshold=c,**metrics,used_hidden_fraction=(t['used'].sum()/t['count'].sum()).item()))
        for i,name in enumerate(names):
            variables.append(dict(threshold=c,variable=name,full_mse=(t['square'][i]/t['count'][i]).item(),full_mae=(t['absolute'][i]/t['count'][i]).item(),used_hidden_fraction=(t['used'][i]/t['count'][i]).item()))
    (ROOT/'replay_audit.json').write_text(json.dumps(dict(cpu_replay=results,formal_cuda=manifest['validation_repair_metrics'],absolute_difference={k:abs(results[0]['full_'+k]-manifest['validation_repair_metrics']['global_'+k]) for k in ('mse','mae')}),indent=2))
    assert abs(results[0]['full_mae']-manifest['validation_repair_metrics']['global_mae'])<1e-5
    assert abs(results[0]['full_mse']-manifest['validation_repair_metrics']['global_mse'])<1e-5
    record=dict(scope='fixed validation only; frozen formal v107 fallback; covariance from Stage1 train histories; no future/test',source_manifest=str(selected),source_checkpoint_sha256=manifest['checkpoint_sha256'],train_history_rows=endpoint,validation_windows=len(protocol.refs['validation']),results=results,hidden_nan_has_no_effect=True,seconds=perf_counter()-started)
    (ROOT/'results.json').write_text(json.dumps(record,indent=2))
    with (ROOT/'per_variable.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(variables[0]));writer.writeheader();writer.writerows(variables)
    status('completed',exit_code=0,**record)


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=repr(error),exit_code=1);raise
