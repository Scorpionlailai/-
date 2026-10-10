"""Throwaway Stage1 screen: train-history conditional prior as repair features."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import copy
import hashlib
import json
import os
import sys

BASE=Path(__file__).resolve().parent
ROOT=BASE/'v117_variable_prior_feature_screen'
RUN=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v107_weather_context_gate_20261009')
sys.path.insert(0,str(RUN))
from weather_common import protocol_for
import torch
from torch import nn
from malr.models.e23.recent_mean import RecentMeanProjector
from screen_v115_dense_capacity_cpu import WideDense
from screen_v104_cpu import suffix,score
from diagnose_joint_variable_covariance_cpu import coefficients,prediction


def prior_features(row,mean,covariance):
    explicit,_,_,query_mean,std,_,observed=row
    observed=observed&explicit.sequence_validity
    visible=torch.where(observed,explicit.observed_sequence*std[:,None]+query_mean[:,None],0)
    solved,uncertainty=coefficients(observed,covariance)
    proposed=prediction(visible,observed,mean[None,None],solved).to(visible.dtype)
    confidence=(1-uncertainty/.01).clamp_min(0).to(visible.dtype)
    confidence=torch.where(~observed&explicit.sequence_validity,confidence,0)
    # Hidden payloads must never enter the prior estimate.
    assert torch.equal(proposed,prediction(torch.where(observed,visible,torch.full_like(visible,float('nan'))),observed,mean[None,None],solved).to(visible.dtype))
    assert not confidence[~observed.any(-1)].any()
    return proposed,confidence


def status(state,**extra):
    record=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    (ROOT/'status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


def main():
    os.nice(10);torch.set_num_threads(4);started=perf_counter();status('running',phase='setup')
    cached=torch.load(BASE/'v97_cpu_screen_inputs.pt',map_location='cpu',weights_only=False)
    plan=json.loads((RUN/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0])
    endpoint=max(ref.target_start for ref in protocol.refs['train'])
    train=torch.from_numpy(protocol.standardized[:endpoint]).double()
    mean=train.mean(0);centered=train-mean;covariance=centered.T@centered/len(train)
    rows=cached['validation_rows']+sum(cached['train_rows'],[])
    priors={id(row):prior_features(row,mean,covariance) for row in rows}
    model=RecentMeanProjector(21,mode='global_dense_context_gate_v6')
    model.load_state_dict({k:v for k,v in cached['initial_state'].items() if not k.startswith('global_dense_refiner.')},strict=False)
    torch.manual_seed(2023);dense=WideDense(256);model.global_dense_refiner=dense
    old_input=dense.input
    with torch.random.fork_rng(devices=[]):
        dense.input=nn.Linear(147,256)
    with torch.no_grad():
        dense.input.weight[:,:105].copy_(old_input.weight)
        dense.input.weight[:,105:].zero_();dense.input.bias.copy_(old_input.bias)
    active_prior=[None]
    def add_features(module,args):
        proposed,confidence=active_prior[0]
        features=args[0]
        delta=(proposed-features[...,:21].detach())*confidence
        return (torch.cat((features,delta,confidence),-1),)
    dense.input.register_forward_pre_hook(add_features)
    def evaluate():
        model.eval();totals={p:{k:torch.zeros(21,dtype=torch.float64) for k in ('square','absolute','count')} for p in ('full','tail')}
        with torch.no_grad():
            for row in cached['validation_rows']:
                active_prior[0]=priors[id(row)];out,error=suffix(model,row);hidden=~row[-1]&out.sequence_validity;error=error.double()
                for scope,length in [('full',96),('tail',32)]:
                    e=error[:,-length:].masked_fill(~hidden[:,-length:],0);t=totals[scope]
                    t['square']+=e.square().sum((0,1));t['absolute']+=e.abs().sum((0,1));t['count']+=hidden[:,-length:].sum((0,1))
        metrics={};variables={}
        for scope,t in totals.items():
            for name,key in [('mse','square'),('mae','absolute')]:
                metrics[scope+'_'+name]=(t[key].sum()/t['count'].sum()).item();variables[scope+'_'+name]=(t[key]/t['count']).tolist()
        return metrics,variables
    initial,_=evaluate()
    assert all(initial[k]==cached['metadata']['initial_metrics'][k] for k in initial)
    design=dict(scope='CPU Stage1 screen only; same two1024 training mask banks and full fixed validation; no forecast/test',train_history_rows=endpoint,seed=2023,epochs=20,updates=160,extra_trainable_parameters=10752,trainable_parameters=sum(p.numel() for p in model.parameters()),width=256,prior_variance_confidence='relu(1-posterior_ratio/.01)',initial_output_exact=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baseline=str(BASE/'v115_dense_capacity_screen/results.json'))
    (ROOT/'design.json').write_text(json.dumps(design,indent=2))
    model.requires_grad_(True);torch.set_rng_state(cached['initial_rng'])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0);history=[];choices={}
    for epoch in range(1,21):
        model.train()
        for row in cached['train_rows'][(epoch-1)%2]:
            active_prior[0]=priors[id(row)];optimizer.zero_grad(set_to_none=True);out,error=suffix(model,row)
            hidden=~row[-1]&out.sequence_validity
            loss=error[hidden].abs().mean()+.25*error[:,-32:][hidden[:,-32:]].abs().mean()
            assert torch.isfinite(loss);loss.backward();optimizer.step()
        metrics,variables=evaluate();history.append(dict(epoch=epoch,**metrics))
        for criterion in ('mse','mae'):
            if criterion not in choices or metrics['full_'+criterion]<choices[criterion]['metrics']['full_'+criterion]:
                choices[criterion]=dict(epoch=epoch,metrics=metrics,per_variable=variables)
                torch.save(dict(refiner_state=copy.deepcopy(model.state_dict()),**choices[criterion],scope=design['scope']),ROOT/(criterion+'_screen.pt'))
        (ROOT/'history.json').write_text(json.dumps(history,indent=2));status('running',phase='screen',epoch=epoch,**metrics)
    record=dict(**design,mse_selected=choices['mse'],mae_selected=choices['mae'],seconds=perf_counter()-started)
    (ROOT/'results.json').write_text(json.dumps(record,indent=2));status('completed',exit_code=0,seconds=record['seconds'],mae_selected=choices['mae']['metrics'])


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',exit_code=1,error=repr(error));raise
