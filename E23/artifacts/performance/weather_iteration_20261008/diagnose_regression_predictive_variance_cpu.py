"""Validation-only numerical V probe fitted solely to jointly visible history."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import json
import os
import torch
import sys
sys.path.insert(0,'/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v107_weather_context_gate_20261009')
from weather_common import protocol_for, ROOT as RUN

BASE=Path(__file__).resolve().parent
ROOT=BASE/'v113_regression_predictive_variance_validation'


def regression(visible_values,observed,coarse,*,minimum_r2,method,prior,prior_r2,ridge):
    # Mask first: NaN or altered hidden payloads must not enter any fitted moment.
    y=torch.where(observed,visible_values,0).double()
    m=observed.double()
    n=torch.einsum('bti,btj->bij',m,m)
    sy=torch.einsum('bti,btj->bij',y,m)
    sx=sy.transpose(1,2)
    sxy=torch.einsum('bti,btj->bij',y,y)
    syy=torch.einsum('bti,btj->bij',y.square(),m)
    sxx=syy.transpose(1,2)
    count=n.clamp_min(1)
    covariance=sxy-sx*sy/count
    vx=(sxx-sx.square()/count).clamp_min(0)
    vy=(syy-sy.square()/count).clamp_min(0)
    slope=(covariance+ridge*count*prior)/(vx+ridge*count)
    intercept=sy/count-slope*sx/count
    r2=(covariance.square()/(vx*vy).clamp_min(1e-16)).clamp(0,1)
    eye=torch.eye(y.shape[-1],dtype=torch.bool)[None]
    valid=(n>=8)&(vx>1e-8*count)&(vy>1e-8*count)&~eye&(r2>=minimum_r2)&(prior_r2>=.95)
    residual=(vy-2*slope*covariance+slope.square()*vx).clamp_min(0)
    residual_variance=(residual/(n-2).clamp_min(1)).clamp_min(1e-8)
    source_deviation=y[:,:,None,:]-sx[:,None]/count[:,None]
    predictive_variance=residual_variance[:,None]*(1+1/count[:,None]+source_deviation.square()/(vx+ridge*count)[:,None])
    values=slope[:,None]*y[:,:,None,:]+intercept[:,None]
    allowed=valid[:,None]&observed[:,:,None,:]
    scores=torch.where(allowed,predictive_variance.reciprocal(),0)
    if method=='best_source':
        ix=scores.argmax(-1,keepdim=True)
        prediction=values.gather(-1,ix).squeeze(-1)
    elif method=='weighted_sources':
        weights=scores.pow(4)
        prediction=(values*weights).sum(-1)/weights.sum(-1).clamp_min(1e-16)
    else:raise ValueError(method)
    available=scores.sum(-1)>0
    return torch.where(~observed & available,prediction.to(coarse.dtype),coarse),available


def main():
    started=perf_counter();torch.set_num_threads(4);os.nice(10)
    ROOT.mkdir(exist_ok=False)
    data=torch.load(BASE/'small_scope_trace_20261009/validation_outputs.pt',map_location='cpu',weights_only=False)['batches']
    plan=json.loads((RUN/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0])
    training_end=max(ref.target_start for ref in protocol.refs['train'])
    training=torch.from_numpy(protocol.standardized[:training_end]).double()
    centered=training-training.mean(0)
    covariance=centered.T@centered/len(centered)
    variance=covariance.diag()
    prior=covariance/variance[None].clamp_min(1e-8)
    prior_r2=covariance.square()/(variance[:,None]*variance[None]).clamp_min(1e-16)
    candidates=[(ridge,method) for ridge in (.0001,.001) for method in ('best_source','weighted_sources')]
    sums={str(candidate):{'square':0.,'absolute':0.,'tail_square':0.,'tail_absolute':0.,'count':0,'tail_count':0,'used':0,
           'variable_absolute':torch.zeros(21,dtype=torch.float64),'variable_count':torch.zeros(21,dtype=torch.float64)} for candidate in candidates}
    no_leak_checked=False
    for batch_index,(coarse_z,target_z,observed,mean,std,t1_z) in enumerate(data,1):
        coarse=coarse_z*std[:,None]+mean[:,None]
        visible=torch.where(observed,target_z,0)*std[:,None]+mean[:,None]
        visible=torch.where(observed,visible,0)
        target=target_z*std[:,None]+mean[:,None]
        for ridge,method in candidates:
            threshold=.95
            repaired,available=regression(visible,observed,coarse,minimum_r2=threshold,method=method,prior=prior,prior_r2=prior_r2,ridge=ridge)
            if batch_index==1:
                corrupted=torch.where(observed,visible,torch.full_like(visible,float('nan')))
                check,check_available=regression(corrupted,observed,coarse,minimum_r2=threshold,method=method,prior=prior,prior_r2=prior_r2,ridge=ridge)
                assert torch.equal(check,repaired) and torch.equal(check_available,available)
                no_leak_checked=True
            assert torch.equal(repaired[observed],coarse[observed])
            e=(repaired-target).double();hidden=~observed
            part=e[hidden];tail=e[:,-32:][hidden[:,-32:]];t=sums[str((ridge,method))]
            t['square']+=part.square().sum().item();t['absolute']+=part.abs().sum().item()
            t['tail_square']+=tail.square().sum().item();t['tail_absolute']+=tail.abs().sum().item()
            t['count']+=part.numel();t['tail_count']+=tail.numel();t['used']+=(hidden&available).sum().item()
            t['variable_absolute']+=torch.where(hidden,e.abs(),0).sum((0,1));t['variable_count']+=hidden.sum((0,1))
    result=[]
    for candidate in candidates:
        t=sums[str(candidate)]
        result.append(dict(slope_ridge_variance=candidate[0],minimum_r2=.95,method=candidate[1],full_mse=t['square']/t['count'],
            full_mae=t['absolute']/t['count'],tail_mse=t['tail_square']/t['tail_count'],tail_mae=t['tail_absolute']/t['tail_count'],
            used_hidden_fraction=t['used']/t['count'],variable_mae=(t['variable_absolute']/t['variable_count']).tolist()))
    record=dict(scope='read-only full fixed validation; paired visible regression, fixed v97 coarse fallback; no optimization/future/test',
                candidates=result,hidden_nan_payload_cannot_change_predictions_verified=no_leak_checked,
                minimum_joint_visible=8,selection_quality='inverse residual predictive variance estimated solely from jointly visible pairs',global_prior_min_r2=.95,prior_train_rows=training_end,prior_scope='union of Stage1 training histories only; excludes final96 training forecast-only rows',seconds=perf_counter()-started,
                caveat='local association is not causal identification; baseline is a bounded diagnostic, not a sealed formal model')
    (ROOT/'results.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__=='__main__':main()
