"""Read-only Gaussian conditional variable probe with train-history covariance."""
from pathlib import Path
from time import perf_counter
import json
import sys
import torch

BASE=Path(__file__).resolve().parent
RUN=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v107_weather_context_gate_20261009')
sys.path.insert(0,str(RUN))
from weather_common import protocol_for


def coefficients(observed,covariance,noise=.0001):
    mask=observed.double()
    matrix=covariance[None,None]*mask[:,:,:,None]*mask[:,:,None,:]
    matrix=matrix+torch.diag_embed(1-mask+noise*mask)
    right=mask[:,:,:,None]*covariance[None,None]
    solved=torch.linalg.solve(matrix,right)
    posterior=(covariance.diag()[None,None]-(right*solved).sum(-2)).clamp_min(0)
    return solved,posterior/covariance.diag()[None,None]


def prediction(visible,observed,mean,solved):
    innovation=torch.where(observed,visible.double()-mean,0)
    return mean+torch.einsum('btij,bti->btj',solved,innovation)


def main():
    started=perf_counter();torch.set_num_threads(4)
    root=BASE/'v114_joint_variable_covariance_validation';root.mkdir(exist_ok=False)
    plan=json.loads((RUN/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0])
    endpoint=max(ref.target_start for ref in protocol.refs['train'])
    train=torch.from_numpy(protocol.standardized[:endpoint]).double()
    train_mean=train.mean(0);centered=train-train_mean;covariance=centered.T@centered/len(train)
    rows=torch.load(BASE/'small_scope_trace_20261009/validation_outputs.pt',map_location='cpu',weights_only=False)['batches']
    candidates=[(mean,threshold) for mean in ('train_mean','visible_window_mean') for threshold in (.001,.01,.05)]
    totals={str(c):dict(square=0.,absolute=0.,tail_square=0.,tail_absolute=0.,count=0,tail_count=0,used=0) for c in candidates}
    for index,(coarse_z,target_z,observed,mean,std,t1) in enumerate(rows,1):
        visible=torch.where(observed,target_z,0)*std[:,None]+mean[:,None]
        visible=torch.where(observed,visible,0)
        truth=target_z*std[:,None]+mean[:,None];coarse=coarse_z*std[:,None]+mean[:,None]
        solved,uncertainty=coefficients(observed,covariance)
        for mean_name,threshold in candidates:
            center=train_mean[None,None] if mean_name=='train_mean' else mean[:,None].double()
            estimated=prediction(visible,observed,center,solved)
            if index==1:
                altered=torch.where(observed,visible,torch.full_like(visible,float('nan')))
                assert torch.equal(estimated,prediction(altered,observed,center,solved))
            active=~observed & (uncertainty<threshold)
            repaired=torch.where(active,estimated.to(coarse.dtype),coarse)
            assert torch.equal(repaired[observed],coarse[observed])
            error=(repaired-truth).double();hidden=~observed
            part=error[hidden];tail=error[:,-32:][hidden[:,-32:]]
            t=totals[str((mean_name,threshold))]
            t['square']+=part.square().sum().item();t['absolute']+=part.abs().sum().item()
            t['tail_square']+=tail.square().sum().item();t['tail_absolute']+=tail.abs().sum().item()
            t['count']+=part.numel();t['tail_count']+=tail.numel();t['used']+=active.sum().item()
    results=[]
    for mean,threshold in candidates:
        t=totals[str((mean,threshold))]
        results.append(dict(mean=mean,posterior_variance_ratio_threshold=threshold,full_mse=t['square']/t['count'],
            full_mae=t['absolute']/t['count'],tail_mse=t['tail_square']/t['tail_count'],tail_mae=t['tail_absolute']/t['tail_count'],
            used_hidden_fraction=t['used']/t['count']))
    record=dict(scope='validation-only Gaussian conditional diagnostic; v97 fallback; no forecast/test; covariance fitted only to union of Stage1 training histories',
                results=results,noise=.0001,train_history_rows=endpoint,
                hidden_nan_payload_has_no_effect_verified=True,seconds=perf_counter()-started,
                caveat='covariance alone is not a learned full repair model or causal effect estimate')
    (root/'results.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))


if __name__=='__main__':main()
