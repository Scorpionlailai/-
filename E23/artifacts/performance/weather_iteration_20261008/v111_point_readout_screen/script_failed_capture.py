"""Bounded Stage1 probe of the existing 65-parameter candidate output projection.

Frozen point features are cached before that projection. Overlap confidence and
tail refinement are recomputed from each updated output, not held fixed.
"""
from pathlib import Path
from dataclasses import fields, is_dataclass, replace
from datetime import datetime,timezone
from time import perf_counter
import copy
import hashlib
import itertools
import json
import os
import sys
import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE/'v111_point_readout_screen'
RUN=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v107_weather_context_gate_20261009')
sys.path.insert(0,str(RUN))
from weather_common import protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,load_stage1_selection,_new_memory,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.models.e23.coordinates import repair_scale
from malr.models.explicit_repair import build_explicit_repair


def status(state,**extra):
    record=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    temporary=ROOT/'status.tmp';temporary.write_text(json.dumps(record,indent=2));temporary.replace(ROOT/'status.json')
    print(json.dumps(record),flush=True)


def detach(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if is_dataclass(value):return replace(value,**{f.name:detach(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):return {k:detach(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(detach(v) for v in value)
    return value


def suffix(model,row):
    inputs,router,points,visible,coarse,last_features,eligible,target,observed=row
    correction=model.point_candidate_readout.output[-1](last_features).squeeze(-1)
    correction=torch.where(eligible,correction,0)*repair_scale(inputs.query_std)[:,:,None,None]
    candidates=coarse+correction/inputs.query_std[:,:,None,None]
    weights=model.overlap_confidence(inputs,router,model.decoder,points,visible,candidates)
    explicit=build_explicit_repair(candidates,inputs.patch_values,inputs.patch_mask,inputs.patch_validity,
                inputs.query_mean,inputs.query_std,window_len=96,patch_len=model.patch_length,
                stride=model.stride,candidate_weights=weights)
    explicit=replace(explicit,repaired_patches=explicit.repaired_patches.masked_fill(~explicit.validity_patches,0))
    explicit=model.point_refiner(explicit,router,inputs.query_mean,inputs.query_std)
    explicit,_=model.recent_mean_projector(explicit,router,points,inputs.query_mean,inputs.query_std)
    assert torch.equal(explicit.normalized_sequence[observed],explicit.observed_sequence[observed])
    return explicit,(explicit.normalized_sequence-target)*inputs.query_std[:,None]


@torch.no_grad()
def score(model,rows):
    model.eval()
    sums={p:{k:torch.zeros(21,dtype=torch.float64) for k in ('squared','absolute','count')} for p in ('full','tail')}
    for row in rows:
        out,error=suffix(model,row);error=error.double();hidden=~row[-1]&out.sequence_validity
        for prefix,length in [('full',96),('tail',32)]:
            e,h=error[:,-length:],hidden[:,-length:]
            sums[prefix]['squared']+=torch.where(h,e.square(),0).sum((0,1))
            sums[prefix]['absolute']+=torch.where(h,e.abs(),0).sum((0,1))
            sums[prefix]['count']+=h.sum((0,1))
    metrics={};variables={}
    for prefix,t in sums.items():
        for name,key in [('mse','squared'),('mae','absolute')]:
            metrics[prefix+'_'+name]=(t[key].sum()/t['count'].sum()).item()
            variables[prefix+'_'+name]=(t[key]/t['count']).tolist()
    return metrics,variables


@torch.no_grad()
def cache_rows(model,protocol,split,epoch,limit=None):
    model.eval();rows=[];captured={}
    def before(module,args):
        inputs,router,evidence,decoder,points,visible,coarse=args
        captured['args']=detach((inputs,router,points,visible,coarse))
    def last(module,args):captured['last_features']=detach(args[0])
    def after(module,args,out):
        inputs=args[0]
        captured['eligible']=detach(inputs.patch_validity & ~inputs.patch_mask & out[1]['valid'].any(-1)[...,None])
    handles=[model.point_candidate_readout.register_forward_pre_hook(before),
             model.point_candidate_readout.output[-1].register_forward_pre_hook(last),
             model.point_candidate_readout.register_forward_hook(after)]
    try:
        iterator=iter_p4_batches(protocol,split,batch_size=128,epoch=epoch)
        if limit:iterator=itertools.islice(iterator,limit)
        for i,batch in enumerate(iterator,1):
            starts=_starts(batch,stride=model.stride,patches=model.patch_count,device=torch.device('cpu'))
            full=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts).explicit
            row=(*captured.pop('args'),captured.pop('last_features'),captured.pop('eligible'),
                 batch.normalized_target,batch.observed_mask)
            cached,_=suffix(model,row)
            difference=((cached.normalized_sequence-full.normalized_sequence)*batch.query_std[:,None]).abs().max().item()
            assert difference<1e-6,('cached branch differs',difference)
            assert not captured
            rows.append(row)
            status('running',phase='build_point_prefix_cache',split=split,epoch=epoch,batch=i,global_max_difference=difference)
    finally:
        for handle in handles:handle.remove()
    return rows


def main():
    os.nice(10);torch.set_num_threads(4);torch.manual_seed(2023);started=perf_counter()
    status('running',phase='setup')
    plan=json.loads((RUN/'plan.json').read_text())
    protocol=protocol_for(plan,plan['scenarios'][0])
    settings=load_e23_settings(Path(plan['models'][0]['config']),protocol=protocol,allow_variable_scaling=True)
    parent_settings=load_e23_settings(Path(plan['parent_settings']),protocol=protocol,allow_variable_scaling=True)
    payload=load_stage1_selection(Path(plan['parent_selection']),protocol,parent_settings,device=torch.device('cpu'))
    model=build_repair_model(protocol,settings,device=torch.device('cpu'))
    missing=model.load_state_dict(payload['model_state'],strict=False)
    assert not missing.unexpected_keys and all('.global_dense_refiner.' in k for k in missing.missing_keys)
    model.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot())
    assert model.point_context_readout is None and model.temporal_evidence_refiner is None
    assert model.repair_input_scaling=='legacy_v1' and model.repair_coordinates=='robust_local_v1'
    model.eval().requires_grad_(False)
    last=model.point_candidate_readout.output[-1];refiner=model.recent_mean_projector
    initial={'last':copy.deepcopy(last.state_dict()),'refiner':copy.deepcopy(refiner.state_dict())}
    rng=torch.get_rng_state().clone()
    train_rows=[cache_rows(model,protocol,'train',epoch,8) for epoch in (0,1)]
    validation_rows=cache_rows(model,protocol,'validation',0)
    initial_metrics,_=score(model,validation_rows)
    expected=json.loads(Path(plan['parent_selection']).read_text())['validation_repair_metrics']
    assert abs(initial_metrics['full_mse']-expected['global_mse'])<1e-6
    assert abs(initial_metrics['full_mae']-expected['global_mae'])<1e-6
    design=dict(scope='CPU bounded Stage1 output-projection probe; two1024-window banks, full5175 fixed validation; no future or test targets; no production changes',
                initial_metrics=initial_metrics,extra_parameters=0,existing_projection_parameters=sum(p.numel() for p in last.parameters()),
                overlap_weights_recomputed=True,tail_refinement_recomputed=True,initial_cached_global_difference_below=1e-6,
                parent_selection=plan['parent_selection'],epochs=20,updates_per_candidate=160,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'design.json').write_text(json.dumps(design,indent=2))
    torch.save(dict(train_rows=train_rows,validation_rows=validation_rows,initial=initial,rng=rng,design=design),ROOT/'point_prefix_inputs.pt')
    results=[]
    for name,last_lr,refiner_lr in [('v111_A_refiner_control',0.,1e-3),('v111_B_last1e4_refiner1e3',1e-4,1e-3),('v111_C_last1e3_refiner1e3',1e-3,1e-3)]:
        last.load_state_dict(initial['last']);refiner.load_state_dict(initial['refiner'])
        model.eval().requires_grad_(False);refiner.requires_grad_(True);last.requires_grad_(last_lr>0)
        groups=[{'params':refiner.parameters(),'lr':refiner_lr}]
        if last_lr:groups.append({'params':last.parameters(),'lr':last_lr})
        optimizer=torch.optim.Adam(groups,weight_decay=0);torch.set_rng_state(rng)
        choices={};history=[];candidate_started=perf_counter()
        for epoch in range(1,21):
            model.eval();refiner.train();last.train()
            for row in train_rows[(epoch-1)%2]:
                optimizer.zero_grad(set_to_none=True);out,error=suffix(model,row)
                hidden=~row[-1]&out.sequence_validity
                loss=error[hidden].abs().mean()+.25*error[:,-32:][hidden[:,-32:]].abs().mean()
                assert torch.isfinite(loss);loss.backward()
                if last_lr:assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all()
                optimizer.step()
            metrics,variables=score(model,validation_rows);history.append(dict(epoch=epoch,**metrics))
            for criterion in ('mse','mae'):
                if criterion not in choices or metrics['full_'+criterion]<choices[criterion]['metrics']['full_'+criterion]:
                    choices[criterion]=dict(epoch=epoch,metrics=metrics,per_variable=variables)
                    torch.save(dict(last=last.state_dict(),refiner=refiner.state_dict(),**choices[criterion],scope=design['scope']),ROOT/(name+'_'+criterion+'_screen.pt'))
            (ROOT/(name+'_history.json')).write_text(json.dumps(history,indent=2))
            status('running',phase='screen',candidate=name,epoch=epoch,**metrics)
        results.append(dict(candidate=name,last_lr=last_lr,refiner_lr=refiner_lr,mse_selected=choices['mse'],mae_selected=choices['mae'],seconds=perf_counter()-candidate_started))
        (ROOT/'results.json').write_text(json.dumps(dict(candidates=results,seconds=perf_counter()-started,scope=design['scope']),indent=2))
    status('completed',exit_code=0,seconds=perf_counter()-started)


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',exit_code=1,error=repr(error));raise
