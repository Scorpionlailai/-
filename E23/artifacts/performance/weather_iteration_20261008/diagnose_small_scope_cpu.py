"""Trace fixed repair weights across branches and stratify numerical extremes.

Read-only validation diagnostics. No model optimization, future targets, test
masks, CSV edits or source changes. Intermediate outputs are not retrained
ablations and cannot be reported as standalone model results.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from time import perf_counter
import csv
import hashlib
import importlib.util
import json
import os
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE/'small_scope_trace_20261009'
WEATHER = Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v107_weather_context_gate_20261009')
sys.path.insert(0,str(WEATHER))
from weather_common import protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model, load_stage1_selection, _new_memory, _starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import T1_SETTINGS
from malr.models.explicit_repair import build_explicit_repair
from malr.models.e23.scale_guard import compress_hidden_normalized_tails


def status(state, **kwargs):
    row=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kwargs)
    tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(ROOT/'status.json')
    print(json.dumps(row),flush=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def update(totals, name, z, target, std, groups):
    error=((z-target)*std[:,None]).double()
    for group, mask in groups.items():
        entry=totals.setdefault((name,group),{k:torch.zeros(21,dtype=torch.float64) for k in ('squared','absolute','count','z_squared')})
        entry['squared']+=(error.square()*mask).sum((0,1))
        entry['absolute']+=(error.abs()*mask).sum((0,1))
        entry['z_squared']+=(z.double().square()*mask).sum((0,1))
        entry['count']+=mask.sum((0,1))


def load_t1():
    path=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/T1/stage2/selection.json')
    selected=json.loads(path.read_text());checkpoint=Path(selected['checkpoint'])
    assert sha(checkpoint)==selected['checkpoint_sha256']
    spec=importlib.util.spec_from_file_location('small_scope_t1_official',Path(selected['profile']['root'])/'models/T1.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    imputer=module.Model(SimpleNamespace(**{**T1_SETTINGS,**selected['profile']['model'],'enc_in':21}))
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state']
    imputer.load_state_dict({k[len('imputer.'):]:v for k,v in payload.items() if k.startswith('imputer.')},strict=True)
    return imputer.eval().requires_grad_(False),dict(selection=str(path),checkpoint_sha256=selected['checkpoint_sha256'])


def main():
    os.nice(10);torch.set_num_threads(4)
    started=perf_counter();status('running',phase='setup')
    plan=json.loads((WEATHER/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0])
    parent=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v97_weather_cached_20261008')
    manifest=parent/'E23-v97/sr010_bl8/stage1/runs/e23-stage1.json'
    settings=load_e23_settings(parent/'v97_weather.yaml',protocol=protocol,allow_variable_scaling=True)
    payload=load_stage1_selection(manifest,protocol,settings,device=torch.device('cpu'))
    model=build_repair_model(protocol,settings,device=torch.device('cpu'))
    model.load_state_dict(payload['model_state'],strict=True)
    model.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot())
    model.eval().requires_grad_(False);before=parameter_hash(model)
    t1,t1_source=load_t1();t1_before=parameter_hash(t1)
    data_manifest=json.loads(Path(plan['dataset_manifest']).read_text())
    columns=data_manifest['columns']
    physical_mean=torch.tensor(data_manifest['train_scaler']['mean'])
    physical_std=torch.tensor(data_manifest['train_scaler']['std'])
    rain_variables=torch.tensor([c.startswith('rain ') or c.startswith('raining ') for c in columns])[None,None]
    capture={};handles=[]
    def candidate_pre(module,args):capture['candidate_before_readout']=args[-1]
    def overlap_pre(module,args):capture['inputs']=args[0]
    def overlap_post(module,args,out):capture['overlap_weights']=out
    def tail_pre(module,args):capture['after_candidate_readout']=args[0].normalized_sequence
    def tail_post(module,args,out):capture['after_recent_point_refiner']=out.normalized_sequence
    def dense_pre(module,args):capture['after_recent_mean']=args[0]
    handles.extend([
        model.point_candidate_readout.register_forward_pre_hook(candidate_pre),
        model.overlap_confidence.register_forward_pre_hook(overlap_pre),
        model.overlap_confidence.register_forward_hook(overlap_post),
        model.point_refiner.register_forward_pre_hook(tail_pre),
        model.point_refiner.register_forward_hook(tail_post),
        model.recent_mean_projector.global_dense_refiner.register_forward_pre_hook(dense_pre),
    ])
    totals={};changes={};top_z=[];top_error=[];out_cache=[]
    names=('before_candidate_readout_fixed_overlap','after_candidate_readout','after_recent_point_refiner',
           'after_recent_mean','v97_final','v105_guard','T1')
    design=dict(scope='fixed validation only, no training or future forecast targets or test data',
        hypotheses=['small query standard deviations amplify normalized values',
                    'locate the first numerical branch producing overestimates',
                    'blind tail compression may improve false events but suppress correctly recovered events'],
        checkpoint_source=str(manifest),checkpoint_sha256=sha(Path(json.loads(manifest.read_text())['checkpoint'])),
        dataset_manifest=plan['dataset_manifest'],validation_mask_sha256=protocol.mask_sha256['validation'],
        t1=t1_source,code_root=str(WEATHER),model_weights_modified=False,branch_outputs=names,
        intermediate_output_note='branch tracing, not retrained ablation; before candidate readout uses actual final overlap weights',
        group_thresholds=dict(small_std=.01,extreme_z=16.,large_z=100.,rain_event_min_physical=1e-4),
        checklist=['verify sealed parent identity','trace fixed weights','compare strata and compression harm','write findings with limitations'],
        script_sha256=sha(Path(__file__)))
    (ROOT/'design.json').write_text(json.dumps(design,indent=2))
    try:
        with torch.no_grad():
            for index,b in enumerate(iter_p4_batches(protocol,'validation',batch_size=128),1):
                capture.clear();starts=_starts(b,stride=model.stride,patches=model.patch_count,device=torch.device('cpu'))
                result=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                out=result.explicit;hidden=~b.observed_mask & out.sequence_validity
                assert torch.equal(out.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
                inp=capture['inputs']
                original=build_explicit_repair(capture['candidate_before_readout'],inp.patch_values,inp.patch_mask,
                    inp.patch_validity,inp.query_mean,inp.query_std,window_len=96,patch_len=16,stride=8,
                    candidate_weights=capture['overlap_weights']).normalized_sequence
                observed_global=torch.where(b.observed_mask,b.normalized_observed*b.query_std[:,None]+b.query_mean[:,None],0)
                t1_global=t1(observed_global,mask=b.observed_mask.float())
                t1_z=torch.where(b.observed_mask,b.normalized_observed,(t1_global-b.query_mean[:,None])/b.query_std[:,None])
                guarded=compress_hidden_normalized_tails(out.normalized_sequence,b.observed_mask,out.sequence_validity,threshold=16.)
                sequences=dict(before_candidate_readout_fixed_overlap=original,
                    after_candidate_readout=capture['after_candidate_readout'],
                    after_recent_point_refiner=capture['after_recent_point_refiner'],
                    after_recent_mean=capture['after_recent_mean'],v97_final=out.normalized_sequence,
                    v105_guard=guarded,T1=t1_z)
                true_global=b.normalized_target*b.query_std[:,None]+b.query_mean[:,None]
                true_physical=true_global*physical_std[None,None]+physical_mean[None,None]
                small=b.query_std[:,None].lt(.01)
                extreme=out.normalized_sequence.abs().gt(16)
                truth_extreme=b.normalized_target.abs().gt(16)
                rain_event=rain_variables & true_physical.gt(1e-4)
                groups=dict(all_hidden=hidden,small_std=hidden&small,ordinary_std=hidden&~small,
                    v97_abs_z_above16=hidden&extreme,v97_abs_z_above100=hidden&out.normalized_sequence.abs().gt(100),
                    v97_extreme_truth_not_extreme=hidden&extreme&~truth_extreme,
                    v97_extreme_truth_extreme=hidden&extreme&truth_extreme,
                    rain_true_zero=hidden&rain_variables&~rain_event,rain_true_event=hidden&rain_event,
                    small_std_rain_true_zero=hidden&small&rain_variables&~rain_event,
                    small_std_rain_true_event=hidden&small&rain_event,
                    tail32_hidden=hidden&torch.arange(96)[None,:,None].ge(64))
                for name,z in sequences.items():update(totals,name,z,b.normalized_target,b.query_std,groups)
                old_error=(out.normalized_sequence-b.normalized_target)*b.query_std[:,None]
                new_error=(guarded-b.normalized_target)*b.query_std[:,None]
                for group,mask in groups.items():
                    c=changes.setdefault(group,{k:0 for k in ('count','changed','helped_absolute','harmed_absolute','delta_squared','delta_absolute')})
                    c['count']+=int(mask.sum());changed=mask&(guarded!=out.normalized_sequence)
                    c['changed']+=int(changed.sum())
                    c['helped_absolute']+=int((changed&(new_error.abs()<old_error.abs()-1e-7)).sum())
                    c['harmed_absolute']+=int((changed&(new_error.abs()>old_error.abs()+1e-7)).sum())
                    c['delta_squared']+=float((new_error.double().square()-old_error.double().square())[mask].sum())
                    c['delta_absolute']+=float((new_error.double().abs()-old_error.double().abs())[mask].sum())
                for metric,container in [('abs_z',top_z),('global_abs_error',top_error)]:
                    values=(out.normalized_sequence.abs() if metric=='abs_z' else old_error.abs()).masked_fill(~hidden,-1)
                    scores,flat=torch.topk(values.flatten(),min(5,int(hidden.sum())))
                    for score,position in zip(scores.tolist(),flat.tolist()):
                        window=position//(96*21);point=(position//21)%96;var=position%21
                        row=dict(rank_by=metric,rank_value=score,variable=columns[var],
                            target_start=b.refs[window].target_start,history_point=point,
                            global_std=float(b.query_std[window,var]),global_mean=float(b.query_mean[window,var]),
                            truth_global=float(true_global[window,point,var]),truth_physical=float(true_physical[window,point,var]),
                            observed_count=int(b.observed_mask[window,:,var].sum()))
                        for name,z in sequences.items():
                            row[name+'_z']=float(z[window,point,var])
                            row[name+'_physical']=float((z[window,point,var]*b.query_std[window,var]+b.query_mean[window,var])*physical_std[var]+physical_mean[var])
                        container.append(row)
                    container.sort(key=lambda row:row['rank_value'],reverse=True);del container[20:]
                out_cache.append(tuple(v.cpu() for v in (out.normalized_sequence,b.normalized_target,b.observed_mask,b.query_mean,b.query_std,t1_z)))
                if index==1 or index%5==0:status('running',phase='branch_trace',batch=index,batches=41,seconds=perf_counter()-started)
    finally:
        for handle in handles:handle.remove()
    assert parameter_hash(model)==before and parameter_hash(t1)==t1_before
    aggregate=[];variables=[]
    for (name,group),s in totals.items():
        n=float(s['count'].sum())
        if not n:continue
        aggregate.append(dict(model=name,group=group,count=int(n),mse=float(s['squared'].sum()/n),
            mae=float(s['absolute'].sum()/n),normalized_value_square_mean=float(s['z_squared'].sum()/n)))
        for v,col in enumerate(columns):
            count=float(s['count'][v])
            if count:
                variables.append(dict(model=name,group=group,variable=col,count=int(count),
                    mse=float(s['squared'][v]/count),mae=float(s['absolute'][v]/count),
                    normalized_value_square_mean=float(s['z_squared'][v]/count)))
    for filename,rows in [('branch_strata.csv',aggregate),('branch_strata_per_variable.csv',variables),
                          ('largest_outputs_trace.csv',top_z+top_error)]:
        with (ROOT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    torch.save(dict(batches=out_cache,scope=design['scope'],source_sha256=design['checkpoint_sha256'],
                    validation_mask_sha256=protocol.mask_sha256['validation']),ROOT/'validation_outputs.pt')
    result=dict(summary=aggregate,compression_changes=changes,seconds=perf_counter()-started,
                weights_unchanged=True,test_used=False,validation_windows=len(protocol.refs['validation']),
                cpu_device=True,main_checkpoint='Stage1 validation MSE selected v97 repair',
                intermediate_outputs_not_retrained_ablations=True)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    status('completed',exit_code=0,seconds=result['seconds'],results=str(ROOT/'results.json'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',exit_code=1,error=repr(exc));raise
