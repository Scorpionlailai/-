"""Validation-only 2x2 phase diagnostics, paired branch outputs and all fixed strata."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import torch
from integrations.v125_ablation.common import data_for,STUDY,verify_lock,file_sha256,write_json,seed_all,state_sha,resources
from integrations.v125_ablation.adapter import V125Adapter

OUT=Path(__file__).resolve().parents[1]


def metric(error,mask):
    values=error[mask].astype(np.float64)
    return dict(mse=float(np.square(values).mean()) if len(values) else None,mae=float(np.abs(values).mean()) if len(values) else None,
        coordinates=int(len(values)),squared_error_sum=float(np.square(values).sum()),absolute_error_sum=float(np.abs(values).sum()))


def runs(observed):
    invisible=(~observed).transpose(0,2,1);lengths=np.zeros(invisible.shape,dtype=np.uint8)
    for b in range(len(invisible)):
        for c in range(invisible.shape[1]):
            edges=np.diff(np.r_[False,invisible[b,c],False].astype(np.int8))
            for a,z in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):lengths[b,c,a:z]=z-a
    return lengths.transpose(0,2,1)


def support(data):
    batches=list(data.batches('validation',128))
    truth=np.concatenate([t.numpy() for t,o,s in batches]);observed=np.concatenate([o.numpy() for t,o,s in batches])
    starts=np.concatenate([s for t,o,s in batches]);valid=data.valid_for(starts)
    timeline=data.masks['validation',2021];base=data.bounds.validation.start
    artificial=np.stack([timeline[s-base:s-base+96] for s in starts])
    assert np.array_equal(observed,valid & artificial)
    ids=data.segment_indices(starts)
    records=np.array([data.segments[int(i)]['record_id'] for i in ids])
    segments=np.array([data.segments[int(i)]['id'] for i in ids])
    return dict(truth=truth,observed=observed,starts=starts,native_valid=valid,artificial_mask=artificial,records=records,segments=segments)


def groups(s):
    observed=s['observed'];lengths=runs(observed);n,t,c=observed.shape
    result=[('all','all',np.ones((n,t,c),dtype=bool))]
    for lo,hi in ((1,8),(9,16),(17,32),(33,96)):
        result.append(('invisible_run',f'{lo}-{hi}',(lengths>=lo)&(lengths<=hi)))
    for a,z in ((0,32),(32,64),(64,96)):
        gate=np.zeros((1,96,1),dtype=bool);gate[:,a:z]=True
        result.append(('history',f'{a}-{z}',np.broadcast_to(gate,observed.shape)))
    fraction=observed.mean((1,2));others=(observed.sum(-1,keepdims=True)-observed)/(c-1)
    for lo,hi in ((0.,.25),(.25,.5),(.5,.75),(.75,1.)):
        take=(fraction>=lo)&((fraction<=hi) if hi==1. else (fraction<hi))
        result.append(('window_visible',f'{lo}-{hi}',np.broadcast_to(take[:,None,None],observed.shape)))
        gate=(others>=lo)&((others<=hi) if hi==1. else (others<hi))
        result.append(('other_variables_visible',f'{lo}-{hi}',gate))
    hidden=s['native_valid'] & ~observed
    for family in ('invisible_run','history','window_visible','other_variables_visible'):
        count=sum((gate & hidden).sum() for f,label,gate in result if f==family)
        assert count==hidden.sum(),family
    return result


def counts(s,gate):
    hidden=s['native_valid'] & ~s['observed'];selected=hidden & gate;used=selected.any((1,2));has_input=gate.any((1,2))
    return dict(coordinates=int(selected.sum()),windows=int(used.sum()),records=int(len(np.unique(s['records'][used]))),continuous_segments=int(len(np.unique(s['segments'][used]))),
        input_windows=int(has_input.sum()),input_records=int(len(np.unique(s['records'][has_input]))),
        native_missing_input_coordinates=int((~s['native_valid'] & gate).sum()),
        native_only_missing_input_coordinates=int((~s['native_valid'] & s['artificial_mask'] & gate).sum()),
        native_and_artificial_missing_coordinates=int((~s['native_valid'] & ~s['artificial_mask'] & gate).sum()),
        artificial_hidden_valid_coordinates=int(selected.sum()),true_visible_coordinates=int((s['observed'] & gate).sum()))


def check_a1(model,s,device):
    truth=torch.from_numpy(s['truth'][:8]).to(device);obs=torch.from_numpy(s['observed'][:8]).to(device);starts=s['starts'][:8]
    visible=torch.where(obs,truth,0.);before=state_sha(model)
    with torch.no_grad(),ExitStack() as stack:
        for name in ('historical','variable','temporal'):
            expert=getattr(model.core,name)
            method='prepare_candidates' if name=='historical' else 'prepare_sources'
            for method_name in (method,'from_cache','forward'):
                stack.enter_context(patch.object(expert,method_name,side_effect=AssertionError('disabled '+name+' '+method_name+' used')))
        result,_,_=model.output(visible,obs,starts)
        for name,item in result.evidence.items():
            assert not item.message.any() and not item.quality.any() and not item.available.any()
            assert not item.candidates.valid.any() and not item.candidates.weights.any()
            assert not result.router.deltas[name].any()
        assert not result.router.available[...,1:].any()
        assert not result.point_candidate_attention['weights'][...,1:].any()
        assert torch.equal(result.explicit.sequence_observed,obs)
    assert state_sha(model)==before
    return dict(passed=True,selected_checkpoint='own A1 coarse checkpoint',expert_prepare_message_forward='all patched to raise, no calls',
        expert_messages_quality_candidates_deltas='all zero/unavailable',point_readout='all candidate attention mass exactly zero, only null',
        downstream='tail and original level use base-only routed state; frontend reads original visible input only',state_unchanged=True)


def evaluate_path(model,s,device,folder):
    folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();before=state_sha(model)
    predictions=np.empty(s['truth'].shape,dtype=np.float32)
    calls={'original_coarse_level':0,'dense64':0,'dense256':0}
    def counter(name):
        def hook(module,args):calls[name]+=1
        return hook
    handles=[model.coarse_projector.register_forward_pre_hook(counter('original_coarse_level'))]
    for index,width in enumerate((64,256)):
        handles.append(model.core.recent_mean_projector.members[index].global_dense_refiner.register_forward_pre_hook(counter('dense'+str(width))))
    model.requires_grad_(False).eval();assert not model.training and not any(p.requires_grad for p in model.parameters())
    with torch.no_grad():
        for offset in range(0,len(s['starts']),128):
            truth=torch.from_numpy(s['truth'][offset:offset+128]).to(device)
            obs=torch.from_numpy(s['observed'][offset:offset+128]).to(device)
            visible=torch.where(obs,truth,0.)
            output=model(visible,obs,s['starts'][offset:offset+128])
            assert output.dtype==torch.float32 and torch.isfinite(output).all()
            assert torch.equal(output[obs],truth[obs])
            predictions[offset:offset+len(output)]=output.cpu().numpy()
    for handle in handles:handle.remove()
    batches=(len(s['starts'])+127)//128
    assert calls['original_coarse_level']==(batches if model.inference_path=='coarse' else 0)
    assert calls['dense64']==(batches if model.inference_path in ('calibration64','ensemble') else 0)
    assert calls['dense256']==(batches if model.inference_path in ('calibration256','ensemble') else 0)
    assert state_sha(model)==before
    error=predictions-s['truth'];hidden=s['native_valid'] & ~s['observed']
    metrics=dict(full96=metric(error,hidden),recent32=metric(error[:,64:],hidden[:,64:]),windows=len(s['starts']),
        records=int(len(np.unique(s['records']))),continuous_segments=int(len(np.unique(s['segments']))),observed_unchanged=True,state_unchanged=True,
        executed_level_and_dense_calls=calls,seconds=time.perf_counter()-start,resources=resources(device))
    np.save(folder/'predictions.npy',predictions,allow_pickle=False)
    metrics['predictions_sha256']=file_sha256(folder/'predictions.npy')
    write_json(folder/'metrics.json',metrics)
    return predictions,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args();device=torch.device(a.device)
    torch.set_num_threads(4);verify_lock(STUDY)
    design=json.loads((OUT/'diagnostic_design.json').read_text());design_sha=file_sha256(OUT/'diagnostic_design.json')
    audit=json.loads((OUT/'asset_and_training_audit.json').read_text());results=[];strata=[];differences=[];isolation=[]
    for dataset in ('PAMAP2','PhysioNet2012'):
        identity=next(x for x in audit['resolved'] if x['dataset']==dataset)
        data=data_for(dataset,identity['scene'],STUDY);s=support(data);definitions=groups(s)
        root=OUT/dataset;root.mkdir(exist_ok=True)
        for name in ('truth','observed','starts','native_valid','artificial_mask','records','segments'):
            np.save(root/(name+'.npy'),s[name],allow_pickle=False)
        write_json(root/'validation_support.json',dict(scene=identity['scene'],model_seed=identity['seed'],mask_seed=2021,
            data_manifest_sha256=data.manifest_sha,validation_mask_sha256=next(m['sha256'] for m in data.manifest['masks'] if m['split']=='validation'),
            windows=len(s['starts']),shape=list(s['truth'].shape),diagnostic_design_sha256=design_sha,
            files={name+'.npy':file_sha256(root/(name+'.npy')) for name in ('truth','observed','starts','native_valid','artificial_mask','records','segments')},
            all_counts=counts(s,np.ones_like(s['observed']))))
        cells={}
        for arm in ('A0','A1'):
            evidence=next(x for x in audit['arms'] if x['dataset']==dataset and x['arm']==arm)
            if evidence['status']!='passed':raise RuntimeError('baseline audit failed '+dataset+' '+arm)
            run=Path(evidence['run']);seed_all(identity['seed']);model=V125Adapter(data,arm,identity['seed']).to(device)
            model.load_state_dict(torch.load(run/'stage1/coarse_selected.pt',map_location=device,weights_only=True));model.inference_path='coarse'
            model.requires_grad_(False).eval();model.epoch=evidence['selected_validation_traces']['coarse']['epoch']
            if arm=='A1':isolation.append(dict(dataset=dataset,**check_a1(model,s,device)));write_json(OUT/'a1_runtime_isolation.json',isolation)
            for path in ('coarse','calibration64','calibration256','ensemble'):
                if path=='calibration64':
                    model.load_state_dict(torch.load(run/'stage1/selected.pt',map_location=device,weights_only=True));model.epoch=0
                model.inference_path=path
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                predictions,m=evaluate_path(model,s,device,root/arm/path)
                phase='coarse' if path=='coarse' else path if path!='ensemble' else 'calibration64+calibration256'
                source_checkpoint=str(run/('stage1/coarse_selected.pt' if path=='coarse' else 'stage1/selected.pt'))
                meta=dict(dataset=dataset,scene=identity['scene'],model_seed=identity['seed'],arm=arm,path=path,
                    cell=('E1' if arm=='A0' else 'E0')+('C0' if path=='coarse' else 'C1') if path in ('coarse','ensemble') else None,
                    checkpoint=source_checkpoint,checkpoint_sha256=file_sha256(source_checkpoint),history_snapshot_hash=model.memory.snapshot_hash(),
                    shared_coarse_tensor_hash=evidence['shared_coarse_and_memory_tensor_hash'],
                    selection=(evidence['selected_validation_traces'][path] if path!='ensemble' else {p:evidence['selected_validation_traces'][p] for p in ('calibration64','calibration256')}),
                    phase=phase,coarse_original_level_retained=True if path=='coarse' else 'branch-specific level copied from coarse then trained; original coarse_projector not executed',
                    data_manifest_sha256=data.manifest_sha,diagnostic_design_sha256=design_sha,**m)
                # Replay validation selection values or compare with the sealed final cache.
                if path!='ensemble':
                    trace=evidence['selected_validation_traces'][path]['validation']
                    deviation=max(abs(m[section][metric]-trace[section][metric]) for section in ('full96','recent32') for metric in ('mse','mae'))
                    assert deviation<2e-6,('validation trace replay mismatch',dataset,arm,path,deviation)
                    meta['saved_validation_trace_max_metric_deviation']=deviation
                if path=='ensemble' or arm=='A0' and path.startswith('calibration'):
                    directory=run/('repair_cache' if path=='ensemble' else 'repair_cache_'+path)/'validation_2021'
                    manifest=json.loads((directory/'manifest.json').read_text());saved=np.load(directory/'completed.npy',mmap_mode='r')
                    assert manifest['binding']['checkpoint_sha256']==evidence['checkpoint_sha256']
                    assert manifest['binding']['history_snapshot_sha256']==model.memory.snapshot_hash()
                    deviation=float(np.abs(predictions-saved).max());assert deviation<2e-5
                    meta['saved_fp32_validation_array_max_deviation']=deviation
                error=predictions-s['truth'];hidden=s['native_valid'] & ~s['observed']
                for family,label,gate in definitions:
                    for section,offset in (('full96',0),('recent32',64)):
                        region=gate.copy();region[:,:offset]=False
                        scored=hidden & region
                        row=dict(dataset=dataset,arm=arm,path=path,cell=meta['cell'],section=section,family=family,group=label,
                            **{**metric(error,scored),**counts(s,region)})
                        strata.append(row)
                if meta['cell']:cells[meta['cell']]=meta
                results.append(meta);write_json(root/arm/path/'provenance_and_metrics.json',meta)
                write_json(OUT/'evaluation_progress.json',results)
                print(json.dumps(dict(dataset=dataset,arm=arm,path=path,full96=m['full96'],recent32=m['recent32'])),flush=True)
                del predictions;gc.collect()
            del model;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        for gain in design['gains']:
            for section in ('full96','recent32'):
                for metric_name in ('mse','mae'):
                    before=cells[gain['baseline']][section][metric_name];after=cells[gain['treatment']][section][metric_name]
                    differences.append(dict(dataset=dataset,name=gain['name'],section=section,metric=metric_name,baseline_cell=gain['baseline'],treatment_cell=gain['treatment'],
                        baseline_error=before,treatment_error=after,absolute_gain=before-after,relative_gain_percent=(before-after)/before*100 if before else None))
        del data,s,definitions;gc.collect()
    write_json(OUT/'validation_comparisons.json',results);write_json(OUT/'four_cell_gains.json',differences)
    for filename,rows in [('stratified_validation.csv',strata),('four_cell_gains.csv',differences)]:
        with (OUT/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Grouped gains use the same registered rows; keep empty bins explicitly.
    grouped=[]
    for dataset in ('PAMAP2','PhysioNet2012'):
        for section in ('full96','recent32'):
            keys=sorted({(r['family'],r['group']) for r in strata if r['dataset']==dataset})
            for family,label in keys:
                cell_rows={r['cell']:r for r in strata if r['dataset']==dataset and r['section']==section and r['family']==family and r['group']==label and r['cell'] is not None}
                assert len({r['coordinates'] for r in cell_rows.values()})==1
                for gain in design['gains']:
                    for metric_name in ('mse','mae'):
                        b=cell_rows[gain['baseline']];t=cell_rows[gain['treatment']];x,y=b[metric_name],t[metric_name]
                        grouped.append(dict(dataset=dataset,section=section,family=family,group=label,gain=gain['name'],metric=metric_name,
                            baseline_error=x,treatment_error=y,absolute_gain=x-y if x is not None and y is not None else None,
                            relative_gain_percent=(x-y)/x*100 if x is not None and x!=0 and y is not None else None,
                            coordinates=b['coordinates'],windows=b['windows'],records=b['records'],continuous_segments=b['continuous_segments']))
    with (OUT/'stratified_gains.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(grouped[0]));writer.writeheader();writer.writerows(grouped)
    write_json(OUT/'evaluation_complete.json',dict(complete=True,paths=len(results),stratified_rows=len(strata),gain_rows=len(grouped),
        design_sha256=design_sha,device=str(device),no_training_or_stage2=True))

if __name__=='__main__':main()
