"""Approved A5-only training, then globally barred evaluation and capacity timing."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from policy6 import OUT,REPO,BASE,COARSE,CAL,DIAG,training_modules,require_barrier
from quality6 import monitor_quality,isolation
from integrations.v125_ablation.common import (STUDY,verify_lock,data_for,file_sha256,write_json,seed_all,
    parameter_hashes,atomic_torch,state_sha,resources,synchronize)
from integrations.v125_ablation.adapter import V125Adapter
coarse_driver,cal_driver=training_modules()
from policy import frozen_hash,tensor_hash,initialize_branch,object_hash


def check_lock():
    verify_lock(STUDY);lock=json.loads((OUT/'design_lock.json').read_text())
    assert file_sha256(OUT/'audit.json')==lock['audit_sha256']
    for name,digest in lock['tools'].items():assert file_sha256(OUT/'tools'/name)==digest,('changed round6 tool',name)
    for name,digest in lock['external_tools'].items():assert file_sha256(Path(name))==digest,('changed reused source',name)
    for name,digest in lock.get('registered_evidence',{}).items():assert file_sha256(OUT/name)==digest,('changed registered evidence',name)
    for name,digest in lock['baseline_tables'].items():assert file_sha256(OUT/'baseline_tables'/name)==digest
    return json.loads((OUT/'audit.json').read_text())


def record_quality_execution(target,counts,model,data,device):
    # A completed selected phase can return without any forward after a process restart.
    if counts.get('shared_router_quality_zero',0):
        write_json(target,dict(passed=True,quality_monitor=counts,evidence_scope='observed phase forwards'))
        return
    if target.exists():
        previous=json.loads(target.read_text())
        assert previous['passed'] and previous['quality_monitor']['shared_router_quality_zero']>0
        return  # Never erase prior training/evaluation execution evidence with empty counters.
    before=state_sha(model);training=model.training
    truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device)
    try:
        model.eval()
        with torch.no_grad(),monitor_quality(model) as replay:
            model.output(torch.where(obs,truth,0.),obs,starts)
    finally:model.train(training)
    assert before==state_sha(model)
    write_json(target,dict(passed=True,quality_monitor=replay,evidence_scope='validation replay only after selected-phase reuse; training monitor counts unavailable',additional_training_updates=0))


def selection(seed,phase):
    if seed==2023:return json.loads((OUT/'audit.json').read_text())['A5_seed2023']['phases'][phase]
    folder=OUT/'A5'/f'seed_{seed}';s=json.loads((folder/phase/'selection.json').read_text())
    s['selection_rule']=s.get('selection',s.get('selection_rule'));return s


def coarse(seed,device):
    audit=check_lock();assert seed in (2024,2025)
    data=data_for('PAMAP2','sr015_bl8',STUDY);seed_all(seed);model=V125Adapter(data,'A5',seed).to(device)
    initialization=parameter_hashes(model);assert initialization==audit['initial_parameter_hashes'][str(seed)]
    folder=OUT/'A5'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);write_json(folder/'initialization.json',initialization)
    with (folder/'coarse_run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with monitor_quality(model) as q_calls:
            s=coarse_driver.train(model,data,folder,device,seed)
        assert file_sha256(Path(s['checkpoint']))==s['checkpoint_sha256']
        model.load_state_dict(torch.load(s['checkpoint'],map_location=device,weights_only=True));model.inference_path='coarse';model.requires_grad_(False).eval();model.epoch=s['selected_epoch']
        record_quality_execution(folder/'coarse_quality_execution.json',q_calls,model,data,device)
        from integrations.v125_ablation.common import evaluate
        validation=evaluate(model,data,'validation',device,batch_size=128)
        assert abs(validation['full96']['mse']-s['best_validation_mse'])<2e-6
        for epoch in range(1,101):
            row=json.loads((folder/'coarse'/f'epoch_{epoch:03}.json').read_text());a0=json.loads((COARSE/'A0'/f'seed_{seed}/coarse'/f'epoch_{epoch:03}.json').read_text())
            for key in ('train_window_order_sha256','train_actual_observation_mask_sha256','optimizer_steps'):assert row[key]==a0[key]
        write_json(folder/'coarse_complete.json',dict(complete=True,selection=s,validation=validation,history_snapshot_hash=model.memory.snapshot_hash(),
            coarse_frozen_prefix_sha256=frozen_hash(model.state_dict()),all100_actual_inputs_paired_with_A0=True))
        write_json(folder/'status.json',dict(phase='coarse_complete_waiting_global_coarse_barrier',seed=seed))


def bind_calibration_sources():
    audit=check_lock();data=data_for('PAMAP2','sr015_bl8',STUDY);sources={};initializations={}
    for seed in (2023,2024,2025):
        if seed!=2023:assert json.loads((OUT/'A5'/f'seed_{seed}/coarse_complete.json').read_text())['complete']
        s=selection(seed,'coarse');seed_all(seed);model=V125Adapter(data,'A5',seed)
        assert file_sha256(Path(s['checkpoint']))==s['checkpoint_sha256']
        model.load_state_dict(torch.load(s['checkpoint'],map_location='cpu',weights_only=True))
        binding=dict(arm='A5',model_seed=seed,coarse_checkpoint=s['checkpoint'],coarse_sha256=s['checkpoint_sha256'],
            library_sha256=model.memory.snapshot_hash(),coarse_frozen_prefix_sha256=frozen_hash(model.state_dict()),
            protocol_sha256=object_hash(audit['protocol']),training_config_sha256=object_hash(audit['protocol']['training']),coarse_selected_epoch=s['selected_epoch'])
        sources[f'A5_{seed}']=binding;initializations[f'A5_{seed}']={}
        baseline=json.loads((CAL/'source_audit.json').read_text())
        for index,width in enumerate((64,256)):
            _,initial=initialize_branch(model,index,seed)
            assert initial['novel_tensor_hash']==baseline['initializations'][f'A0_{seed}'][str(width)]['novel_tensor_hash']
            initializations[f'A5_{seed}'][str(width)]=initial
        del model;gc.collect()
    result=dict(sources=sources,initializations=initializations,protocol=audit['protocol'],parent_audit_sha256=file_sha256(OUT/'audit.json'),
        generated_only_after_all_coarse_selected=True,registered_unix=time.time())
    if (OUT/'calibration_sources.json').exists():
        old=json.loads((OUT/'calibration_sources.json').read_text());assert old['sources']==sources and old['initializations']==initializations
    else:write_json(OUT/'calibration_sources.json',result)
    write_json(OUT/'calibration_sources_lock.json',dict(sha256=file_sha256(OUT/'calibration_sources.json'),parent_design_lock_sha256=file_sha256(OUT/'design_lock.json')))
    return result


def calibration(seed,device):
    check_lock();assert seed in (2024,2025)
    sources=json.loads((OUT/'calibration_sources.json').read_text());assert file_sha256(OUT/'calibration_sources.json')==json.loads((OUT/'calibration_sources_lock.json').read_text())['sha256']
    binding=sources['sources'][f'A5_{seed}'];data=data_for('PAMAP2','sr015_bl8',STUDY)
    seed_all(seed);model=V125Adapter(data,'A5',seed).to(device);model.load_state_dict(torch.load(binding['coarse_checkpoint'],map_location=device,weights_only=True))
    folder=OUT/'A5'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'calibration_run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index in (0,1):
            with monitor_quality(model) as q_calls:
                cal_driver.train_branch(model,data,folder,device,index,seed,binding,sources)
            record_quality_execution(folder/f'calibration{(64,256)[index]}_quality_execution.json',q_calls,model,data,device)
        model.requires_grad_(False).eval();assert frozen_hash(model.state_dict())==binding['coarse_frozen_prefix_sha256']
        atomic_torch(folder/'selected.pt',model.state_dict())
        write_json(folder/'selected_provenance.json',dict(checkpoint=str(folder/'selected.pt'),checkpoint_sha256=file_sha256(folder/'selected.pt'),
            frozen_prefix_sha256=frozen_hash(model.state_dict()),tensor_sha256=tensor_hash(model.state_dict()),source_binding=binding))
        write_json(folder/'calibration_complete.json',dict(complete=True,seed=seed,source_binding=binding,weights=[.5,.5],source_file_sha256=file_sha256(OUT/'calibration_sources.json')))
        write_json(folder/'status.json',dict(phase='calibration_complete_waiting_all_checkpoint_barrier',seed=seed))


def lock_all_checkpoints():
    audit=check_lock();entries={};details={}
    for seed in (2023,2024,2025):
        if seed!=2023:assert json.loads((OUT/'A5'/f'seed_{seed}/calibration_complete.json').read_text())['complete']
        for phase in ('coarse','calibration64','calibration256'):
            s=selection(seed,phase);assert file_sha256(Path(s['checkpoint']))==s['checkpoint_sha256']
            identity=f'A5_{seed}_{phase}';entries[identity]=s['checkpoint_sha256'];details[identity]=s
        final=Path(audit['A5_seed2023']['final_checkpoint']) if seed==2023 else OUT/'A5'/f'seed_{seed}/selected.pt'
        details[f'A5_{seed}_ensemble']=dict(checkpoint=str(final),checkpoint_sha256=file_sha256(final),weights=[.5,.5])
    require_barrier(entries)
    for seed in (2023,2024,2025):
        b=audit['A0'][str(seed)];assert file_sha256(Path(b['selected_checkpoint']))==b['selected_checkpoint_sha256']
    data=dict(complete=True,locked_unix=time.time(),baseline_commit=BASE,checkpoint_hashes=entries,details=details,
        calibration_sources_sha256=file_sha256(OUT/'calibration_sources.json'),parent_design_lock_sha256=file_sha256(OUT/'design_lock.json'))
    if (OUT/'all_checkpoint_lock.json').exists():
        previous=json.loads((OUT/'all_checkpoint_lock.json').read_text());assert previous['checkpoint_hashes']==entries and previous['details']==details
    else:write_json(OUT/'all_checkpoint_lock.json',data)
    return json.loads((OUT/'all_checkpoint_lock.json').read_text())


def require_actual_barrier():
    check_lock();path=OUT/'all_checkpoint_lock.json'
    if not path.exists():raise ValueError('new test/stratification replay forbidden before all checkpoint lock')
    b=json.loads(path.read_text());assert b['complete'];require_barrier(b['checkpoint_hashes'])
    for value in b['details'].values():assert file_sha256(Path(value['checkpoint']))==value['checkpoint_sha256']
    return b


def evaluate_seed(seed,device):
    barrier=require_actual_barrier();data=data_for('PAMAP2','sr015_bl8',STUDY);seed_all(seed)
    model=V125Adapter(data,'A5',seed).to(device);folder=OUT/'A5'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    sources=json.loads((OUT/'calibration_sources.json').read_text());binding=sources['sources'][f'A5_{seed}'];results=[];full_started=time.perf_counter()
    final=barrier['details'][f'A5_{seed}_ensemble'];coarse_s=barrier['details'][f'A5_{seed}_coarse']
    for path in ('coarse','calibration64','calibration256','ensemble'):
        source=coarse_s if path=='coarse' else final
        model.load_state_dict(torch.load(source['checkpoint'],map_location=device,weights_only=True));model.requires_grad_(False).eval();model.inference_path=path
        model.epoch=coarse_s['selected_epoch'] if path=='coarse' else 0;model.batch_index=0
        assert model.memory.snapshot_hash()==binding['library_sha256'] and frozen_hash(model.state_dict())==binding['coarse_frozen_prefix_sha256']
        before=state_sha(model)
        with monitor_quality(model) as q_calls:
            for split,mask_seed in [('validation',2021)]+[('test',x) for x in data.test_seeds]:
                target=folder/'evaluation'/path/f'{split}_{mask_seed}';completed=target/'result.json'
                epoch=coarse_s['selected_epoch'] if path=='coarse' else {p:barrier['details'][f'A5_{seed}_{p}']['selected_epoch'] for p in ('calibration64','calibration256')} if path=='ensemble' else barrier['details'][f'A5_{seed}_{path}']['selected_epoch']
                meta=dict(arm='A5',model_seed=seed,split=split,mask_seed=mask_seed,phase=path,checkpoint=source['checkpoint'],checkpoint_sha256=source['checkpoint_sha256'],
                    coarse_checkpoint_sha256=binding['coarse_sha256'],history_snapshot_hash=binding['library_sha256'],selected_epoch=epoch,
                    selection_metric='validation_full96_mse' if path=='coarse' else 'fixed0.5/0.5 of own validation_MAE-selected branches' if path=='ensemble' else 'validation_full96_mae',
                    data_manifest_sha256=data.manifest_sha,mask_sha256=next(m['sha256'] for m in data.manifest['masks'] if m['split']==split and m['seed']==mask_seed),
                    all_checkpoint_lock_sha256=file_sha256(OUT/'all_checkpoint_lock.json'),baseline_commit=BASE)
                if completed.exists():
                    result=json.loads(completed.read_text());assert result['binding']==meta
                    assert file_sha256(target/'predictions.npy')==result['metrics']['predictions_sha256'];results.append(str(completed));continue
                write_json(folder/'status.json',dict(phase='post_lock_evaluation',path=path,split=split,mask_seed=mask_seed,pid=os.getpid()))
                s=coarse_driver.support(data,split,mask_seed);pred,m=coarse_driver.diagnostic.evaluate_path(model,s,device,target)
                hidden=s['native_valid'] & ~s['observed'];error=pred-s['truth'];rows=[]
                for family,label,gate in coarse_driver.diagnostic.groups(s):
                    for section,offset in [('full96',0),('recent32',64)]:
                        region=gate.copy();region[:,:offset]=False
                        rows.append(dict(section=section,family=family,group=label,**{**coarse_driver.diagnostic.metric(error,hidden & region),**coarse_driver.diagnostic.counts(s,region)}))
                if split=='validation' and path!='ensemble':
                    trace=barrier['details'][f'A5_{seed}_{path}'];v=trace.get('validation')
                    if v is None:
                        stage=OUT/'A5'/f'seed_{seed}'/path;v=json.loads((stage/f"epoch_{trace['selected_epoch']:03}.json").read_text())['validation']
                    assert max(abs(m[sec][k]-v[sec][k]) for sec in ('full96','recent32') for k in ('mse','mae'))<2e-6
                write_json(completed,dict(binding=meta,metrics=m,strata=rows));results.append(str(completed))
                print(json.dumps(dict(event='post_lock_evaluation',**meta,full96=m['full96'])),flush=True)
                del s,pred,error,rows;gc.collect()
        record_quality_execution(folder/f'{path}_evaluation_quality_execution.json',q_calls,model,data,device)
        assert before==state_sha(model)
    for split,mask in [('validation',2021)]+[('test',x) for x in data.test_seeds]:
        arrays={p:np.load(folder/'evaluation'/p/f'{split}_{mask}/predictions.npy',mmap_mode='r') for p in ('calibration64','calibration256','ensemble')}
        diff=float(np.max(np.abs(arrays['ensemble']-(.5*arrays['calibration64']+.5*arrays['calibration256']))));assert diff<3e-5
    write_json(folder/'evaluation_complete.json',dict(complete=True,seed=seed,results=results,paths=4,validation=1,test_positions=20,
        evaluation_seconds=time.perf_counter()-full_started,all_checkpoint_lock_sha256=file_sha256(OUT/'all_checkpoint_lock.json'),fixed_half_half_combination_verified=True))
    write_json(folder/'status.json',dict(phase='evaluation_complete',seed=seed))


def capacity(device):
    barrier=require_actual_barrier();audit=json.loads((OUT/'audit.json').read_text());data=data_for('PAMAP2','sr015_bl8',STUDY)
    rows=[];truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device);visible=torch.where(obs,truth,0.)
    for seed in (2023,2024,2025):
        for arm in ('A0','A5'):
            seed_all(seed);model=V125Adapter(data,arm,seed).to(device)
            if arm=='A0':coarse_checkpoint=audit['A0'][str(seed)]['source_binding']['coarse_checkpoint'];final_checkpoint=audit['A0'][str(seed)]['selected_checkpoint']
            else:coarse_checkpoint=barrier['details'][f'A5_{seed}_coarse']['checkpoint'];final_checkpoint=barrier['details'][f'A5_{seed}_ensemble']['checkpoint']
            for path in ('coarse','calibration64','calibration256','ensemble'):
                ck=coarse_checkpoint if path=='coarse' else final_checkpoint;model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.inference_path=path;model.epoch=0;model.batch_index=0
                from integrations.v125_ablation.run import configure_phase
                if path!='ensemble':trainable=sum(p.numel() for p in configure_phase(model,path))
                else:trainable=sum(p.numel() for member in model.core.recent_mean_projector.members for p in member.parameters())
                q_audit=isolation(model,data,device,path) if arm=='A5' else None
                before_capacity=state_sha(model)
                allocated=sum(p.numel() for p in model.parameters());model.requires_grad_(True).eval();model.zero_grad(set_to_none=True)
                called=set();handles=[]
                for module in model.modules():
                    def hook(mod,args):called.update(id(p) for p in mod.parameters(recurse=False))
                    handles.append(module.register_forward_pre_hook(hook))
                model(visible[:1],obs[:1],starts[:1]).square().mean().backward()
                reachable={id(p) for p in model.parameters() if p.grad is not None}
                invoked=sum(p.numel() for p in model.parameters() if id(p) in called)
                active=sum(p.numel() for p in model.parameters() if id(p) in (called | reachable))
                for h in handles:h.remove()
                model.zero_grad(set_to_none=True);model.requires_grad_(False).eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    model(visible,obs,starts);synchronize(device);times=[]
                    for _ in range(5):
                        start=time.perf_counter();model(visible,obs,starts);synchronize(device);times.append(time.perf_counter()-start)
                assert state_sha(model)==before_capacity,'capacity measurement changed checkpoint state'
                rows.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,training_seed=seed,mask_seed=2021,split='validation',phase=path,
                    checkpoint_sha256=file_sha256(Path(ck)),allocated_parameters=allocated,phase_trainable_parameters=trainable,
                    quality_execution_proof=q_audit,all_experts_active=True,uniform_fusion=False,
                    inference_invoked_parameters=invoked,inference_invoked_or_gradient_reachable_parameters=active,
                    active_count_definition='union of directly invoked module parameters and autograd-reachable parameters; some masked quality projection overhead remains allocated/invoked',
                    batch_size=128,forward_seconds=times,median_forward_seconds=float(np.median(times)),per_window_ms=float(np.median(times))*1000/128,
                    resources=resources(device),measurement='same GPU sequential after all training stopped;1warmup+5timed FP32 eval forwards;no data-loading/copy time'))
                write_json(OUT/'capacity.json',rows)
            del model;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'capacity_complete.json',dict(complete=True,paths=24,device=str(device),no_training_concurrent=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['coarse','calibration','evaluate','capacity']);p.add_argument('--seed',type=int,choices=[2023,2024,2025]);a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda:0')
    if a.phase=='coarse':coarse(a.seed,device)
    elif a.phase=='calibration':calibration(a.seed,device)
    elif a.phase=='evaluate':evaluate_seed(a.seed,device)
    else:capacity(device)
if __name__=='__main__':main()
