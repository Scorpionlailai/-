"""Original v125 calibrators only, own frozen coarse source, fixed reporting."""
import argparse
from contextlib import ExitStack
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from integrations.v125_ablation.common import (STUDY,verify_lock,data_for,file_sha256,write_json,seed_all,
    atomic_torch,rng_state,restore_rng,evaluate,resources,synchronize,state_sha)
from integrations.v125_ablation.adapter import V125Adapter
from integrations.v125_ablation.run import ledger_start,ledger_update
from policy import frozen_hash,tensor_hash,initialize_branch,calibration_loss,source_matches

OUT=Path(__file__).resolve().parents[1]
COARSE=Path('/home/xyw/paper/E24/artifacts/v125_pamap2_coarse_seed_stability_20261019')
DIAG=Path('/home/xyw/paper/E24/artifacts/v125_contribution_diagnostics_20261019')
sys.path.insert(0,str(COARSE/'tools'))
from worker import support
sys.path.insert(0,str(DIAG/'tools'))
import evaluate as diagnostic


def check_lock():
    verify_lock(STUDY);lock=json.loads((OUT/'design_lock.json').read_text())
    for name,digest in lock['tools'].items():assert file_sha256(OUT/'tools'/name)==digest,('changed execution tool',name)
    assert file_sha256(OUT/'source_audit.json')==lock['source_audit_sha256']
    for name,digest in lock['external_tools'].items():assert file_sha256(Path(name))==digest
    return json.loads((OUT/'source_audit.json').read_text())


def no_mutation(*args,**kwargs):raise AssertionError('frozen retrieval/prototype mutation attempted')


def train_branch(model,data,folder,device,index,seed,binding,audit):
    width=(64,256)[index];phase=f'calibration{width}';stage=folder/phase;stage.mkdir(exist_ok=True)
    params,initial=initialize_branch(model,index,seed)
    assert initial==audit['initializations'][f'{model.variant}_{seed}'][str(width)]
    write_json(stage/'initialization.json',initial)
    frozen=frozen_hash(model.state_dict(),index);library=model.memory.snapshot_hash()
    member=model.core.recent_mean_projector.members[index]
    optimizer=torch.optim.Adam(params,lr=.001,weight_decay=0.)
    selected_file=stage/'selected.pt';selection_file=stage/'selection.json';latest=stage/'resume.pt'
    if selection_file.exists():
        selection=json.loads(selection_file.read_text());assert source_matches(binding,selection['source_binding'])
        assert file_sha256(selected_file)==selection['checkpoint_sha256']
        member.load_state_dict(torch.load(selected_file,map_location=device,weights_only=True));return selection
    steps=0;best=float('inf');selected=None;start=1;phase_seconds=0.
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False);assert source_matches(binding,saved['source_binding'])
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        model.core.historical.retrieval.load_state_dict(saved['retrieval_ema'])
        steps=saved['steps'];best=saved['best'];selected=saved['selected'];start=saved['epoch']+1
        phase_seconds=saved['phase_seconds'];restore_rng(saved['rng']);del saved
        assert frozen_hash(model.state_dict(),index)==frozen
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    ledger=ledger_start(folder,phase);frozen_params=[p for p in model.parameters() if not p.requires_grad]
    try:
        with ExitStack() as stack:
            for name in ('stage_patch','rekey','rekey_batch','publish_epoch'):stack.enter_context(patch.object(model.memory,name,side_effect=no_mutation))
            stack.enter_context(patch.object(model,'after_step',side_effect=no_mutation))
            stack.enter_context(patch.object(model,'publish_epoch',side_effect=no_mutation))
            for epoch in range(start,101):
                begin=time.perf_counter();model.eval();member.train();model.epoch=100
                assert not model.training and member.training
                losses=[];order=hashlib.sha256();masks=hashlib.sha256()
                for batch_index,(truth,observed,starts) in enumerate(data.batches('train',128,epoch=epoch-1,seed=2023),1):
                    assert len(starts)==128
                    order.update(np.ascontiguousarray(starts,dtype=np.int64).tobytes());masks.update(observed.numpy().tobytes())
                    model.batch_index=batch_index;truth,observed=truth.to(device),observed.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    candidate=model(torch.where(observed,truth,0.),observed,starts)
                    valid=torch.from_numpy(data.valid_for(starts)).to(device)
                    loss=calibration_loss(candidate,truth,observed,valid)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite '+phase)
                    loss.backward();assert not any(p.grad is not None for p in frozen_params)
                    optimizer.step();steps+=1;losses.append(loss.item())
                assert steps==epoch*85 and batch_index==85
                assert frozen_hash(model.state_dict(),index)==frozen and model.memory.snapshot_hash()==library
                validation=evaluate(model,data,'validation',device,batch_size=128);value=validation['full96']['mae']
                assert value is not None
                if value<best:
                    best=value;selected=epoch;atomic_torch(selected_file,member.state_dict())
                synchronize(device);phase_seconds+=time.perf_counter()-begin
                record=dict(phase=phase,epoch=epoch,optimizer_steps=steps,train_loss=float(np.mean(losses)),validation=validation,
                    best_validation_mae=best,selected_epoch=selected,phase_seconds=phase_seconds,model_seed=seed,
                    training_mask_seed=2023,train_window_order_sha256=order.hexdigest(),train_actual_observation_mask_sha256=masks.hexdigest(),
                    frozen_tensor_sha256=frozen,history_snapshot_hash=library,resources=resources(device),pid=os.getpid())
                atomic_torch(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                    retrieval_ema=model.core.historical.retrieval.state_dict(),epoch=epoch,steps=steps,best=best,selected=selected,
                    phase_seconds=phase_seconds,source_binding=binding))
                write_json(stage/f'epoch_{epoch:03}.json',record);ledger_update(ledger,device)
                write_json(folder/'status.json',record);print(json.dumps(record),flush=True)
        assert steps==8500
        step_values=sorted(set(int(v['step']) for v in optimizer.state.values()));assert step_values==[8500]
        ledger_update(ledger,device,final=True)
        selection=dict(phase=phase,source_binding=binding,checkpoint=str(selected_file),checkpoint_sha256=file_sha256(selected_file),
            selected_epoch=selected,best_validation_mae=best,selection='validation_full96_mae',epochs=100,optimizer_steps=steps,
            optimizer_step_values=step_values,batch_size=128,trainable_parameters=sum(p.numel() for p in params),
            parameters_with_optimizer_state=len(optimizer.state),precision='float32',optimizer='Adam',learning_rate=.001,weight_decay=0.,
            loop_seconds=phase_seconds,training_seconds=sum(x['seconds'] for x in ledger[1]),segments=ledger[1],resources=resources(device),
            frozen_tensor_sha256=frozen,history_snapshot_hash=library,frozen_state_unchanged=True,prototype_mutations=0,
            original_coarse_mean_frozen=True,novel_initialization_sha256=initial['novel_tensor_hash'],
            complete_initialization_sha256=initial['complete_initial_tensor_hash'],reused=False)
        write_json(selection_file,selection)
        member.load_state_dict(torch.load(selected_file,map_location=device,weights_only=True));return selection
    except BaseException:
        ledger_update(ledger,device,final=True,failed=True);raise


def reuse_branches(model,folder,device,binding,record):
    assert record['eligible'] and source_matches(binding,record['observed_binding'])
    for index,width in enumerate((64,256)):
        b=record['branches'][str(width)];checkpoint=Path(b['checkpoint']);assert file_sha256(checkpoint)==b['checkpoint_sha256']
        model.core.recent_mean_projector.members[index].load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        stage=folder/f'calibration{width}';stage.mkdir(exist_ok=True)
        write_json(stage/'selection.json',dict(phase=f'calibration{width}',source_binding=binding,checkpoint=str(checkpoint),
            checkpoint_sha256=b['checkpoint_sha256'],selected_epoch=b['selected_epoch'],best_validation_mae=b['validation']['full96']['mae'],
            selection='validation_full96_mae',epochs=100,optimizer_steps=8500,batch_size=128,training_seconds=b['seconds'],
            trainable_parameters=b['trainable_parameters'],reused=True,incremental_training_seconds=0.,
            resources=dict(max_allocated_gib=b['max_allocated_gib'],max_reserved_gib=b['max_reserved_gib'],peak_memory_scope=b['peak_memory_scope']),
            frozen_state_unchanged=True,history_snapshot_hash=binding['library_sha256'],original_audit=record))
    assert frozen_hash(model.state_dict())==binding['coarse_frozen_prefix_sha256']


def evaluate_paths(model,data,folder,device,binding):
    model.requires_grad_(False).eval();model.epoch=0;model.batch_index=0
    assert model.memory.snapshot_hash()==binding['library_sha256']
    assert frozen_hash(model.state_dict())==binding['coarse_frozen_prefix_sha256']
    before=state_sha(model);started=time.perf_counter();evaluations=[]
    selections={str(width):json.loads((folder/f'calibration{width}/selection.json').read_text()) for width in (64,256)}
    branch_hashes={width:selections[width]['checkpoint_sha256'] for width in ('64','256')}
    for split,mask_seed in [('validation',2021)]+[('test',s) for s in data.test_seeds]:
        s=support(data,split,mask_seed);definitions=diagnostic.groups(s);hidden=s['native_valid'] & ~s['observed'];outputs={}
        for path in ('calibration64','calibration256','ensemble'):
            model.inference_path=path;target=folder/'evaluation'/path/f'{split}_{mask_seed}';completed=target/'result.json'
            result_binding=dict(arm=model.variant,model_seed=model.seed,split=split,mask_seed=mask_seed,phase=path,
                coarse_checkpoint=binding['coarse_checkpoint'],coarse_checkpoint_sha256=binding['coarse_sha256'],
                branch_checkpoint_sha256=branch_hashes,history_snapshot_hash=binding['library_sha256'],
                data_manifest_sha256=data.manifest_sha,protocol_sha256=binding['protocol_sha256'],training_config_sha256=binding['training_config_sha256'],
                selected_epoch={width:selections[width]['selected_epoch'] for width in ('64','256')} if path=='ensemble' else selections[path.removeprefix('calibration')]['selected_epoch'],
                selection_metric='fixed0.5/0.5 combination of own validation_MAE-selected branches' if path=='ensemble' else 'validation_full96_mae',
                mask_sha256=next(m['sha256'] for m in data.manifest['masks'] if m['split']==split and m['seed']==mask_seed),
                design_lock_sha256=file_sha256(OUT/'design_lock.json'),combination_weights=[.5,.5] if path=='ensemble' else None)
            if completed.exists():
                result=json.loads(completed.read_text());assert result['binding']==result_binding
                assert file_sha256(target/'predictions.npy')==result['metrics']['predictions_sha256']
                predictions=np.load(target/'predictions.npy',mmap_mode='r')
            else:
                write_json(folder/'status.json',dict(phase='evaluation',path=path,split=split,mask_seed=mask_seed,pid=os.getpid()))
                predictions,m=diagnostic.evaluate_path(model,s,device,target);error=predictions-s['truth'];strata=[]
                for family,label,gate in definitions:
                    for section,offset in [('full96',0),('recent32',64)]:
                        region=gate.copy();region[:,:offset]=False
                        strata.append(dict(section=section,family=family,group=label,
                            **{**diagnostic.metric(error,hidden & region),**diagnostic.counts(s,region)}))
                result=dict(binding=result_binding,metrics=m,strata=strata)
                if split=='validation' and path!='ensemble':
                    assert abs(m['full96']['mae']-selections[path.removeprefix('calibration')]['best_validation_mae'])<2e-6
                write_json(completed,result)
            outputs[path]=predictions;evaluations.append(str(completed))
            print(json.dumps(dict(event='evaluation',**result_binding,full96=result['metrics']['full96'])),flush=True)
        # Original averaging is in window-normalized coordinates, so FP32 order differs slightly from global averaging.
        max_deviation=float(np.max(np.abs(outputs['ensemble']-(outputs['calibration64']*.5+outputs['calibration256']*.5))))
        assert max_deviation<3e-5,('fixed ensemble mismatch',max_deviation)
        write_json(folder/'evaluation'/f'combination_check_{split}_{mask_seed}.json',dict(max_absolute_fp32_deviation=max_deviation,weights=[.5,.5],passed=True))
        del s,definitions,outputs;gc.collect()
    assert before==state_sha(model) and model.memory.snapshot_hash()==binding['library_sha256']
    write_json(folder/'evaluation_complete.json',dict(complete=True,arm=model.variant,model_seed=model.seed,
        paths=['calibration64','calibration256','ensemble'],validation=1,test_positions=20,results=evaluations,
        source_binding=binding,evaluation_seconds=time.perf_counter()-started,frozen_state_unchanged=True,weights=[.5,.5]))
    write_json(folder/'status.json',dict(phase='complete',arm=model.variant,model_seed=model.seed,pid=os.getpid()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['A0','A1']);p.add_argument('--seed',required=True,type=int,choices=[2023,2024,2025]);a=p.parse_args()
    audit=check_lock();torch.set_num_threads(4);device=torch.device('cuda:0');seed_all(a.seed)
    data=data_for('PAMAP2','sr015_bl8',STUDY);assert data.manifest_sha==audit['protocol']['data_manifest_sha256']
    binding=audit['sources'][f'{a.arm}_{a.seed}'];assert file_sha256(Path(binding['coarse_checkpoint']))==binding['coarse_sha256']
    folder=OUT/a.arm/f'seed_{a.seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=V125Adapter(data,a.arm,a.seed).to(device)
        model.load_state_dict(torch.load(binding['coarse_checkpoint'],map_location=device,weights_only=True))
        assert frozen_hash(model.state_dict())==binding['coarse_frozen_prefix_sha256'] and model.memory.snapshot_hash()==binding['library_sha256']
        write_json(folder/'source_binding.json',binding)
        if f'{a.arm}_{a.seed}' in audit['reuse']:reuse_branches(model,folder,device,binding,audit['reuse'][f'{a.arm}_{a.seed}'])
        else:
            for index in (0,1):train_branch(model,data,folder,device,index,a.seed,binding,audit)
        model.requires_grad_(False).eval();model.inference_path='ensemble'
        atomic_torch(folder/'selected.pt',model.state_dict())
        write_json(folder/'selected_provenance.json',dict(checkpoint_sha256=file_sha256(folder/'selected.pt'),
            tensor_sha256=tensor_hash(model.state_dict()),source_binding=binding,weights=[.5,.5]))
        evaluate_paths(model,data,folder,device,binding)
if __name__=='__main__':main()
