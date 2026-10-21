"""Full-budget, epoch-resumable training for the locked ablation matrix."""
import argparse
import fcntl
import json
import os
import time
import traceback

import numpy as np
import torch

from .adapter import V125Adapter
from .common import *
from .cache import RepairCache
from .variants import DISABLED


def ledger_start(folder, phase):
    path=folder/f'{phase}_segments.json'
    records=json.loads(path.read_text()) if path.exists() else []
    for item in records:
        if item.get('status')=='running':
            item['status']='interrupted_without_final_peak'
            item['seconds']=item.get('last_seconds',0.)
            item['unrecorded_tail_seconds']='unknown'
    record=dict(pid=os.getpid(),started_unix=time.time(),status='running',seconds=0.)
    records.append(record);write_json(path,records)
    return path,records,time.perf_counter()


def ledger_update(ledger, device, final=False, failed=False):
    path,records,started=ledger
    records[-1].update(last_seconds=time.perf_counter()-started,resources=resources(device))
    if final:
        records[-1].update(seconds=time.perf_counter()-started,status='failed' if failed else 'complete')
    write_json(path,records)


def configure_phase(model, phase):
    if phase=='coarse':
        model.inference_path='coarse'
        # Same construction order, trainability and RNG consumption as the parent.
        model.core.recent_mean_projector.requires_grad_(False)
        params=[p for p in model.parameters() if p.requires_grad]
    else:
        model.requires_grad_(False)
        member=model.core.recent_mean_projector.members[0 if phase=='calibration64' else 1]
        member.requires_grad_(True)
        model.inference_path=phase
        params=list(member.parameters())
    return params


def train_stage1(model,data,output,device,seed):
    stage=output/'stage1';stage.mkdir(exist_ok=True)
    if (stage/'selection.json').exists():
        model.load_state_dict(torch.load(stage/'selected.pt',map_location=device,weights_only=True))
        model.inference_path='ensemble';model.requires_grad_(False).eval()
        return
    ledger=ledger_start(output,'stage1')
    try:
        steps=0;selections={};phases=[]
        for phase_no,phase in enumerate(('coarse','calibration64','calibration256')):
            if phase_no==1:
                model.load_state_dict(torch.load(stage/'coarse_selected.pt',map_location=device,weights_only=True))
            if phase_no:
                member=model.core.recent_mean_projector.members[phase_no-1]
                member.load_state_dict(model.coarse_projector.state_dict(),strict=False)
                seed_all(seed)
                model.epoch=100
            params=configure_phase(model,phase)
            trainable=sum(p.numel() for p in params)
            optimizer=torch.optim.Adam(params,lr=.001,weight_decay=0.)
            latest=stage/f'{phase}_resume.pt'
            best=float('inf');start_epoch=1;phase_seconds=0.
            if latest.exists():
                saved=torch.load(latest,map_location=device,weights_only=False)
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                model.epoch=saved['model_epoch'];model.batch_index=saved['batch_index']
                retrieval_state=saved.get('retrieval_ema')
                if retrieval_state is None:
                    # Upgrade the first epoch-boundary checkpoints: EMA/frozen
                    # tensors were saved as model aliases; restore their scalar
                    # clock through the official validated EMA loader.
                    retrieval_state=model.core.historical.retrieval.state_dict()
                    retrieval_state['frozen_epoch']=saved['epoch'] if phase=='coarse' and 'historical' not in model.disabled else 0
                model.core.historical.retrieval.load_state_dict(retrieval_state)
                steps=saved['steps'];selections=saved['selections'];best=saved['best']
                start_epoch=saved['epoch']+1;phase_seconds=saved['phase_seconds']
                restore_rng(saved['rng']);del saved
            for epoch in range(start_epoch,101):
                begin=time.perf_counter()
                if phase=='coarse':
                    model.epoch=epoch
                    if 'historical' not in model.disabled:model.core.historical.retrieval.begin_epoch(epoch)
                    model.train()
                else:
                    model.eval();member.train()
                losses=[]
                for index,(truth,observed,starts) in enumerate(data.batches('train',128,epoch=epoch-1,seed=2023),1):
                    model.batch_index=index
                    truth,observed=truth.to(device),observed.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if phase=='coarse':loss=model.loss(truth,observed,starts)
                    else:
                        candidate=model(torch.where(observed,truth,0.),observed,starts)
                        valid=torch.from_numpy(data.valid_for(starts)).to(device)
                        loss=mean_or_zero(hidden_error(candidate,truth,observed,valid).abs())+.25*mean_or_zero(hidden_error(candidate[:,-32:],truth[:,-32:],observed[:,-32:],valid[:,-32:]).abs())
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite '+phase)
                    loss.backward();optimizer.step();steps+=1;losses.append(loss.item())
                    if phase=='coarse':model.after_step()
                if phase=='coarse':model.publish_epoch(epoch)
                validation=evaluate(model,data,'validation',device,batch_size=128)
                key='mse' if phase=='coarse' else 'mae'
                value=validation['full96'][key]
                if value is None:raise ValueError('validation has no valid hidden coordinates')
                if value<best:
                    best=value;selections[phase]=epoch
                    atomic_torch(stage/f'{phase}_selected.pt',model.state_dict() if phase=='coarse' else member.state_dict())
                synchronize(device);phase_seconds+=time.perf_counter()-begin
                record=dict(subphase=phase,epoch=epoch,total_epoch=phase_no*100+epoch,optimizer_steps=steps,
                    train_loss=float(np.mean(losses)),validation=validation,phase_seconds=phase_seconds,
                    resources=resources(device),trainable_parameters=trainable,model_seed=seed,training_mask_seed=2023)
                atomic_torch(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=None,
                    rng=rng_state(),retrieval_ema=model.core.historical.retrieval.state_dict(),
                    epoch=epoch,model_epoch=model.epoch,batch_index=model.batch_index,
                    steps=steps,selections=selections,best=best,phase_seconds=phase_seconds,resources=resources(device)))
                append(stage/'curve.jsonl',record);ledger_update(ledger,device)
                write_json(output/'status.json',dict(phase='stage1_training',**record))
                print(json.dumps(record),flush=True)
            expected=(len(data.starts['train'])//128)*100*(phase_no+1)
            if steps!=expected:raise AssertionError('incorrect full-budget update count')
            phases.append(dict(phase=phase,seconds=phase_seconds,trainable_parameters=trainable,epochs=100,selection=selections[phase]))
            if phase_no:
                member.load_state_dict(torch.load(stage/f'{phase}_selected.pt',map_location=device,weights_only=True))
            del optimizer
        model.inference_path='ensemble';model.requires_grad_(False).eval()
        atomic_torch(stage/'selected.pt',model.state_dict())
        ledger_update(ledger,device,final=True)
        write_json(stage/'selection.json',dict(actual_epochs=300,batch_size=128,optimizer_steps=steps,
            selected_epochs=selections,combination_weights=[.5,.5],phases=phases,
            checkpoint_sha256=file_sha256(stage/'selected.pt'),history_snapshot_hash=model.memory.snapshot_hash(),
            segments=json.loads(ledger[0].read_text()),training_seconds=sum(x['seconds'] for x in ledger[1]),
            model_seed=seed,training_mask_seed=2023,precision='float32',scaler=None,resources=resources(device)))
    except BaseException:
        ledger_update(ledger,device,final=True,failed=True);raise


def evaluate_bank(model,data,output,device,label,head=None,cache=None):
    path=output/'evaluation'/f'{label}.json'
    if path.exists():return
    folder=path.parent;folder.mkdir(exist_ok=True)
    started=time.perf_counter();rows=[]
    before=state_sha(model)
    for seed in data.test_seeds:
        item=evaluate(model,data,'test',device,batch_size=128,mask_seed=seed,head=head,cache=cache)
        item['mask_sha256']=next(x['sha256'] for x in data.manifest['masks'] if x['split']=='test' and x['seed']==seed)
        rows.append(item);write_json(folder/f'{label}_{seed}.json',item)
    assert state_sha(model)==before
    write_json(path,dict(groups=rows,aggregate=aggregate(rows),test_seconds=time.perf_counter()-started,
                         training_runs=1,position_groups=20,position_standard_deviation_ddof=1))


def train_stage2(model,data,output,device,cache,seed):
    from integrations.pamap2_baselines.run import build_head
    stage=output/'stage2';stage.mkdir(exist_ok=True)
    if (stage/'selection.json').exists():return
    model.requires_grad_(False).eval()
    for p in model.parameters():p.grad=None
    head=build_head(data,device)
    # All arms at seed2023 share the parent public PatchTST initialization.
    initial=state_sha(head);repair_initial=state_sha(model)
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-4,weight_decay=0.)
    best=dict(mae=float('inf'),mse=float('inf'));selected={};steps=0;start=0
    latest=stage/'resume.pt';ledger=ledger_start(output,'stage2')
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        best=saved['best'];selected=saved['selected'];steps=saved['steps'];start=saved['epoch']
    try:
        for epoch in range(start,100):
            model.eval();head.train()
            for completed,observed,starts in cache.batches('train',128,epoch=epoch):
                assert len(starts)==128, 'forecast BN requires full effective batch'
                completed,observed=completed.to(device),observed.to(device)
                prediction=forecast(head,completed,observed,data,starts)
                future=torch.from_numpy(np.stack([data.values[s+96:s+192] for s in starts])).to(device).permute(0,2,1)
                valid=torch.from_numpy(data.valid_for(starts,offset=96)).to(device).permute(0,2,1)
                loss=mean_or_zero((prediction-future)[valid].square())
                if not torch.isfinite(loss):raise FloatingPointError('forecast loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
                assert not any(p.grad is not None for p in model.parameters())
            validation=evaluate(model,data,'validation',device,batch_size=128,head=head,cache=cache)
            assert state_sha(model)==repair_initial and not model.training
            for key in best:
                value=validation['forecast'][key]
                if value<best[key]:
                    best[key]=value;selected[key]=epoch+1;atomic_torch(stage/f'selected_{key}.pt',head.state_dict())
            atomic_torch(latest,dict(head=head.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                scaler=None,best=best,selected=selected,steps=steps,epoch=epoch+1))
            record=dict(epoch=epoch+1,optimizer_steps=steps,validation=validation,resources=resources(device))
            append(stage/'curve.jsonl',record);ledger_update(ledger,device)
            write_json(output/'status.json',dict(phase='stage2_training',**record));print(json.dumps(record),flush=True)
        assert steps==len(data.starts['train'])//128*100
        ledger_update(ledger,device,final=True)
        write_json(stage/'selection.json',dict(epochs=100,batch_size=128,optimizer_steps=steps,precision='float32',
            head_initial_sha256=initial,repair_state_sha256=repair_initial,model_seed=seed,head_seed=2023,
            training_mask_seed=2023,learning_rate=1e-4,weight_decay=0.,loss='pure_forecast_mse',
            main_selection='validation_mae',secondary_selection='validation_mse',selected_epochs=selected,
            parent_stage1_sha256=file_sha256(output/'stage1/selected.pt'),
            checkpoints={key:file_sha256(stage/f'selected_{key}.pt') for key in best},
            segments=ledger[1],training_seconds=sum(x['seconds'] for x in ledger[1]),resources=resources(device)))
    except BaseException:
        ledger_update(ledger,device,final=True,failed=True);raise


def execute(args):
    verify_lock(args.root)
    data=data_for(args.dataset,args.scene,args.root)
    device=torch.device('cuda:0')
    output=output_path(args.root,args.dataset,args.scene,args.variant,args.seed);output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seed_all(args.seed);model=V125Adapter(data,args.variant,args.seed).to(device)
        initialization=parameter_hashes(model)
        config=dict(variant=args.variant,model_seed=args.seed,training_mask_seed=2023,data_manifest_sha256=data.manifest_sha,
            config=model.config,initialization=initialization,disabled_experts=sorted(model.disabled),
            formal_tests='2026101801-2026101820',design_lock_sha256=file_sha256(args.root/'design_lock.json'))
        plan=output/'plan.json'
        if plan.exists() and json.loads(plan.read_text())!=config:raise ValueError('changed run binding')
        write_json(plan,config)
        if args.phase=='stage1':
            train_stage1(model,data,output,device,args.seed)
            from .capacity import measure
            measure(model,data,output,device)
            model.epoch=0;model.batch_index=0
            # Raw FP32 validation/test arrays are retained for both paired branches.
            if args.variant=='A0':
                for path in ('calibration64','calibration256'):
                    model.inference_path=path
                    branch_cache=RepairCache(model,data,output,device,128)
                    branch_cache.load('validation')
                    for seed in data.test_seeds:branch_cache.load('test',seed)
                    branch_cache.allow_generation=False
                    write_json(output/f'{path}_validation.json',evaluate(model,data,'validation',device,batch_size=128,cache=branch_cache))
                    evaluate_bank(model,data,output,device,path,cache=branch_cache)
                    write_json(output/f'{path}_cache_audit.json',dict(records=branch_cache.records,
                        generation_seconds=sum(r['generation_seconds'] for r in branch_cache.records.values())))
                    del branch_cache
            model.inference_path='ensemble'
            cache=RepairCache(model,data,output,device,128)
            for split in ('train','validation'):cache.load(split)
            for seed in data.test_seeds:cache.load('test',seed)
            write_json(output/'cache_audit.json',dict(records=cache.records,
                generation_seconds=sum(r['generation_seconds'] for r in cache.records.values()),
                train_validation_generation_seconds=sum(r['generation_seconds'] for k,r in cache.records.items() if not k.startswith('test_')),
                test_generation_seconds=sum(r['generation_seconds'] for k,r in cache.records.items() if k.startswith('test_'))))
            cache.allow_generation=False
            evaluate_bank(model,data,output,device,'repair',cache=cache)
            write_json(output/'status.json',dict(phase='stage1_complete'))
        elif args.phase=='stage2':
            model.load_state_dict(torch.load(output/'stage1/selected.pt',map_location=device,weights_only=True))
            model.requires_grad_(False).eval()
            cache=RepairCache(model,data,output,device,128,allow_generation=False)
            for split in ('train','validation'):cache.load(split)
            for seed in data.test_seeds:cache.load('test',seed)
            train_stage2(model,data,output,device,cache,args.seed)
            from integrations.pamap2_baselines.run import build_head
            head=build_head(data,device)
            for key in ('mae','mse'):
                head.load_state_dict(torch.load(output/f'stage2/selected_{key}.pt',map_location=device,weights_only=True))
                evaluate_bank(model,data,output,device,'forecast_'+key,head=head,cache=cache)
            write_json(output/'status.json',dict(phase='stage2_complete'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=STUDY)
    p.add_argument('--dataset',choices=['PAMAP2','PhysioNet2012'],required=True)
    p.add_argument('--scene',choices=['sr015_bl8','sr010_bl16'],required=True)
    p.add_argument('--variant',choices=list(DISABLED),required=True)
    p.add_argument('--seed',type=int,choices=[2023,2024,2025],default=2023)
    p.add_argument('--phase',choices=['stage1','stage2'],required=True)
    args=p.parse_args();torch.set_num_threads(4)
    try:execute(args)
    except BaseException as exc:
        output=output_path(args.root,args.dataset,args.scene,args.variant,args.seed);output.mkdir(parents=True,exist_ok=True)
        write_json(output/'status.json',dict(phase='failed',error=str(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
