"""Train/evaluate one author baseline under the sealed Weather data protocol."""
from time import perf_counter
PROCESS_STARTED=perf_counter()

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from integrations.e24.data import read_manifest,load_dataset_protocol,with_test_mask
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import _atomic_json,_atomic_torch_save
from integrations.forecast_compare.author_profiles import load_author_profiles,epoch_end_learning_rate,author_coifnet_loss
from integrations.forecast_compare.author_provenance import author_source_identity
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources,selected_training_resources
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history
from integrations.forecast_compare.t1_frozen import FrozenT1ForecastModel
from integrations.forecast_compare.crib_p4 import crib_input,crib_training_loss
from integrations.forecast_compare.bitgraph_p4 import bitgraph_inputs
from integrations.forecast_compare.ctf_p4 import ctf_inputs,group_by_full_patch_visibility
from malr.evaluation.report import file_sha256
from .models import ROOT,build_author,build_head,frozen_forecaster,S4MWrapper


def coifnet_inputs(*args):
    from integrations.forecast_compare.coifnet_p4 import coifnet_inputs as implementation
    return implementation(*args)


def load_timestamp_features(*args):
    from integrations.forecast_compare.coifnet_p4 import load_timestamp_features as implementation
    return implementation(*args)


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def autocast(training,device):
    dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}.get(training['precision'])
    return torch.autocast('cuda',dtype=dtype) if device.type=='cuda' and dtype is not None else nullcontext()


def freeze_hook(model):
    def save_output(module,args,kwargs,output):module._weather_candidate=output.detach()
    model.imputer.register_forward_hook(save_output,with_kwargs=True)


def forward(name,model,batch,stamps):
    if isinstance(model,FrozenT1ForecastModel):
        prediction=model(batch)
        values,_=t1_inputs(batch)
        return prediction,torch.where(batch.observed_mask,values,model.imputer._weather_candidate.float())
    if name=='CRIB':return model(crib_input(batch),test_flag=True)[4].permute(0,2,1),None
    if name=='CoIFNet':
        values,mask,time=coifnet_inputs(batch,stamps)
        output=model.impute(values,mask,feat=time)
        return output[:,96:].permute(0,2,1),torch.where(batch.observed_mask,values,output[:,:96])
    if name=='BiTGraph':
        values,mask=bitgraph_inputs(batch)
        return model(values,mask,0).squeeze(-1).permute(0,2,1),None
    if name=='S4M':return model(batch),None
    values,mask=ctf_inputs(batch);prediction=torch.empty_like(batch.forecast_target)
    for indices in group_by_full_patch_visibility(mask):
        prediction[indices]=model(values[indices],None,None,None,missing_flag=mask[indices]).permute(0,2,1)
    return prediction,None


def objective(name,model,batch,stamps,cfg,stage):
    if stage=='repair':
        values,mask=t1_inputs(batch)
        if name=='SAITS':return model.original_loss(values,mask,t1_clean_history(batch))
        if name=='BRITS':return model.original_loss(values,mask)
        return (model(values,mask=mask).float()-t1_clean_history(batch)).square()[~batch.observed_mask].mean()
    if isinstance(model,FrozenT1ForecastModel):return (model(batch)-batch.forecast_target).square().mean()
    if name=='CRIB':return crib_training_loss(model(crib_input(batch)),batch.forecast_target.permute(0,2,1))
    if name=='CoIFNet':
        values,mask,time=coifnet_inputs(batch,stamps)
        return author_coifnet_loss(model.impute(values,mask,feat=time),values,mask,
                                   batch.forecast_target.permute(0,2,1),weight=cfg['model']['loss_lambda'])
    prediction,_=forward(name,model,batch,stamps);error=prediction-batch.forecast_target
    return error.abs().mean() if name=='BiTGraph' else error.square().mean()


@torch.no_grad()
def score(name,model,protocol,split,device,stamps,training,stage,*,limit=None):
    model.eval()
    squared=absolute=count=windows=0
    repair={k:0 for k in ('sq96','abs96','n96','sq32','abs32','n32')}
    eval_batch=16 if name=='S4M' else 32 if name=='ChannelTokenFormer' else 128
    # Hold S4M bank-entry sampling constant across mask draws and validation replay.
    rng=torch.random.fork_rng(devices=[device.index or 0]) if name=='S4M' else nullcontext()
    with rng:
        if name=='S4M':torch.manual_seed(2023)
        for index,cpu in enumerate(iter_p4_batches(protocol,split,batch_size=eval_batch),1):
            batch=cpu.to(device)
            with autocast(training,device):
                if stage=='repair':
                    values,mask=t1_inputs(batch);candidate=model(values,mask=mask).float()
                    repaired=torch.where(batch.observed_mask,values,candidate)
                    error=(candidate-t1_clean_history(batch))[~batch.observed_mask]
                else:
                    prediction,repaired=forward(name,model,batch,stamps)
                    if prediction.shape!=batch.forecast_target.shape:raise ValueError('forecast output shape mismatch')
                    error=prediction.float()-batch.forecast_target
            error=error.double();squared+=error.square().sum().item();absolute+=error.abs().sum().item();count+=error.numel()
            if repaired is not None:
                values,_=t1_inputs(batch)
                if not torch.equal(repaired[batch.observed_mask],values[batch.observed_mask]):raise RuntimeError('visible values changed')
                repair_error=(repaired.float()-t1_clean_history(batch)).double();hidden=~batch.observed_mask
                for length in (96,32):
                    part=repair_error[:,-length:][hidden[:,-length:]]
                    repair['sq'+str(length)]+=part.square().sum().item();repair['abs'+str(length)]+=part.abs().sum().item();repair['n'+str(length)]+=part.numel()
            windows+=len(batch.refs)
            if limit and index>=limit:break
    if not count:raise ValueError('empty evaluation')
    result=dict(mse=squared/count,mae=absolute/count,windows=windows,target_elements=count)
    for length in (96,32):
        n=repair['n'+str(length)]
        result.update({f'repair{length}_mse':repair['sq'+str(length)]/n if n else None,
                       f'repair{length}_mae':repair['abs'+str(length)]/n if n else None,
                       f'hidden{length}_coordinates':n})
    return result


def identity(args,cfg,protocol):
    base=author_source_identity(args.profiles,cfg,protocol)
    base['protocol_base_seed']=base.pop('training_seed')
    return {**base,
            'dataset_identity':'Weather','dataset_manifest_sha256':file_sha256(args.dataset_manifest),
            'weather_adapter_sha256':file_sha256(Path(__file__)),
            'weather_model_adapter_sha256':file_sha256(ROOT/'integrations/weather_baselines/models.py'),
            'profile_seed':cfg['seed'],'stage2_seed':2023,'evaluation_drop_last':False}


def extra_state(model):return model.memory_state() if isinstance(model,S4MWrapper) else None
def load_extra(model,state):
    if state is not None:model.restore_memory(state)


def construct(name,cfg,profiles,protocol,device,stage,parent=None):
    if stage=='forecast' and cfg['kind']=='imputer':
        seed(2023);head=build_head(protocol).to(device)
        expected='fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561'
        initial=parameter_hash(head)
        if initial!=expected:raise ValueError('shared head initialization differs from E24-v79')
        imputer=build_author(name,cfg,protocol,device)
        if parent:
            selected=json.loads(Path(parent).read_text())
            path=Path(selected['checkpoint'])
            if file_sha256(path)!=selected['checkpoint_sha256']:raise ValueError('Stage1 checkpoint changed')
            payload=torch.load(path,map_location=device,weights_only=False)
            imputer.load_state_dict(payload['model_state'],strict=True)
        model=frozen_forecaster(imputer,head).to(device);freeze_hook(model)
        return model,initial
    seed(cfg['seed']);return build_author(name,cfg,protocol,device),None


def train(args,cfg,profiles,protocol,device,ident):
    training=profiles['shared_forecast_stage'] if args.stage=='forecast' and cfg['kind']=='imputer' else cfg['training']
    protocol=replace(protocol,training_seed=2023 if args.stage=='forecast' and cfg['kind']=='imputer' else cfg['seed'])
    model,head_sha=construct(args.model,cfg,profiles,protocol,device,args.stage,args.parent)
    if cfg['kind']=='imputer' and args.stage=='forecast':
        params=[p for p in model.predictor.parameters() if p.requires_grad]
        assert all(not p.requires_grad for p in model.imputer.parameters())
        initial_repair=parameter_hash(model.imputer)
    else:params=[p for p in model.parameters() if p.requires_grad];initial_repair=None
    if training.get('optimizer','adam')=='sgd':
        optimizer=torch.optim.SGD(params,lr=training['learning_rate'],weight_decay=training['weight_decay'],momentum=training['momentum'])
    else:
        optimizer=torch.optim.Adam(params,lr=training['learning_rate'],weight_decay=training['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=training['precision']=='float16')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    selected_path=output/'selection.json';checkpoint=output/'selected.pt';latest=output/'latest.pt'
    if selected_path.exists():raise FileExistsError('finalized stage exists')
    best=mse_best=float('inf');best_epoch=mse_epoch=start_epoch=stale=updates=0
    best_validation=mse_validation=None
    if latest.exists():
        if not args.resume:raise FileExistsError('partial stage requires --resume')
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['identity']!=ident:raise ValueError('resume source/data identity changed')
        model.load_state_dict(saved['model_state']);load_extra(model,saved.get('extra_state'))
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        for key in ('best','mse_best','best_epoch','mse_epoch','stale','updates','best_validation','mse_validation'):
            locals_value=saved[key]
            if key=='best':best=locals_value
            elif key=='mse_best':mse_best=locals_value
            elif key=='best_epoch':best_epoch=locals_value
            elif key=='mse_epoch':mse_epoch=locals_value
            elif key=='stale':stale=locals_value
            elif key=='updates':updates=locals_value
            elif key=='best_validation':best_validation=locals_value
            else:mse_validation=locals_value
        start_epoch=saved['epochs_completed']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    stamps=load_timestamp_features(Path(protocol.config.data.datasets['Weather'].path)) if args.model=='CoIFNet' else None
    dual=args.stage=='forecast' and cfg['kind']=='imputer'
    def save_choice(path,epoch,validation):
        _atomic_torch_save(dict(model_state=model.state_dict(),extra_state=extra_state(model),
                               best_epoch=epoch,validation=validation,identity=ident),path)
    completed=start_epoch;total=len(protocol.refs['train'])//training['batch_size']
    end_epoch=start_epoch if training['patience'] is not None and stale>=training['patience'] else training['epochs']
    for epoch in range(start_epoch,end_epoch):
        model.train();losses=[];total=len(protocol.refs['train'])//training['batch_size']
        if initial_repair and (model.imputer.training or any(p.requires_grad for p in model.imputer.parameters())):raise RuntimeError('imputer is not frozen')
        for index,cpu in enumerate(iter_p4_batches(protocol,'train',batch_size=training['batch_size'],epoch=epoch),1):
            batch=cpu.to(device)
            if isinstance(model,S4MWrapper) and index==1:model.warm(batch);continue
            optimizer.zero_grad(set_to_none=True)
            with autocast(training,device):loss=objective(args.model,model,batch,stamps,cfg,args.stage)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();updates+=1
            if initial_repair and any(p.grad is not None for p in model.imputer.parameters()):raise RuntimeError('repair gradients appeared')
            losses.append(loss.item())
            if index==1 or index%100==0 or index==total:
                print(json.dumps(dict(stage=args.stage,epoch=epoch+1,batch=index,batches=total,loss=loss.item(),optimizer_updates=updates)),flush=True)
            if args.smoke:break
        validation=score(args.model,model,protocol,'validation',device,stamps,training,args.stage,limit=1 if args.smoke else None)
        if not args.smoke and validation['windows']!=5175:raise RuntimeError('validation windows dropped')
        if initial_repair and parameter_hash(model.imputer)!=initial_repair:raise RuntimeError('frozen repair changed')
        if validation[training['selection_metric']]<best:
            best=validation[training['selection_metric']];best_epoch=epoch+1;best_validation=validation;stale=0
            save_choice(checkpoint,epoch+1,validation)
        else:stale+=1
        if dual and validation['mse']<mse_best:
            mse_best=validation['mse'];mse_epoch=epoch+1;mse_validation=validation
            save_choice(output/'selected-mse.pt',epoch+1,validation)
        completed=epoch+1
        for group in optimizer.param_groups:group['lr']=epoch_end_learning_rate(training,completed)
        _atomic_torch_save(dict(model_state=model.state_dict(),extra_state=extra_state(model),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),identity=ident,epochs_completed=completed,best=best,mse_best=mse_best,
            best_epoch=best_epoch,mse_epoch=mse_epoch,stale=stale,updates=updates,best_validation=best_validation,
            mse_validation=mse_validation,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),latest)
        print(json.dumps(dict(model=args.model,stage=args.stage,epoch=completed,validation=validation,best_epoch=best_epoch)),flush=True)
        if args.smoke or training['patience'] is not None and stale>=training['patience']:break
    record=dict(**ident,model=args.model,stage=args.stage,profile=cfg,training=training,
        training_seed=protocol.training_seed,epochs_completed=completed,best_epoch=best_epoch,best_validation=best_validation,
        optimizer_updates=updates,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=file_sha256(checkpoint),
        status='smoke' if args.smoke else 'validation_finalized',test_metrics_seen=False,
        predictor_initial_sha256=head_sha,repair_update_policy='frozen_v1' if initial_repair else None,
        repair_initial_sha256=initial_repair,repair_final_sha256=parameter_hash(model.imputer) if initial_repair else None,
        parent_selection=str(args.parent.resolve()) if args.parent else None,
        parent_selection_sha256=file_sha256(args.parent) if args.parent else None,
        actual_training_windows_per_epoch=total*training['batch_size'],actual_training_batches_per_epoch=total,
        s4m_first_batch_warmup_only=isinstance(model,S4MWrapper),stage2_loss='forecast_mse_only' if initial_repair else None)
    _atomic_json(record,selected_path)
    if dual:
        _atomic_json({**record,'best_epoch':mse_epoch,'best_validation':mse_validation,
            'training':{**training,'selection_metric':'mse'},'checkpoint':str((output/'selected-mse.pt').resolve()),
            'checkpoint_sha256':file_sha256(output/'selected-mse.pt'),
            'training_resource_parent_manifest':str(selected_path.resolve()),'training_resource_parent_sha256':file_sha256(selected_path)},output/'selection-mse.json')
    return selected_path


@torch.no_grad()
def evaluate(args,cfg,profiles,protocol,device,ident):
    record=json.loads(args.selection.read_text())
    if record['status']!='validation_finalized' or any(record.get(k)!=v for k,v in ident.items()):raise ValueError('selected identity mismatch')
    path=Path(record['checkpoint'])
    if file_sha256(path)!=record['checkpoint_sha256']:raise ValueError('selected weights changed')
    parent=Path(record['parent_selection']) if record.get('parent_selection') else None
    if parent and file_sha256(parent)!=record['parent_selection_sha256']:raise ValueError('selected Stage1 changed')
    model,head=construct(args.model,cfg,profiles,protocol,device,record['stage'],parent)
    payload=torch.load(path,map_location=device,weights_only=False)
    if payload['identity']!=ident:raise ValueError('payload identity mismatch')
    model.load_state_dict(payload['model_state'],strict=True);load_extra(model,payload.get('extra_state'));model.eval()
    before=parameter_hash(model)
    if parent and parameter_hash(model.imputer)!=record['repair_initial_sha256']:raise ValueError('Stage2 repair differs from parent')
    training=record['training']
    stamps=load_timestamp_features(Path(protocol.config.data.datasets['Weather'].path)) if args.model=='CoIFNet' else None
    validation=score(args.model,model,protocol,'validation',device,stamps,training,'forecast')
    if any(abs(validation[k]-record['best_validation'][k])>1e-6 for k in ('mse','mae')):raise ValueError('validation replay differs')
    manifest=read_manifest(args.dataset_manifest);rows=[]
    for item in [x for x in manifest['masks'] if x['split']=='test']:
        draw=with_test_mask(protocol,manifest,item)
        metrics=score(args.model,model,draw,'test',device,stamps,training,'forecast')
        if metrics['windows']!=10444:raise ValueError('test windows dropped')
        rows.append(dict(seed=item['seed'],mask_sha256=item['sha256'],**metrics))
        print(json.dumps(dict(model=args.model,split='test',**rows[-1])),flush=True)
    if parameter_hash(model)!=before:raise RuntimeError('test changed model state')
    summary={}
    for key in ('mse','mae','repair96_mse','repair96_mae','repair32_mse','repair32_mae'):
        values=[r[key] for r in rows]
        summary[key]=None if any(v is None for v in values) else dict(mean=float(np.mean(values)),mask_std_ddof1=float(np.std(values,ddof=1)))
    resources=selected_training_resources(([parent] if parent else [])+[args.selection])
    result=dict(model=args.model,dataset='Weather',protocol_id=manifest['protocol_id'],start_rate=.1,block_len=8,
        selection=str(args.selection.resolve()),selection_sha256=file_sha256(args.selection),checkpoint_sha256=record['checkpoint_sha256'],
        validation=validation,draws=rows,summary=summary,training_resources=resources,repair_frozen_verified=bool(parent),
        native_forecaster=cfg['kind']!='imputer',stage1_repair_metrics_available=cfg['kind']=='imputer',
        repair_metric_note='frozen Stage1 explicit repair' if parent else 'native history reconstruction' if args.model=='CoIFNet' else 'not applicable; no explicit imputation output',
        training_seed_std=None,drr=None,model_state_unchanged=True,**ident)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'twenty-test-draws.json').exists():raise FileExistsError('evaluation exists')
    _atomic_json(result,args.output/'twenty-test-draws.json')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','evaluate','preflight'])
    parser.add_argument('--model',required=True)
    parser.add_argument('--dataset-manifest',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,required=True)
    parser.add_argument('--stage',choices=['repair','forecast'])
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);device=torch.device(args.device)
    profiles=load_author_profiles(args.profiles);cfg=profiles['models'][args.model]
    manifest=read_manifest(args.dataset_manifest)
    protocol=load_dataset_protocol(args.dataset_manifest,config_path=Path(manifest['config_path']),artifacts=args.dataset_manifest.resolve().parent)
    ident=identity(args,cfg,protocol)
    if args.action=='preflight':
        from .preflight import check
        check(args,cfg,profiles,protocol,device,ident);return
    if args.action=='evaluate':
        if args.selection is None:raise ValueError('evaluation needs selection')
        with TrainingResources(args.output,device=device,batch_size=128,model=args.model,mode='evaluation',started_at=PROCESS_STARTED) as resource:
            evaluate(args,cfg,profiles,protocol,device,ident)
        return
    if args.stage is None:raise ValueError('training needs stage')
    if args.stage=='repair' and cfg['kind']!='imputer':raise ValueError('native forecaster has no Stage1')
    if args.stage=='forecast' and cfg['kind']=='imputer' and args.parent is None:raise ValueError('frozen Stage2 requires finalized Stage1')
    training=profiles['shared_forecast_stage'] if args.stage=='forecast' and cfg['kind']=='imputer' else cfg['training']
    with TrainingResources(args.output,device=device,batch_size=training['batch_size'],model=args.model,
                           mode='smoke' if args.smoke else 'training',started_at=PROCESS_STARTED,
                           precision=training['precision'],amp=training['precision']!='float32') as resources:
        with resources.stage(args.stage):selected=train(args,cfg,profiles,protocol,device,ident)
        resources.selection_manifest=selected


if __name__=='__main__':main()
