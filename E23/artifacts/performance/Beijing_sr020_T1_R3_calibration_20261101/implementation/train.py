"""Full100 epoch/resumable main and single-calibration stages."""
import argparse,fcntl,hashlib,json,time,traceback,os
from pathlib import Path
import numpy as np
import torch
from aligned_h_model import install
install()
from bootstrap import ROOT,verify_sources
from data import load_data
from variants import make_repair,make_calibrator,retrieval_state,restore_retrieval,library_hash
from single import FrozenRepair,SingleBranchModel,apply_head
from support import sha,write_json,save,files_binding,rng_state,restore_rng,digest,recover,folder,artifact_root
from integrations.pamap2_baselines.run import evaluate,resources,seed_all,synchronize
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero

def binding(data,variant,seed,stage,smoke,coarse=None):
    return {**files_binding(),'dataset':data.manifest_sha,'dataset_name':data.dataset,'variant':variant,'seed':seed,'scene':data.scene if hasattr(data,'scene') else None,
        'stage':stage,'epochs':2 if smoke else 100,'batch_size':128,'smoke_batches':smoke or None,'coarse':coarse,'mask_order_seed':2023,'lr':.001}


def load_main(data,scene,seed,variant,device,smoke=False):
    from reuse import load_main_checkpoint
    return load_main_checkpoint(data,scene,seed,variant,device,smoke)

def load_candidate(data,scene,seed,variant,device,metric='mae',smoke=False):
    from reuse import load_model_checkpoint
    return load_model_checkpoint(data,scene,seed,variant,device,metric,smoke)

def export_candidate(path,model,metadata):
    save(path,{'model':model.state_dict(),'retrieval':retrieval_state(model.frozen.repair),**metadata})
def load_export(path,data,device):
    state=torch.load(path,map_location=device,weights_only=False);repair=make_repair(data,state['variant'],state['seed']);head=make_calibrator(data.n_vars,state['variant'],state['seed'])
    result=SingleBranchModel(FrozenRepair(repair),head).to(device);result.load_state_dict(state['model'],strict=True);restore_retrieval(repair,state['retrieval']);repair.epoch=state['coarse_epoch'];return result.eval()

def run(args):
    verify_sources();torch.set_num_threads(4);seed_all(args.seed);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    data=load_data(os.environ['A64_DATASET'],args.scene,artifact_root()/'assets');smoke=args.smoke_batches;epochs=2 if smoke else 100
    updates_per_epoch=smoke if smoke else len(data.starts['train'])//128

    path=folder(args.scene,args.seed,args.variant,args.stage,smoke);path.mkdir(parents=True,exist_ok=True)
    lock=(path/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    coarse=None;before=None;lib=None
    if args.stage=='main':model=make_repair(data,args.variant,args.seed).to(device);optimized=model
    else:
        repair,coarse=load_main(data,args.scene,args.seed,args.variant,device,smoke)
        # Replay selected main before accepting a frozen prefix.
        actual=evaluate(repair,data,'validation',device,batch_size=128,max_batches=1 if smoke and not coarse['reused'] else None)
        for m in ['mse','mae']:
            if abs(actual['full96'][m]-coarse['validation']['full96'][m])>1e-6:raise RuntimeError('main replay differs')
        frozen=FrozenRepair(repair);head=make_calibrator(data.n_vars,args.variant,args.seed).to(device)
        model=SingleBranchModel(frozen,head);optimized=head;before=digest(repair);lib=library_hash(repair)
    bound=binding(data,args.variant,args.seed,args.stage,smoke,None if coarse is None else coarse['sha256'])
    if (path/'binding.json').exists() and json.loads((path/'binding.json').read_text())!=bound:raise RuntimeError('source/data binding differs')
    write_json(path/'binding.json',bound)
    if (path/'complete.json').exists():return
    params=[p for p in optimized.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001,weight_decay=0.)
    # Match the prior private head stream seed; construction and replay must not shift dropout.
    seed_all(args.seed);start=1;updates=0;best={'mse':float('inf'),'mae':float('inf')};best_epoch={'mse':0,'mae':0};prior_seconds=0.;prefix_seconds=0.;update_seconds=0.
    if (path/'latest.pt').exists():
        state=torch.load(path/'latest.pt',map_location=device,weights_only=False)
        if state['binding']!=bound:raise RuntimeError('resume binding differs')
        model.load_state_dict(state['model'],strict=True);restore_retrieval(model if args.stage=='main' else model.frozen.repair,state['retrieval']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        start=state['epoch']+1;updates=state['updates'];best=state['best'];best_epoch=state['best_epoch'];prior_seconds=state['seconds'];prefix_seconds=state['prefix_seconds'];update_seconds=state['update_seconds'];recover(path,state['epoch'])
    write_json(path/'model.json',{'parameters':sum(p.numel() for p in model.parameters()),'stage_trainable':sum(p.numel() for p in params),'variant':args.variant,'main_width':64,'calibration_width':64 if args.stage=='calibration' else None,'tail_refiner':False,'candidate_access':args.variant!='CTX'})
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    began=time.perf_counter()
    try:
        for epoch in range(start,epochs+1):
            eb=time.perf_counter();fingerprint=hashlib.sha256();count=0;total=0.;gradient_names=[]
            if args.stage=='main':
                model.epoch=epoch
                if args.variant!='CTX':model.core.historical.retrieval.begin_epoch(epoch)
                model.train()
            else:model.eval();head.train()
            for truth,seen,starts in data.batches('train',128,epoch=epoch-1,seed=2023,max_batches=smoke):
                fingerprint.update(starts.tobytes());fingerprint.update(seen.numpy().tobytes());truth,seen=truth.to(device),seen.to(device)
                opt.zero_grad(set_to_none=True)
                if args.stage=='main':model.batch_index=count+1;loss=model.loss(truth,seen,starts)
                else:
                    t=time.perf_counter();batch=frozen.extract(torch.where(seen,truth,0),seen,starts);synchronize(device);prefix_seconds+=time.perf_counter()-t
                    t=time.perf_counter();prediction=apply_head(head,batch);valid=torch.from_numpy(data.valid_for(starts)).to(device)
                    loss=mean_or_zero(hidden_error(prediction,truth,seen,valid).abs())+.25*mean_or_zero(hidden_error(prediction[:,-32:],truth[:,-32:],seen[:,-32:],valid[:,-32:]).abs())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward()
                if count==0:
                    for n,p in optimized.named_parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite gradient '+n)
                            if bool(p.grad.abs().sum()>0):gradient_names.append(n)
                opt.step()
                if args.stage=='main':model.after_step()
                else:synchronize(device);update_seconds+=time.perf_counter()-t
                total+=float(loss.detach());count+=1;updates+=1
            if count!=updates_per_epoch:raise RuntimeError('partial epoch')
            if args.stage=='main':model.publish_epoch(epoch)
            model.eval();validation=evaluate(model,data,'validation',device,batch_size=128,max_batches=1 if smoke else None)
            if args.stage=='calibration':
                if any(p.grad is not None or p.requires_grad for p in repair.parameters()) or library_hash(repair)!=lib:raise RuntimeError('frozen repair changed')
            seconds=prior_seconds+time.perf_counter()-began
            for metric in (['mse'] if args.stage=='main' else ['mse','mae']):
                if validation['full96'][metric]<best[metric]:
                    best[metric]=validation['full96'][metric];best_epoch[metric]=epoch
                    state={'epoch':epoch,'validation':validation,'binding':bound,'retrieval':retrieval_state(model if args.stage=='main' else repair),'library_sha256':library_hash(model if args.stage=='main' else repair)}
                    state['model' if args.stage=='main' else 'head']=(model if args.stage=='main' else head).state_dict()
                    save(path/('best.pt' if args.stage=='main' else f'best_{metric}.pt'),state)
            row={'epoch':epoch,'updates_this_epoch':count,'updates_total':updates,'batch_fingerprint':fingerprint.hexdigest(),'train_loss':total/count,'validation':validation,'best_epoch':best_epoch,'epoch_seconds':time.perf_counter()-eb,'seconds':seconds,'nonzero_gradient_parameters_first_batch':gradient_names,'frozen_digest':before,'library_sha256':library_hash(model if args.stage=='main' else repair),'resources':resources(device)}
            write_json(path/'epochs'/f'{epoch:03}.json',row)
            save(path/'latest.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'retrieval':retrieval_state(model if args.stage=='main' else repair),'epoch':epoch,'updates':updates,'best':best,'best_epoch':best_epoch,'binding':bound,'rng':rng_state(),'seconds':seconds,'prefix_seconds':prefix_seconds,'update_seconds':update_seconds})
            recover(path,epoch);write_json(path/'status.json',{'status':'training',**row});print(json.dumps({'stage':args.stage,'variant':args.variant,'seed':args.seed,'epoch':epoch,'validation':validation['full96'],'seconds':row['epoch_seconds']}),flush=True)
        if updates!=epochs*updates_per_epoch:raise RuntimeError('incomplete training budget')
        if args.stage=='calibration' and digest(repair)!=before:raise RuntimeError('frozen prefix mutated')
        checkpoints={'mse':sha(path/'best.pt')} if args.stage=='main' else {m:sha(path/f'best_{m}.pt') for m in ['mse','mae']}
        done={'status':'smoke_complete' if smoke else 'complete','epochs':epochs,'updates':updates,'updates_per_epoch':updates_per_epoch,'seconds':prior_seconds+time.perf_counter()-began,'binding':bound,'best_epoch':best_epoch,'checkpoints':checkpoints,'resources':resources(device),'frozen_before':before,'frozen_after':None if before is None else digest(repair),'prefix_seconds':prefix_seconds,'update_seconds':update_seconds,'coarse':coarse}
        write_json(path/'complete.json',done);write_json(path/'status.json',done)
    except BaseException as e:
        with (path/'failures.jsonl').open('a') as f:f.write(json.dumps({'error':repr(e),'traceback':traceback.format_exc()})+'\n')
        write_json(path/'status.json',{'status':'failed','error':repr(e)});raise

def main():
    p=argparse.ArgumentParser(description='Portable final R3: full100 main + frozen single64 calibration')
    p.add_argument('--dataset',choices=['ETTh1','Weather','Exchange-Rate','PAMAP2','BeijingAirQuality','PhysioNet2012'],default='ETTh1')
    p.add_argument('--data-root',type=Path,required=True,help='directory containing data/raw and/or data/processed')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--variant',choices=['R3','H_GLOBAL'],default='R3')
    p.add_argument('--seed',type=int,default=2023);p.add_argument('--scene',default='sr015_bl8')
    p.add_argument('--stage',choices=['main','calibration','both'],default='both')
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-batches',type=int,default=None)
    args=p.parse_args();os.environ['A64_DATASET']=args.dataset;os.environ['R2_DATA_ROOT']=str(args.data_root.resolve());os.environ['R2_OUTPUT_ROOT']=str(args.output_root.resolve())
    if args.variant=='R3':args.variant='H_GLOBAL'
    for stage in (['main','calibration'] if args.stage=='both' else [args.stage]):
        args.stage=stage;run(args)
if __name__=='__main__':main()
