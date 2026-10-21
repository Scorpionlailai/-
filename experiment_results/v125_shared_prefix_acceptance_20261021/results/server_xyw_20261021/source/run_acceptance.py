"""NEW_PAMAP2_FINITE_ACCEPTANCE: A0 seed2023,2epochs/branch,850total updates."""
import argparse
from contextlib import ExitStack
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
import torch

PKG=Path('/home/xyw/paper-v125-shared-prefix-acceptance/experiment_results/v125_shared_prefix_acceptance_20261021')
CAL=Path('/home/xyw/paper/E24/artifacts/v125_calibration_confirmation_20261020')
DIAG=Path('/home/xyw/paper/E24/artifacts/v125_contribution_diagnostics_20261019')
sys.path.insert(0,str(CAL/'tools'))
from policy import initialize_branch,frozen_hash,tensor_hash
from integrations.v125_ablation.common import (STUDY,verify_lock,data_for,file_sha256,write_json,seed_all,
    atomic_torch,rng_state,restore_rng,evaluate,resources,synchronize)
from integrations.v125_ablation.adapter import V125Adapter
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero
from acceptance_utils import tree_hash,stream_call,finish_calibration,compare_trees
spec=importlib.util.spec_from_file_location('provided_capture',PKG/'code/local_worktree/integrations/e23/shared_frozen_calibration.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

EVIDENCE='NEW_PAMAP2_FINITE_ACCEPTANCE'

def write_csv(path,rows,template=None):
    fields=[]
    if template:fields=(PKG/'templates'/template).read_text().splitlines()[0].split(',')
    fields=list(dict.fromkeys(fields+[k for row in rows for k in row]))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def cpu_clone(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def build_model(data,device,source):
    seed_all(2023);m=V125Adapter(data,'A0',2023).to(device)
    assert file_sha256(Path(source['coarse_checkpoint']))==source['coarse_sha256']
    m.load_state_dict(torch.load(source['coarse_checkpoint'],map_location=device,weights_only=True))
    for i in (0,1):initialize_branch(m,i,2023)
    m.requires_grad_(False).eval();m.epoch=100;m.batch_index=0
    for member in m.core.recent_mean_projector.members:member.requires_grad_(True).train()
    assert m.memory.snapshot_hash()==source['library_sha256'] and frozen_hash(m.state_dict())==source['coarse_frozen_prefix_sha256']
    return m


def capture(model,visible,observed,starts):
    before=tree_hash(rng_state());norms=[];model.inference_path='ensemble'
    def prefix():
        result,norm,positions=model.output(visible,observed,starts);norms.append(norm);return result
    args=helper.capture_frozen_calibration_inputs(model.core,prefix)
    assert tree_hash(rng_state())==before,'frozen prefix consumed Python/NumPy/Torch RNG'
    assert len(norms)==1 and torch.equal(args[0].sequence_observed,observed)
    assert torch.equal(args[0].normalized_sequence[observed],args[0].observed_sequence[observed])
    return args,norms[0]


def shared_output(model,index,batch,context):
    truth,observed,starts,valid=batch;args,norm=context
    explicit=model.core.recent_mean_projector.members[index](*args)[0]
    assert torch.equal(explicit.normalized_sequence[observed],explicit.observed_sequence[observed])
    visible=torch.where(observed,truth,0.)
    return finish_calibration(explicit,norm,visible,observed,model.core.repair_input_scaling)


def no_mutation(*args,**kwargs):raise AssertionError('frozen prototype/key/EMA mutation attempted')


def metric_rows(predictions,truth,observed,valid,aggregate,execution,branch,epoch,selected,checkpoint_sha):
    error=predictions-truth;hidden=valid & ~observed;n,t,c=truth.shape;rows=[]
    def add(mask,band='full96',gap='all',variable='all',primary=None):
        e=error[mask].astype(np.float64);count=len(e)
        stats=dict(valid_hidden_count=count,squared_error_sum=float(np.square(e).sum()) if count else None,
            absolute_error_sum=float(np.abs(e).sum()) if count else None,mse=float(np.square(e).mean()) if count else None,mae=float(np.abs(e).mean()) if count else None)
        if primary:stats.update(valid_hidden_count=primary['coordinates'],**{k:primary[k] for k in ('squared_error_sum','absolute_error_sum','mse','mae')})
        rows.append(dict(evidence_class=EVIDENCE,execution=execution,arm='A0',training_seed=2023,epoch=epoch,branch=branch,split='validation',mask_seed=2021,
            band=band,gap_bin=gap,variable=variable,windows=int(mask.any((1,2)).sum()),**stats,selected_epoch=selected,checkpoint_sha256=checkpoint_sha,
            empty_group=count==0,scale='training_global_standardized',native_missing_not_scored=True))
    add(hidden,primary=aggregate['full96'])
    for start,end,label in ((0,32,'first32'),(32,64,'middle32'),(64,96,'last32')):
        mask=hidden.copy();mask[:,:start]=False;mask[:,end:]=False;add(mask,band=label,primary=aggregate['recent32'] if start==64 else None)
    # Actual invisibility includes native missingness and is truncated at window boundaries.
    sys.path.insert(0,str(DIAG/'tools'));import evaluate as diag
    lengths=diag.runs(observed)
    for lo,hi in ((1,8),(9,16),(17,32),(33,96)):add(hidden & (lengths>=lo) & (lengths<=hi),gap=f'{lo}-{hi}')
    for variable in range(c):
        mask=np.zeros_like(hidden);mask[:,:,variable]=hidden[:,:,variable];add(mask,variable=str(variable))
    return rows


class Arm:
    def __init__(self,name,data,device,source,out):
        self.name=name;self.data=data;self.device=device;self.source=source;self.folder=out/name;self.folder.mkdir()
        self.start=time.perf_counter();self.timings=dict(prepare_seconds=0.,train_update_seconds=0.,validation_seconds=0.,save_seconds=0.)
        tick=time.perf_counter();self.model=build_model(data,device,source);self.members=self.model.core.recent_mean_projector.members
        self.opts=[torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0.) for m in self.members]
        self.streams=[]
        for _ in (0,1):seed_all(2023);self.streams.append(rng_state())
        self.steps=[0,0];self.new_updates=[0,0];self.best=[float('inf'),float('inf')];self.selected=[None,None];self.best_states=[None,None]
        self.histories=[[],[]];self.epoch_checks=[];self.metrics=[];self.main_gradients=[None,None];self.cursors=[0,0]
        self.category='prepare';self.calls={};self.hook=self.model.core.front_context.register_forward_pre_hook(self.count)
        self.frozen=frozen_hash(self.model.state_dict());self.library=self.model.memory.snapshot_hash();self.retrieval=tree_hash(self.model.core.historical.retrieval.state_dict())
        self.timings['prepare_seconds']+=time.perf_counter()-tick
        self.source_lock_sha=None;self.prefix_inputs=[];self.train_peak_allocated=0.;self.train_peak_reserved=0.;self.last_progress=time.perf_counter()
    def count(self,*args):self.calls[self.category]=self.calls.get(self.category,0)+1
    def guards(self):
        stack=ExitStack()
        for name in ('stage_patch','rekey','rekey_batch','publish_epoch'):stack.enter_context(patch.object(self.model.memory,name,side_effect=no_mutation))
        stack.enter_context(patch.object(self.model,'after_step',side_effect=no_mutation));stack.enter_context(patch.object(self.model,'publish_epoch',side_effect=no_mutation))
        stack.enter_context(patch.object(self.model.core.historical.retrieval,'begin_epoch',side_effect=no_mutation))
        return stack
    def check_frozen(self):
        assert frozen_hash(self.model.state_dict())==self.frozen
        assert self.model.memory.snapshot_hash()==self.library
        assert tree_hash(self.model.core.historical.retrieval.state_dict())==self.retrieval
        assert not any(p.grad is not None for p in self.model.parameters() if not p.requires_grad)
    def fetch(self,iterator,index):
        tick=time.perf_counter();truth,observed,starts=stream_call(self.streams[index],lambda:next(iterator))
        valid=self.data.valid_for(starts)
        result=(truth.to(self.device),observed.to(self.device),starts,torch.from_numpy(valid).to(self.device))
        synchronize(self.device);self.timings['prepare_seconds']+=time.perf_counter()-tick
        return result
    def update(self,index,batch,context=None):
        truth,observed,starts,valid=batch;self.model.batch_index=self.steps[index]%85+1
        self.model.eval()
        if context is not None:
            for member in self.members:member.train()
        else:self.members[index].train()
        self.opts[index].zero_grad(set_to_none=True)
        if context is None:
            self.model.inference_path='calibration'+str((64,256)[index])
            candidate=self.model(torch.where(observed,truth,0.),observed,starts)
        else:candidate=shared_output(self.model,index,batch,context)
        assert torch.equal(candidate[observed],truth[observed])
        main=mean_or_zero(hidden_error(candidate,truth,observed,valid).abs())
        tail=mean_or_zero(hidden_error(candidate[:,-32:],truth[:,-32:],observed[:,-32:],valid[:,-32:]).abs())
        loss=main+.25*tail;assert torch.isfinite(loss)
        if self.main_gradients[index] is None:
            gradients=torch.autograd.grad(main,tuple(self.members[index].parameters()),allow_unused=True,retain_graph=True)
            finite=all(g is None or torch.isfinite(g).all() for g in gradients)
            norm=sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5
            assert finite and np.isfinite(norm) and norm>0
            self.main_gradients[index]=dict(norm=norm,finite=True,tensors=[None if g is None else g.detach().cpu().clone() for g in gradients])
        loss.backward();assert not any(p.grad is not None for p in self.model.parameters() if not p.requires_grad)
        self.opts[index].step();self.opts[index].zero_grad(set_to_none=True)
        self.steps[index]+=1;self.new_updates[index]+=1
        self.histories[index].append(dict(main=float(main.detach()),loss=float(loss.detach())))
        if self.new_updates[index]%25==0 or time.perf_counter()-self.last_progress>=240:
            synchronize(self.device)
            progress=dict(phase=self.name,branch=(64,256)[index],branch_updates=self.steps[index],new_updates=self.new_updates,
                branch_steps=self.steps,epoch_cursor=self.cursors,batch_cursor=self.steps[index]%85,
                elapsed_arm_seconds=time.perf_counter()-self.start,resources=resources(self.device),pid=os.getpid())
            write_json(self.folder.parent/'status.json',progress)
            print(json.dumps(dict(event='bounded_training_progress',**progress)),flush=True);self.last_progress=time.perf_counter()
    def validation(self,index,epoch):
        self.category='validation';self.model.inference_path='calibration'+str((64,256)[index]);parts=[]
        def collect(module,args,output):
            values,observed,starts=args
            parts.append((output.detach().cpu().numpy(),np.stack([self.data.values[s:s+96] for s in starts]),observed.detach().cpu().numpy(),self.data.valid_for(starts)))
        h=self.model.register_forward_hook(collect);tick=time.perf_counter()
        try:result=stream_call(self.streams[index],lambda:evaluate(self.model,self.data,'validation',self.device,batch_size=128))
        finally:h.remove()
        synchronize(self.device);self.timings['validation_seconds']+=time.perf_counter()-tick
        assert result['windows']==1306 and result['mask_seed']==2021
        if epoch>0 and result['full96']['mae']<self.best[index]:
            self.best[index]=result['full96']['mae'];self.selected[index]=epoch;self.best_states[index]=cpu_clone(self.members[index].state_dict())
        checkpoint=self.folder/f'branch{(64,256)[index]}_epoch{epoch}.pt';tick=time.perf_counter();atomic_torch(checkpoint,self.members[index].state_dict());self.timings['save_seconds']+=time.perf_counter()-tick
        pred,truth,obs,valid=[np.concatenate([p[i] for p in parts]) for i in range(4)]
        np.save(self.folder/f'branch{(64,256)[index]}_epoch{epoch}_validation.npy',pred,allow_pickle=False)
        self.metrics.extend(metric_rows(pred,truth,obs,valid,result,self.name,str((64,256)[index]),epoch,self.selected[index],file_sha256(checkpoint)))
        self.check_frozen()
        if epoch==0:
            self.epoch_checks.append(dict(execution=self.name,branch=(64,256)[index],epoch=0,steps=0,
                train_order_sha256=None,train_mask_sha256=None,main_gradient_norm=None,main_gradient_finite=None,
                frozen_tensor_sha256=self.frozen,library_hash=self.library,retrieval_hash=self.retrieval,
                loss_history_sha256=tree_hash(self.histories[index]),model_tensor_hash=tree_hash(self.members[index].state_dict()),
                optimizer_hash=tree_hash(self.opts[index].state_dict()),rng_hash=tree_hash(self.streams[index]),validation_mae=result['full96']['mae'],
                validation=result,best_mae=None,selected_epoch=None,checkpoint_sha256=file_sha256(checkpoint),
                explanation='epoch0diagnostic only;no updates/selection/gradient probe'))
        return result,checkpoint
    def epoch_end(self,index,epoch,order,masks):
        validation,checkpoint=self.validation(index,epoch);self.cursors[index]=epoch
        grad=self.main_gradients[index]
        row=dict(execution=self.name,branch=(64,256)[index],epoch=epoch,steps=self.steps[index],train_order_sha256=order.hexdigest(),train_mask_sha256=masks.hexdigest(),
            main_gradient_norm=grad['norm'] if grad else None,main_gradient_finite=grad['finite'] if grad else None,
            frozen_tensor_sha256=self.frozen,library_hash=self.library,retrieval_hash=self.retrieval,
            loss_history_sha256=tree_hash(self.histories[index]),model_tensor_hash=tree_hash(self.members[index].state_dict()),
            optimizer_hash=tree_hash(self.opts[index].state_dict()),rng_hash=tree_hash(self.streams[index]),validation_mae=validation['full96']['mae'],
            validation=validation,best_mae=self.best[index],selected_epoch=self.selected[index],checkpoint_sha256=file_sha256(checkpoint))
        self.epoch_checks.append(row);write_json(self.folder/f'branch{(64,256)[index]}_epoch{epoch}_audit.json',row)
        print(json.dumps(dict(event='epoch_completed',execution=self.name,branch=row['branch'],epoch=epoch,steps=row['steps'],validation=validation,resources=resources(self.device))),flush=True)
    def save(self,label):
        self.check_frozen();payload=dict(model=self.model.state_dict(),optimizers=[o.state_dict() for o in self.opts],streams=self.streams,
            ambient_rng=rng_state(),retrieval_ema=self.model.core.historical.retrieval.state_dict(),source=self.source,
            epoch_cursor=self.cursors,batch_cursor=0,steps=self.steps,best=self.best,selected=self.selected,best_states=self.best_states,
            histories=self.histories,epoch_checks=self.epoch_checks,main_gradients=self.main_gradients,model_epoch=self.model.epoch,model_batch_index=self.model.batch_index,
            library_hash=self.library,frozen_hash=self.frozen,retrieval_hash=self.retrieval,
            source_lock_sha256=file_sha256(self.folder.parent/'source_lock.json'))
        tick=time.perf_counter();path=self.folder/(label+'.pt');atomic_torch(path,payload);self.timings['save_seconds']+=time.perf_counter()-tick
        return path
    def load_resume(self,path):
        tick=time.perf_counter();p=torch.load(path,map_location=self.device,weights_only=False);assert p['source']==self.source and p['batch_cursor']==0
        self.model.load_state_dict(p['model']);self.model.core.historical.retrieval.load_state_dict(p['retrieval_ema'])
        for o,state in zip(self.opts,p['optimizers']):o.load_state_dict(state)
        self.streams=cpu_clone(p['streams']);self.cursors=p['epoch_cursor'];self.steps=p['steps'];self.best=p['best'];self.selected=p['selected'];self.best_states=cpu_clone(p['best_states'])
        self.histories=p['histories'];self.epoch_checks=[{**x,'execution':self.name,'inherited_from_checkpoint':True} for x in p['epoch_checks']];self.main_gradients=cpu_clone(p['main_gradients'])
        self.model.epoch=p['model_epoch'];self.model.batch_index=p['model_batch_index'];restore_rng(p['ambient_rng']);self.check_frozen()
        self.timings['prepare_seconds']+=time.perf_counter()-tick;del p
    def finish(self):
        path=self.save('final_state');peak=resources(self.device)
        peak['max_allocated_gib']=self.train_peak_allocated/2**30;peak['max_reserved_gib']=self.train_peak_reserved/2**30
        result=dict(execution=self.name,timings=self.timings,steps=self.steps,new_updates=self.new_updates,best=self.best,selected=self.selected,
            histories=self.histories,epoch_checks=self.epoch_checks,metrics=self.metrics,checkpoint=str(path),checkpoint_sha256=file_sha256(path),
            resources=peak,actual_prefix_calls=self.calls.get('train',0),prefix_calls_by_phase=self.calls,
            model_tensor_hash=tree_hash(self.model.state_dict()),optimizer_hashes=[tree_hash(o.state_dict()) for o in self.opts],rng_hashes=[tree_hash(s) for s in self.streams],
            best_state_hashes=[tree_hash(s) for s in self.best_states],frozen_tensor_hash=self.frozen,library_hash=self.library,retrieval_hash=self.retrieval,
            total_parameters=sum(p.numel() for p in self.model.parameters()),trainable_parameters=sum(p.numel() for m in self.members for p in m.parameters()),
            total_wall_seconds=time.perf_counter()-self.start)
        write_json(self.folder/'result.json',result);self.hook.remove()
        return result
    def release(self):
        del self.model,self.members,self.opts;gc.collect();torch.cuda.empty_cache()


def train_reference(arm):
    for i in (0,1):
        arm.validation(i,0);arm.save(f'branch{(64,256)[i]}_epoch0_resume')
        for epoch in (1,2):
            iterator=stream_call(arm.streams[i],lambda:iter(arm.data.batches('train',128,epoch=epoch-1,seed=2023)))
            order=hashlib.sha256();mask=hashlib.sha256()
            for batch_index in range(85):
                batch=arm.fetch(iterator,i);order.update(batch[2].tobytes());mask.update(batch[1].cpu().numpy().tobytes())
                arm.category='train';synchronize(arm.device);torch.cuda.reset_peak_memory_stats(arm.device);tick=time.perf_counter()
                stream_call(arm.streams[i],lambda:arm.update(i,batch));synchronize(arm.device);arm.timings['train_update_seconds']+=time.perf_counter()-tick
                arm.train_peak_allocated=max(arm.train_peak_allocated,torch.cuda.max_memory_allocated(arm.device));arm.train_peak_reserved=max(arm.train_peak_reserved,torch.cuda.max_memory_reserved(arm.device))
            try:stream_call(arm.streams[i],lambda:next(iterator))
            except StopIteration:pass
            else:raise AssertionError('more than85training batches')
            arm.epoch_end(i,epoch,order,mask);arm.save(f'branch{(64,256)[i]}_epoch{epoch}_resume')


def train_shared(arm,epochs):
    if epochs[0]==1:
        for i in (0,1):arm.validation(i,0)
        arm.save('epoch0')
    for epoch in epochs:
        iterators=[stream_call(arm.streams[i],lambda:iter(arm.data.batches('train',128,epoch=epoch-1,seed=2023))) for i in (0,1)]
        orders=[hashlib.sha256(),hashlib.sha256()];masks=[hashlib.sha256(),hashlib.sha256()]
        for b in range(85):
            batches=[arm.fetch(it,i) for i,it in enumerate(iterators)]
            assert np.array_equal(batches[0][2],batches[1][2]) and torch.equal(batches[0][0],batches[1][0]) and torch.equal(batches[0][1],batches[1][1])
            for i in (0,1):orders[i].update(batches[i][2].tobytes());masks[i].update(batches[i][1].cpu().numpy().tobytes())
            arm.model.eval()
            for member in arm.members:member.train()
            arm.category='train';synchronize(arm.device);torch.cuda.reset_peak_memory_stats(arm.device);tick=time.perf_counter()
            arm.model.batch_index=b+1
            truth,observed,starts,valid=batches[0];context=capture(arm.model,torch.where(observed,truth,0.),observed,starts)
            for i in (0,1):stream_call(arm.streams[i],lambda:arm.update(i,batches[i],context))
            del context;synchronize(arm.device);arm.timings['train_update_seconds']+=time.perf_counter()-tick
            arm.train_peak_allocated=max(arm.train_peak_allocated,torch.cuda.max_memory_allocated(arm.device));arm.train_peak_reserved=max(arm.train_peak_reserved,torch.cuda.max_memory_reserved(arm.device))
        for i,it in enumerate(iterators):
            try:stream_call(arm.streams[i],lambda:next(it))
            except StopIteration:pass
            else:raise AssertionError('more than85training batches')
            arm.epoch_end(i,epoch,orders[i],masks[i])
        arm.save(f'epoch{epoch}')


def first_batch_preflight(data,device,source,out):
    torch.cuda.reset_peak_memory_stats(device)
    tick=time.perf_counter();model=build_model(data,device,source);truth,observed,starts=next(data.batches('train',128,epoch=0,seed=2023))
    truth,observed=truth.to(device),observed.to(device);valid=torch.from_numpy(data.valid_for(starts)).to(device);batch=(truth,observed,starts,valid)
    model.batch_index=1;context=capture(model,torch.where(observed,truth,0.),observed,starts);checks=[]
    for i in (0,1):
        seed_all(2023);state=rng_state();candidate_state=cpu_clone(state)
        model.inference_path='calibration'+str((64,256)[i]);params=tuple(model.core.recent_mean_projector.members[i].parameters())
        def ref():
            pred=model(torch.where(observed,truth,0.),observed,starts);main=mean_or_zero(hidden_error(pred,truth,observed,valid).abs())
            grads=torch.autograd.grad(main,params,allow_unused=True);return pred.detach(),[None if g is None else g.detach() for g in grads]
        reference=stream_call(state,ref)
        def shared():
            pred=shared_output(model,i,batch,context);main=mean_or_zero(hidden_error(pred,truth,observed,valid).abs())
            grads=torch.autograd.grad(main,params,allow_unused=True);return pred.detach(),[None if g is None else g.detach() for g in grads]
        candidate=stream_call(candidate_state,shared)
        comparison=compare_trees(reference,candidate)
        if not comparison['exact_equal'] or tree_hash(state)!=tree_hash(candidate_state):
            write_json(out/'single_batch_preflight_failure.json',dict(branch=(64,256)[i],comparison=comparison,
                reference_rng_sha256=tree_hash(state),shared_rng_sha256=tree_hash(candidate_state),budget_updates=0,
                reference_output_sha256=tree_hash(reference[0]),shared_output_sha256=tree_hash(candidate[0]),
                reference_gradient_sha256=tree_hash(reference[1]),shared_gradient_sha256=tree_hash(candidate[1])))
            atomic_torch(out/'single_batch_failure_detail.pt',dict(reference=reference,shared=candidate))
        assert comparison['exact_equal'],comparison
        assert tree_hash(state)==tree_hash(candidate_state)
        gradients=candidate[1];norm=sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
        assert norm>0 and np.isfinite(norm) and all(g is None or torch.isfinite(g).all() for g in gradients)
        assert torch.equal(candidate[0][observed],truth[observed])
        checks.append(dict(branch=(64,256)[i],output_and_main_MAE_gradients_exact=True,max_abs_difference=0.,main_gradient_norm=norm,finite=True,
            gradient_tensor_hash=tree_hash(gradients),independent_rng_exact=True))
    assert frozen_hash(model.state_dict())==source['coarse_frozen_prefix_sha256'] and model.memory.snapshot_hash()==source['library_sha256']
    write_json(out/'single_batch_preflight.json',dict(passed=True,batch_size=128,checks=checks,seconds=time.perf_counter()-tick,
        source_scale_wrapper_operation_order_preserved=True,optimizer_updates=0,resources=resources(device)))
    del model,context,reference,candidate;gc.collect();torch.cuda.empty_cache()
    return checks


def inference_profile(data,device,source,checkpoint,execution,out):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=build_model(data,device,source);model.load_state_dict(saved['model'])
    for i in (0,1):model.core.recent_mean_projector.members[i].load_state_dict(saved['best_states'][i])
    model.requires_grad_(False).eval();truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device);visible=torch.where(obs,truth,0.)
    # Selected coarse is independent of the diagnostic calibrators and remains exactly its parent source.
    before=frozen_hash(model.state_dict());profiles=[];model.zero_grad(set_to_none=True)
    for path,label in [('coarse','coarse'),('calibration64','single64'),('calibration256','single256'),('ensemble','dual')]:
        model.inference_path=path;calls=[0];h=model.core.front_context.register_forward_pre_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(5):pred=model(visible,obs,starts);assert torch.equal(pred[obs],truth[obs]);synchronize(device)
            times=[]
            for _ in range(20):
                synchronize(device);tick=time.perf_counter();pred=model(visible,obs,starts);synchronize(device);times.append(time.perf_counter()-tick)
        h.remove();assert calls[0]==25
        profiles.append(dict(path=label,batch_size=128,warmups=5,repeats=20,seconds=times,prefix_calls=calls[0],
            median_ms_per_window=float(np.median(times))*1000/128,p95_ms_per_window=float(np.percentile(times,95))*1000/128,
            windows_per_second=128/float(np.median(times)),peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30,parameters=sum(p.numel() for p in model.parameters()),input_already_on_GPU=True,gpu_shared=False))
    assert frozen_hash(model.state_dict())==before
    # Full validation with diagnostic selected dual and unchanged parent coarse. Reporting only, not selection.
    metrics=[]
    for path,label in [('coarse','coarse'),('ensemble','dual')]:
        model.inference_path=path;parts=[]
        def collect(module,args,output):
            _,observed,starts=args;parts.append((output.cpu().numpy(),np.stack([data.values[s:s+96] for s in starts]),observed.cpu().numpy(),data.valid_for(starts)))
        h=model.register_forward_hook(collect)
        with torch.no_grad():aggregate=evaluate(model,data,'validation',device,batch_size=128)
        h.remove();arrays=[np.concatenate([p[i] for p in parts]) for i in range(4)]
        metrics+=metric_rows(*arrays,aggregate,execution,label,2,saved['selected'] if label=='dual' else 'parent_coarse96',file_sha256(checkpoint))
    write_json(out/execution/'inference.json',profiles);write_json(out/execution/'selected_dual_and_coarse_metrics.json',metrics)
    del model,saved;gc.collect();torch.cuda.empty_cache()
    return profiles,metrics


def compare_executions(out):
    reference=torch.load(out/'reference/final_state.pt',map_location='cpu',weights_only=False)
    shared=torch.load(out/'shared/final_state.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(out/'resume/final_state.pt',map_location='cpu',weights_only=False)
    checks=[]
    for name,candidate in [('reference_vs_shared',shared),('shared_vs_resume',resumed)]:
        expected=reference if name=='reference_vs_shared' else shared
        for field in ('model','optimizers','streams','retrieval_ema','steps','epoch_cursor','batch_cursor','best','selected','best_states','histories','main_gradients'):
            comparison=compare_trees(expected[field],candidate[field])
            checks.append(dict(comparison=name,field=field,reference_value=tree_hash(expected[field]),candidate_value=tree_hash(candidate[field]),
                exact_equal=comparison['exact_equal'],max_abs_difference=comparison['max_abs_difference'],passed=comparison['exact_equal'],explanation=comparison['failures']))
        reference_epochs={(x['branch'],x['epoch']):x for x in expected['epoch_checks']};candidate_epochs={(x['branch'],x['epoch']):x for x in candidate['epoch_checks']}
        for identity,row in reference_epochs.items():
            other=candidate_epochs[identity]
            for field in ('train_order_sha256','train_mask_sha256','loss_history_sha256','model_tensor_hash','optimizer_hash','rng_hash','validation','selected_epoch','best_mae'):
                c=compare_trees(row[field],other[field]);checks.append(dict(comparison=name,field=f'branch{identity[0]}/epoch{identity[1]}/{field}',
                    reference_value=tree_hash(row[field]),candidate_value=tree_hash(other[field]),exact_equal=c['exact_equal'],max_abs_difference=c['max_abs_difference'],passed=c['exact_equal'],explanation=c['failures']))
    write_csv(out/'paired_checks.csv',checks,'paired_checks.csv');assert all(x['passed'] for x in checks),'exact finite-acceptance gate failed'
    return checks


def validate_sources(asset_lock,out):
    assets=json.loads(asset_lock.read_text());assert assets['passed'] and len(assets['checks'])==9
    for row in assets['checks']:assert file_sha256(Path(row['resolved_path']))==row['resolved_expected_sha256']==row['actual_sha256']
    verify_lock(STUDY);audit=json.loads((CAL/'source_audit.json').read_text());source=audit['sources']['A0_2023']
    assert source['coarse_sha256']=='387678a492e7808229135bf2b352eba46b0092c1ff0197fc7c1664084d39569c'
    assert audit['protocol_sha256']=='cf3e694c10c5b4cd8256971bbd5a1783cd2b4abe32eca53caa6f74532273a110'
    data=data_for('PAMAP2','sr015_bl8',STUDY);assert data.manifest_sha=='da4825fe5bacbc0642eb246c647ffc080b8abcd9b801603bbd401ac3e830cbd6'
    assert len(data.starts['validation'])==1306 and len(data.starts['train'])//128==85
    original_batches=data.batches
    def batches(split,*a,**kw):
        if split=='test':raise AssertionError('test batching forbidden in this acceptance')
        return original_batches(split,*a,**kw)
    data.batches=batches
    lock=dict(package_commit='392fc08425371c5a7aad7db47b89440c9548ae7c',asset_inventory_sha256=file_sha256(asset_lock),source=source,
        protocol_sha256=audit['protocol_sha256'],data_manifest_sha256=data.manifest_sha,original_design_lock_sha256=file_sha256(STUDY/'design_lock.json'),
        helper_sha256=file_sha256(PKG/'code/local_worktree/integrations/e23/shared_frozen_calibration.py'),
        execution_sources={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
        source_audit_sha256=file_sha256(CAL/'source_audit.json'),original_runtime_worker_sha256=file_sha256(CAL/'tools/calibration_worker.py'),
        original_policy_sha256=file_sha256(CAL/'tools/policy.py'),current_common_adapter_run_verified_from_original_design_lock=True,
        test_policy='original SharedData initialization validates complete manifest/mask metadata; no test batches, scoring or selection allowed',registered_unix=time.time())
    write_json(out/'source_lock.json',lock);return data,source


def run(args):
    assert args.arm=='A0' and args.seed==2023 and args.epochs==2 and args.batch_size==128 and args.modes==['reference','shared','resume']
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_device(device)
    started=time.perf_counter();data,source=validate_sources(args.asset_lock,out)
    backend=dict(torch=torch.__version__,python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),
        cpu_threads=torch.get_num_threads(),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),float32_matmul_precision=torch.get_float32_matmul_precision(),
        GPU_shared=False,settings='original installed defaults explicitly recorded and kept identical; CUBLAS_WORKSPACE_CONFIG follows handoff; new controlled comparison,not historical100epoch bitwise claim')
    identity=dict(evidence_class=EVIDENCE,dataset='PAMAP2',scene='sr015_bl8',arm='A0',training_seed=2023,epochs_per_branch=2,batch_size=128,
        planned_total_branch_updates=850,reference_updates=340,shared_updates=340,resume_new_updates=170,
        source=source,backend=backend,validation_windows=1306,training_order_seed=2023,training_mask_seed=2023,validation_mask_seed=2021,
        source_implementation_commit=json.loads((Path(__file__).parent/'implementation_commit.json').read_text()) if (Path(__file__).parent/'implementation_commit.json').exists() else None,
        gates=dict(exact_max_tensor_difference=0.,train_update_time_reduction_min=.15,train_peak_allocated_increase_max=.05,inference_median_regression_max=.05,
            inference_scope='all four original paths:coarse,single64,single256,dual;timing variation is uncertainty,not relaxed thresholds'),
        selection='epoch0diagnostic only;own branch validation_full96_MAE strict< across completedepochs1/2;fixed.5/.5',test_accessed=False,Stage2=False,formal_model_promoted=False)
    write_json(out/'identity.json',identity)
    for command,name in [([sys.executable,'-m','pip','freeze'],'pip-freeze.txt'),(['nvidia-smi','-q'],'nvidia-smi.txt')]:
        try:p=subprocess.run(command,text=True,capture_output=True);(out/name).write_text(p.stdout+p.stderr+f'\nexit_code={p.returncode}\n')
        except OSError as e:(out/name).write_text(str(e)+'\n')
    first_batch_preflight(data,device,source,out)
    results={};epochrows=[];metricrows=[]
    for mode in ('reference','shared','resume'):
        write_json(out/'status.json',dict(phase=mode,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,resources=resources(device)))
        arm=Arm(mode,data,device,source,out)
        if mode=='resume':arm.load_resume(out/'shared/epoch1.pt')
        torch.cuda.reset_peak_memory_stats(device)
        with arm.guards():
            if mode=='reference':train_reference(arm)
            else:train_shared(arm,[2] if mode=='resume' else [1,2])
        result=arm.finish();results[mode]=result;epochrows.extend(arm.epoch_checks);metricrows.extend(arm.metrics)
        assert result['steps']==[170,170] and result['new_updates']==([85,85] if mode=='resume' else [170,170])
        assert result['actual_prefix_calls']==(340 if mode=='reference' else 85 if mode=='resume' else 170)
        arm.release();del arm
    assert sum(sum(x['new_updates']) for x in results.values())==850
    write_csv(out/'epoch_audit.csv',epochrows,'epoch_audit.csv');write_csv(out/'metrics.csv',metricrows,'metrics.csv')
    checks=compare_executions(out)
    inference={}
    for mode in ('reference','shared'):
        inference[mode],extra=inference_profile(data,device,source,results[mode]['checkpoint'],mode,out);metricrows+=extra
    write_csv(out/'epoch_audit.csv',epochrows,'epoch_audit.csv');write_csv(out/'metrics.csv',metricrows,'metrics.csv')
    parent=json.loads((STUDY/'PAMAP2/sr015_bl8/A0/seed_2023/stage1/selection.json').read_text())
    historical=next(x['seconds'] for x in parent['phases'] if x['phase']=='coarse');resource_rows=[]
    for mode,result in results.items():
        profiles=inference.get(mode,[{}])
        for profile in profiles:
            resource_rows.append(dict(evidence_class=EVIDENCE,execution=mode,arm='A0',seed=2023,device=backend['gpu'],torch=backend['torch'],cuda=backend['cuda'],python=backend['python'],
                parameters=result['total_parameters'],trainable_parameters=result['trainable_parameters'],updates_per_branch=result['new_updates'][0],cumulative_updates_per_branch=170,
                prefix_calls=result['actual_prefix_calls'],**result['timings'],total_wall_seconds=result['total_wall_seconds'],
                other_wall_seconds=result['total_wall_seconds']-sum(result['timings'].values()),historical_parent_seconds=historical,historical_cost_scope='same parent coarse training referenced once,not new cost and not charged onceperarm',
                train_peak_allocated_GiB=result['resources']['max_allocated_gib'],train_peak_reserved_GiB=result['resources']['max_reserved_gib'],
                inference_path=profile.get('path'),inference_batch_size=profile.get('batch_size'),inference_warmups=profile.get('warmups'),inference_repeats=profile.get('repeats'),
                inference_median_ms_per_window=profile.get('median_ms_per_window'),inference_p95_ms_per_window=profile.get('p95_ms_per_window'),inference_windows_per_second=profile.get('windows_per_second'),
                inference_peak_allocated_GiB=profile.get('peak_allocated_GiB'),inference_peak_reserved_GiB=profile.get('peak_reserved_GiB'),inference_actual_prefix_calls=profile.get('prefix_calls'),gpu_shared=False,
                inference_not_measured_reason='resume duplicates shared endpoint; not an inference performance arm' if mode=='resume' else None,
                input_already_on_GPU=True,train_peak_scope='training update sections only,reset before each batch;validation/inference excluded'))
    warm=json.loads((out/'single_batch_preflight.json').read_text())
    resource_rows.append(dict(evidence_class=EVIDENCE,execution='single_batch_warm_preflight',arm='A0',seed=2023,
        device=backend['gpu'],torch=backend['torch'],cuda=backend['cuda'],python=backend['python'],
        updates_per_branch=0,cumulative_updates_per_branch=0,prepare_seconds=warm['seconds'],total_wall_seconds=warm['seconds'],
        train_update_seconds=None,validation_seconds=None,save_seconds=None,train_peak_allocated_GiB=None,train_peak_reserved_GiB=None,
        warm_peak_allocated_GiB=warm['resources']['max_allocated_gib'],warm_peak_reserved_GiB=warm['resources']['max_reserved_gib'],
        gpu_shared=False,explanation='One registered main-gradient/output preflight on real batch128;zero optimizer updates;not a training or inference timing arm'))
    write_csv(out/'resources.csv',resource_rows,'resources.csv')
    ref,shared=results['reference'],results['shared'];reduction=1-shared['timings']['train_update_seconds']/ref['timings']['train_update_seconds']
    memory_increase=shared['resources']['max_allocated_gib']/ref['resources']['max_allocated_gib']-1
    regressions={a['path']:b['median_ms_per_window']/a['median_ms_per_window']-1 for a,b in zip(inference['reference'],inference['shared'])}
    resource_gates=dict(update_time_reduction=reduction,update_time_pass=reduction>=.15,train_peak_allocated_increase=memory_increase,memory_pass=memory_increase<=.05,
        inference_median_regressions=regressions,inference_pass=all(x<=.05 for x in regressions.values()),inference_repeated_samples=inference)
    resource_pass=resource_gates['update_time_pass'] and resource_gates['memory_pass'] and resource_gates['inference_pass']
    manifest=dict(status='PASS' if resource_pass else 'RESOURCE_GATE_FAILED_OR_UNCERTAIN',evidence_class=EVIDENCE,package_commit=identity['source_implementation_commit']['package_commit'] if identity['source_implementation_commit'] else '392fc08425371c5a7aad7db47b89440c9548ae7c',
        implementation_commit=identity['source_implementation_commit'],source_asset_lock_sha256=file_sha256(args.asset_lock),all_exact_checks_passed=True,resource_gates_passed=resource_pass,
        formal_model_promoted=False,test_accessed=False,Stage2_run=False,failures=[] if resource_pass else ['one or more preregistered resource gates failed or timing uncertainty; thresholds unchanged'],missing_fields=[],
        gates=resource_gates,executed_total_optimizer_updates=850,parent_coarse_cost_seconds=historical,warm_preflight_seconds=warm['seconds'],total_acceptance_wall_seconds=time.perf_counter()-started,
        local_checkpoints={k:dict(path=v['checkpoint'],sha256=v['checkpoint_sha256']) for k,v in results.items()})
    write_json(out/'result_manifest.json',manifest)
    lines=['# PAMAP2冻结粗前缀计算复用有限验收','',f'证据类别：{EVIDENCE}；结论：{manifest["status"]}。',
        'A0 seed2023、每支2轮、原batch128/Adam/.001/MAE+.25recentMAE，另共享第1轮checkpoint恢复重放第2轮，总850次分支更新；无测试、Stage2或正式权重晋级。',
        '原六权重与三依赖均通过封存哈希，依赖路径映射有记录。共享臂保留V125Adapter原归一化、外层尺度恢复运算顺序及观测值回填，完整原校准member独立优化。',
        f'所有loss、首批主MAE梯度、完整验证、各支选模、权重、Adam与Python/NumPy/CPU/CUDA RNG以及恢复终点逐值一致（最大tensor差0）；检查项{len(checks)}。',
        f'更新段节省：{reduction*100:.2f}%；peak allocated变化：{memory_increase*100:.2f}%；各原推理路径中位回退：{regressions}。',
        '独立推理为batch128、5warmup+20repeat，样本完整保存；受波动影响的失败不放宽5%阈值。训练工作流及推理分别reset peak。',
        '## 四问','',
        '1. 支持的判断：在本A0/PAMAP2两轮范围内，共享同一批次的确定性冻结前缀能否保持原双分支的独立优化轨迹，见严格等价与资源门槛分别判定。',
        '2. 收益来源：若耗时门槛通过，收益来自计算复用，精度逐值不变；不是新模型或新增精度收益。更新段节省不能称为完整Stage1节省。',
        '3. 限制：仅一模型seed和两轮；同卡顺序计时仍有系统波动。原数据初始化验证完整清单/测试掩码元数据，但禁止任何test批次、评价或选模；历史成本只引用不重新计算。',
        '4. 是否扩大：仅在严格等价、恢复和所有资源门槛均通过时，建议由原窗口审核是否扩大有限验收；失败或不确定时先分析差异，不自动扩展100轮或其他变体。',
        '配置、source lock、环境、各epoch日志、四张CSV、完整状态哈希与本地checkpoint路径/SHA均随本报告保留；大型权重与验证数组不进Git。']
    lines+=['','## 逐臂实测时间（秒）','','|臂|准备|更新段|完整验证|checkpoint保存|总wall|','|---|---|---|---|---|---|']
    for mode,value in results.items():
        t=value['timings'];lines.append(f"|{mode}|{t['prepare_seconds']:.3f}|{t['train_update_seconds']:.3f}|{t['validation_seconds']:.3f}|{t['save_seconds']:.3f}|{value['total_wall_seconds']:.3f}|")
    lines+=['','其他wall包含分层指标汇总、日志/哈希及未并入四分段的开销；不把更新段收益称作整个训练流程收益。',
        '','## 完整验证（1306窗口，原mask2021）','','|臂|分支|epoch|全96MSE|全96MAE|近32MSE|近32MAE|选中epoch|','|---|---|---|---|---|---|---|---|']
    for mode in ('reference','shared','resume'):
        for row in results[mode]['epoch_checks']:
            v=row['validation'];lines.append(f"|{mode}|{row['branch']}|{row['epoch']}|{v['full96']['mse']:.9f}|{v['full96']['mae']:.9f}|{v['recent32']['mse']:.9f}|{v['recent32']['mae']:.9f}|{row['selected_epoch']}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n');write_json(out/'status.json',dict(phase='complete',result=manifest['status'],elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(status=manifest['status'],all_exact_checks_passed=True,resource_gates=resource_gates),default=str),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--asset-lock',type=Path,required=True);p.add_argument('--arm',choices=['A0'],required=True);p.add_argument('--seed',type=int,choices=[2023],required=True)
    p.add_argument('--epochs',type=int,choices=[2],required=True);p.add_argument('--batch-size',type=int,choices=[128],required=True);p.add_argument('--modes',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except BaseException:
        import traceback
        if args.output.exists():
            error=traceback.format_exc();(args.output/'failure.txt').write_text(error)
            write_json(args.output/'result_manifest.json',dict(status='FAILED',evidence_class=EVIDENCE,package_commit='392fc08425371c5a7aad7db47b89440c9548ae7c',
                all_exact_checks_passed=False,resource_gates_passed=None,formal_model_promoted=False,test_accessed=False,Stage2_run=False,failures=[error],missing_fields=['see failure stage; unmeasured values are not zero']))
            # Return all four required tables even when preflight stops before training; unmeasured cells remain empty.
            for filename in ('metrics.csv','resources.csv','epoch_audit.csv','paired_checks.csv'):
                if not (args.output/filename).exists():
                    write_csv(args.output/filename,[dict(evidence_class=EVIDENCE,execution='stopped_after_failure',
                        explanation='Not measured: run stopped at the logged failure; no automatic rerun or budget expansion')],filename)
            (args.output/'report.md').write_text('# PAMAP2有限验收：失败并停止\n\n'+error+'\n\n未增加预算、变体或自动重跑。精度/资源未测项为空，详见四张CSV和失败日志。\n')
        raise
if __name__=='__main__':main()
