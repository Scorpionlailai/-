"""Stage1 only: frozen v79 coarse repair and history, train a zero-init dense residual."""
from time import perf_counter
ENTRY_STARTED=perf_counter()
from pathlib import Path
import json,sys,hashlib,copy
import torch
from weather_common import ROOT,protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (build_repair_model,load_stage1_selection,_new_memory,_starts,
 _validation_repair_metrics,_paired_payload,_atomic_torch_save,_atomic_json,_file_sha256,_identity,_revision,_memory_state_sha256)
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources

def coarse_hash(model):
 d=hashlib.sha256()
 for name,value in model.state_dict().items():
  if not name.startswith('recent_mean_projector.'):
   d.update(name.encode());d.update(value.detach().cpu().contiguous().numpy().tobytes())
 return d.hexdigest()

def bank_content(memory):
 state=copy.deepcopy(memory.state_dict());state['snapshot']['epoch']=0
 return _memory_state_sha256(state)

def train(scene_name):
 torch.set_num_threads(4);torch.manual_seed(2023)
 plan=json.loads((ROOT/'plan.json').read_text());scene=next(s for s in plan['scenarios'] if s['name']==scene_name);entry=plan['models'][0]
 output=ROOT/entry['label']/scene_name/'stage1';assert not output.exists();device=torch.device(plan['device'])
 with TrainingResources(output,device=device,batch_size=128,model=entry['label'],started_at=ENTRY_STARTED) as resources:
  with resources.stage('repair_refinement'):
   protocol=protocol_for(plan,scene);settings=load_e23_settings(Path(entry['config']),protocol=protocol,allow_variable_scaling=True);parent_settings=load_e23_settings(Path(plan['parent_settings']),protocol=protocol,allow_variable_scaling=True)
   selection=Path(plan['parent_selection'])
   original=json.loads(selection.read_text());payload=load_stage1_selection(selection,protocol,parent_settings,device=device)
   model=build_repair_model(protocol,settings,device=device)
   incompatible=model.load_state_dict(payload['model_state'],strict=False)
   assert not incompatible.unexpected_keys and incompatible.missing_keys and all('.global_dense_refiner.' in k for k in incompatible.missing_keys)
   model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot())
   model.eval().requires_grad_(False);refiner=model.recent_mean_projector;refiner.requires_grad_(True)
   assert sum(p.numel() for p in model.parameters() if p.requires_grad)==entry['trainable_parameters']
   original_hash=coarse_hash(model);bank_hash=bank_content(memory)
   # Initial zero residual must exactly reproduce the parent validation score.
   initial=_validation_repair_metrics(model,protocol,settings,device=device)
   assert abs(initial['global_mse']-original['validation_repair_metrics']['global_mse'])<1e-6
   optimizer=torch.optim.Adam(refiner.parameters(),lr=plan['actual_refinement_learning_rate'],weight_decay=0)
   base_epoch=int(payload['epoch']);best=float('inf');best_epoch=0;history=[];start=perf_counter()
   checkpoint=output/'checkpoints/e23-stage1.pt';manifest=output/'runs/e23-stage1.json'
   for local_epoch in range(1,plan['refinement_epochs']+1):
    epoch=base_epoch+local_epoch
    # Advance paired state metadata only. Keep the original frozen key encoder and bank contents.
    retrieval=model.historical.retrieval;retrieval.ema_encoder.load_state_dict(retrieval.frozen_encoder.state_dict());retrieval.begin_epoch(epoch)
    model.eval();refiner.train()
    for index,cpu in enumerate(iter_p4_batches(protocol,'train',batch_size=128,epoch=local_epoch-1),1):
     b=cpu.to(device);starts=_starts(b,stride=model.stride,patches=model.patch_count,device=device)
     optimizer.zero_grad(set_to_none=True)
     out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts,epoch=epoch,explore_variable=False,specialist_exposure=0).explicit
     hidden=~b.observed_mask & out.sequence_validity
     error=(out.normalized_sequence-b.normalized_target)*b.query_std[:,None]
     loss=error[hidden].square().mean()+plan['refinement_loss_tail_weight']*error[:,-32:][hidden[:,-32:]].square().mean()+plan['refinement_loss_tail_mae_weight']*error[:,-32:][hidden[:,-32:]].abs().mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
     if index==1 or index%10==0:print(f'stage=repair_refinement epoch={local_epoch}/100 batch={index}/{len(protocol.refs["train"])//128} loss={loss.item():.6f}',flush=True)
    memory.publish_epoch(epoch);model.historical.refresh_snapshot(memory.snapshot())
    metrics=_validation_repair_metrics(model,protocol,settings,device=device);history.append({'epoch':epoch,'refinement_epoch':local_epoch,'metrics':metrics})
    if metrics['global_mse']<best:
     best=metrics['global_mse'];best_epoch=epoch;best_local=local_epoch
     paired=_paired_payload(model,memory,epoch=epoch,validation_mse=best,settings=settings,validation_metrics=metrics)
     _atomic_torch_save(paired,checkpoint)
    assert coarse_hash(model)==original_hash and bank_content(memory)==bank_hash
    _atomic_json({'selection_metric_space':'train_standardized','initial_parent_metrics':initial,'epochs':history},output/'runs/e23-stage1-history.json')
    print(f'stage=repair_refinement epoch={local_epoch}/100 val_repair_mse={metrics["global_mse"]:.8f} best_refinement_epoch={best_local}',flush=True)
   saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
   record={'stage':'repair','best_epoch':best_epoch,'best_refinement_epoch':best_local,'epochs_completed':100,'best_val_repair_mse':best,'checkpoint':str(checkpoint),'checkpoint_sha256':_file_sha256(checkpoint),'memory_sha256':saved['memory_sha256'],'key_sha256':saved['key_sha256'],'validation_repair_metrics':saved['validation_repair_metrics'],'code_revision':_revision(),'optimizer_state_restored':False,'refinement_loss':'full_hidden_mse_plus_last32_mse_plus_last32_mae','refinement_loss_tail_mae_weight':plan['refinement_loss_tail_mae_weight'],'refinement_loss_tail_weight':plan['refinement_loss_tail_weight'],'learning_rate':plan['actual_refinement_learning_rate'],'training_seconds':perf_counter()-start,'selection':'lowest fixed-validation hidden-coordinate global repair MSE','stage1_training_batch_size':128,'validation_batch_size':128,'optimizer_updates':len(protocol.refs['train'])//128*100,'training_windows_per_epoch':len(protocol.refs['train'])//128*128,'stage1_aux_forecast':'none','coarse_parameters_frozen_verified':False,'expert_parameters_frozen_verified':True,'parent_mean_projector_trainable_parameters':plan['stage1_parent_mean_trainable_parameters'],'bank_contents_frozen_verified':True,'parent_selection':str(selection),'parent_checkpoint_sha256':original['checkpoint_sha256'],'parent_training_seconds':original['training_seconds'],'parent_epochs_completed':original['epochs_completed'],'total_stage1_epochs':original['epochs_completed']+100,'trainable_parameters':entry['trainable_parameters'],'frozen_expert_hash':original_hash,**_identity(protocol,settings)}
   _atomic_json(record,manifest);resources.selection_manifest=manifest
 print(json.dumps({'manifest':str(manifest),'resources':str(resources.report_path)}),flush=True)

if __name__=='__main__':train(sys.argv[1])
