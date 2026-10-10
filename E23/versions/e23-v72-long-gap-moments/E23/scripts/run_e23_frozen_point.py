"""Predeclared frozen-repair controls; lock all selections before tests."""
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v68_frozen_point_sr015_bl8_20261005')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def sources():
 paths=[WORK/'pyproject.toml']
 for folder in ['src','integrations','scripts','configs','tests']:
  paths.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
 return {str(p.relative_to(WORK)):sha(p) for p in sorted(paths)}
def main():
 plan=read(ROOT/'plan.json');assert read(ROOT/'preflight.json')['status']==read(ROOT/'test-results.json')['status']=='passed'
 definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),protected=read(ROOT/'protected-files.json'),plan_sha256=sha(ROOT/'plan.json'),scenario_sha256=sha(plan['scenario_manifest']),bank_sha256=sha(plan['test_bank']))
 write(ROOT/'frozen-definition.json',definition)
 with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as archive:
  for rel in definition['sources']:archive.add(WORK/rel,arcname=rel,recursive=False)
 def verify():
  assert sources()==definition['sources'] and sha(ROOT/'plan.json')==definition['plan_sha256']
  assert sha(plan['scenario_manifest'])==definition['scenario_sha256'] and sha(plan['test_bank'])==definition['bank_sha256']
  for p,h in definition['protected'].items():assert sha(p)==h,p
 env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MALR_MISSING_SCENARIO':plan['scenario_manifest']}
 def run(command,cell,stage):
  verify();write(cell/(stage+'-command.json'),dict(command=command,environment={'MALR_MISSING_SCENARIO':plan['scenario_manifest']}));print(json.dumps(dict(event='start',cell=cell.name,stage=stage,utc=datetime.now(timezone.utc).isoformat())),flush=True)
  with (cell/(stage+'.log')).open('x') as log:subprocess.run(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  verify()
 parent=plan['initial_parent'].copy();cells=[]
 for condition in plan['conditions']:
  source=condition['source_parent'];options={k:source[k] for k in ['point_router','overlap_fusion','stage1_structure','point_candidate_readout','variable_calibration','front_variable_context','repair_coordinates','point_variable_readout','historical_key_policy','recent_mean_projection','temporal_evidence_readout','forecast_padding','repair_input_scaling','point_context_readout']}
  primary=ROOT/(condition['name']+'_mae');primary.mkdir();(primary/'stage1').symlink_to(Path(condition['stage1_manifest']).parents[1],target_is_directory=True)
  trial=dict(label=condition['label']+'-MAE',config=condition['config'],config_key=condition['name'],parent=parent.copy(),architecture_parent=parent['label'],only_change=condition['only_change'],cached_stage1=condition['stage1_manifest'],repair_update_policy='frozen_v1',stage2_policy_name='frozen_point',training_seed=condition['training_seed'],**options);write(primary/'plan.json',trial)
  common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),'--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),'--e23-config',condition['config'],'--training-seed',str(condition['training_seed']),'--device','cuda:1']
  run([sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--stage1-manifest',condition['stage1_manifest'],'--selection-metric','mae','--forecast-loss',plan['forecast_loss'],'--repair-retention',plan['repair_retention'],'--repair-update-policy','frozen_v1','--forecast-padding',options['forecast_padding'],'--extra-mse-checkpoint','--output',str(primary/'stage2')],primary,'stage2')
  secondary=ROOT/(condition['name']+'_mse');secondary.mkdir();(secondary/'stage1').symlink_to(Path(condition['stage1_manifest']).parents[1],target_is_directory=True);(secondary/'stage2/runs').mkdir(parents=True);(secondary/'stage2/runs/e23-stage2.json').write_bytes((primary/'stage2/runs/e23-stage2-mse.json').read_bytes());secondary_trial={**trial,'label':condition['label']+'-MSE'};write(secondary/'plan.json',secondary_trial)
  choices=[]
  for cell,spec,metric in [(primary,trial,'mae'),(secondary,secondary_trial,'mse')]:
   record=read(cell/'stage2/runs/e23-stage2.json');assert record['epochs_completed']==100 and record['repair_trainable_parameters']==0 and record['repair_update_policy']=='frozen_v1' and record['repair_learning_rate']==0 and record['predictor_initial_sha256']==plan['predictor_initial_sha256'];assert record['loss']=='forecast_mse_plus_0p2_mae' and record['repair_retention']=='uniform_v1' and record['selection_metric']==metric
   validation=dict(mse=record['best_val_mse'],mae=record['best_val_mae']);accepted=validation['mse']<parent['validation']['mse'] and validation['mae']<=parent['validation']['mae']
   lock=dict(parent=parent.copy(),validation=validation,selected_epoch=record['best_epoch'],selection_metric=metric,checkpoint_sha256=record['checkpoint_sha256'],accepted_on_validation=accepted,matched_v41_validation_gate=validation['mse']<plan['reference_v41_validation']['mse'] and validation['mae']<=plan['reference_v41_validation']['mae'],test_metrics_seen=False);write(cell/'validation-lock-before-tests.json',lock);cells.append((cell,spec,lock));choices.append((cell,spec,lock));print(json.dumps(dict(event='validation_locked',cell=cell.name,**lock)),flush=True)
  for cell,spec,lock in choices:
   if lock['accepted_on_validation']:
    parent=dict(label=spec['label'],root=str(cell),validation=lock['validation'],selection_metric=lock['selection_metric'],training_seed=condition['training_seed'],stage1_training_batch_size=read(Path(condition['stage1_manifest']))['stage1_training_batch_size'],repair_update_policy='frozen_v1',**options);break
 # Both training runs and every parent decision finish before any old20 evaluation.
 for cell,spec,lock in cells:write(cell/'parent-choice-before-tests.json',dict(next_parent=parent,uses_test_metrics=False,predeclared_priority=[c['name'] for c in plan['conditions']],within_condition_priority=['mae','mse']))
 results=[]
 for cell,spec,lock in cells:
  run([sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',spec['label'],'--training-seed',str(spec['training_seed']),'--e23-config',spec['config'],'--selection',str(cell/'stage2/runs/e23-stage2.json'),'--test-mask-bank',plan['test_bank'],'--output',str(cell/'evaluation'),'--device','cuda:1'],cell,'test')
  run([sys.executable,'scripts/diagnose_e23_front_context.py','--cell',str(cell)],cell,'diagnostics');score=read(cell/'evaluation/twenty-test-draws.json');assert len(score['test_draws'])==20
  diag=read(cell/'recovery-diagnostics.json');assert diag[0]['metrics']==diag[1]['metrics'] and diag[0]['last32_level_mse']==diag[1]['last32_level_mse'],'Frozen repair diagnostic differs'
  result=dict(label=spec['label'],cell=str(cell),trial=spec,validation_lock=lock,test_mean=score['test_mean'],test_std=score['test_mask_std_ddof1'],next_parent=parent,frozen_repair_state_and_output_verified=True);write(cell/'completed-audit.json',result);results.append(result);print(json.dumps(dict(event='scored',label=spec['label'],test_mean=score['test_mean'])),flush=True)
 verify();write(ROOT/'completed-audit.json',dict(status='complete',conditions=results,retained_parent=parent,stage1_reused=True,new_stage1_runs=0,new_stage2_runs=len(plan['conditions']),test_used_for_checkpoint_or_automatic_parent_selection=False,source_unchanged=True,protected_files_unchanged=True));print(json.dumps(dict(event='complete',retained_parent=parent)),flush=True)
if __name__=='__main__':main()
