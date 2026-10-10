"""Fixed seed cohort; original head architecture, author T1 remains cached."""
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v65_seed_search_sr015_bl8_20261005')
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
 plan=read(ROOT/'plan.json');assert read(ROOT/'preflight.json')['status']=='passed'
 definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),protected=read(ROOT/'protected-files.json'),plan_sha256=sha(ROOT/'plan.json'),scenario_sha256=sha(plan['scenario_manifest']),bank_sha256=sha(plan['test_bank']));write(ROOT/'frozen-definition.json',definition)
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
 initial=plan['initial_parent'];parent=initial.copy();results=[]
 options={k:initial[k] for k in ['point_router','overlap_fusion','stage1_structure','point_candidate_readout','variable_calibration','front_variable_context','repair_coordinates','point_variable_readout','historical_key_policy','recent_mean_projection','temporal_evidence_readout','forecast_padding','repair_input_scaling']}
 for seed in plan['seeds']:
  config=plan['configs'][str(seed)];expected_head=plan['predictor_initial_sha256_by_seed'][str(seed)];batch=plan['stage1_training_batch_size']
  primary=ROOT/f'seed_{seed}_mae';primary.mkdir();trial=dict(label=f'E23-v65-seed{seed}-MAE',training_seed=seed,config=config,config_key=str(seed),parent=parent.copy(),architecture_parent=initial['label'],only_change='training_random_seed',repair_update_policy=plan['repair_update_policy'],**options);write(primary/'plan.json',trial)
  common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),'--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),'--e23-config',config,'--training-seed',str(seed),'--device','cuda:1']
  s1=primary/'stage1/runs/e23-stage1.json'
  run([sys.executable,'-m','integrations.e23.run','train-repair',*common,'--repair-batch-size',str(batch),'--output',str(primary/'stage1')],primary,'stage1');assert read(s1)['epochs_completed']==100
  run([sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--stage1-manifest',str(s1),'--selection-metric','mae','--forecast-loss',plan['forecast_loss'],'--repair-retention',plan['repair_retention'],'--repair-update-policy',plan['repair_update_policy'],'--forecast-padding',options['forecast_padding'],'--extra-mse-checkpoint','--output',str(primary/'stage2')],primary,'stage2')
  secondary=ROOT/f'seed_{seed}_mse';secondary.mkdir();(secondary/'stage1').symlink_to(primary/'stage1',target_is_directory=True);(secondary/'stage2/runs').mkdir(parents=True);(secondary/'stage2/runs/e23-stage2.json').write_bytes((primary/'stage2/runs/e23-stage2-mse.json').read_bytes());secondary_trial={**trial,'label':f'E23-v65-seed{seed}-MSE','cached_stage1':str(s1)};write(secondary/'plan.json',secondary_trial)
  choices=[]
  for cell,spec,metric in [(primary,trial,'mae'),(secondary,secondary_trial,'mse')]:
   record=read(cell/'stage2/runs/e23-stage2.json');assert record['epochs_completed']==100 and record['predictor_initial_sha256']==expected_head and record['selection_metric']==metric and record['repair_update_policy']==plan['repair_update_policy'];assert record['loss']=='forecast_mse_plus_0p2_mae' and record['repair_retention']=='uniform_v1'
   validation=dict(mse=record['best_val_mse'],mae=record['best_val_mae']);accepted=validation['mse']<parent['validation']['mse'] and validation['mae']<=parent['validation']['mae']
   lock=dict(parent=parent.copy(),validation=validation,selected_epoch=record['best_epoch'],selection_metric=metric,checkpoint_sha256=record['checkpoint_sha256'],accepted_on_validation=accepted,test_metrics_seen=False);write(cell/'validation-lock-before-tests.json',lock);choices.append((cell,spec,lock));print(json.dumps(dict(event='validation_locked',cell=cell.name,**lock)),flush=True)
  for cell,spec,lock in choices:
   if lock['accepted_on_validation']:
    parent=dict(label=spec['label'],root=str(cell),validation=lock['validation'],selection_metric=lock['selection_metric'],training_seed=seed,stage1_training_batch_size=batch,repair_update_policy=plan['repair_update_policy'],**options);break
  for cell,spec,lock in choices:write(cell/'parent-choice-before-tests.json',dict(next_parent=parent,uses_test_metrics=False,within_seed_priority=['mae','mse']))
  for cell,spec,lock in choices:
   run([sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',spec['label'],'--training-seed',str(seed),'--e23-config',config,'--selection',str(cell/'stage2/runs/e23-stage2.json'),'--test-mask-bank',plan['test_bank'],'--output',str(cell/'evaluation'),'--device','cuda:1'],cell,'test')
   run([sys.executable,'scripts/diagnose_e23_front_context.py','--cell',str(cell)],cell,'diagnostics');score=read(cell/'evaluation/twenty-test-draws.json');assert len(score['test_draws'])==20 and score['validation_mask_sha256']==plan['validation_mask_sha256']
   result=dict(label=spec['label'],cell=str(cell),trial=spec,validation_lock=lock,test_mean=score['test_mean'],test_std=score['test_mask_std_ddof1'],next_parent=parent);write(cell/'completed-audit.json',result);results.append(result);print(json.dumps(dict(event='scored',label=spec['label'],test_mean=score['test_mean'])),flush=True)
  write(ROOT/f'seed_{seed}-completed.json',dict(training_seed=seed,retained_parent=parent,conditions=[r for r in results if r['trial']['training_seed']==seed],test_used_for_checkpoint_or_automatic_parent_selection=False))
 verify();write(ROOT/'completed-audit.json',dict(status='complete',conditions=results,retained_parent=parent,stage1_reused=False,new_stage1_runs=len(plan['seeds']),new_stage2_runs=len(plan['seeds']),test_used_for_checkpoint_or_automatic_parent_selection=False,source_unchanged=True,protected_files_unchanged=True));print(json.dumps(dict(event='complete',retained_parent=parent)),flush=True)
if __name__=='__main__':main()
