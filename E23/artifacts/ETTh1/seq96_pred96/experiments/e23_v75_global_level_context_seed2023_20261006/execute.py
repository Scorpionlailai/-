"""Run the locked recovery trial with validation locks and shared test masks."""
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')

def stage2_lock(record, plan, scene, metric):
    expected = dict(seed=plan['training_seed'], epochs_completed=100, selection_metric=metric,
                    repair_update_policy='frozen_v1', repair_trainable_parameters=0,
                    repair_retention='uniform_v1', loss='forecast_mse_plus_0p2_mae',
                    forecast_input='explicit_v1', forecast_mixer='none',
                    predictor_initial_sha256=plan['predictor_initial_sha256'],
                    validation_mask_sha256=scene['validation_mask_sha256'],
                    stage1_gap_structure='long_gap_moments_v1',gap_min_length=16,
                    gap_level_weight=.2,gap_trend_weight=.2,gap_weighting='coordinate_count',
                    e23_config_sha256=sha(plan['config']),front_level_context='global_level_v1',front_context_features=7,front_evidence_features=5,front_level_features=2,front_level_extra_parameters=128,front_level_sources='observed_global_values_and_visible_mean',front_level_scale='global_train_standardized',front_level_projection='parallel_zero_2_to_64',front_level_no_observations='zero')
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'locked v75 Stage2 {key} mismatch')
    validation = dict(mse=record['best_val_mse'], mae=record['best_val_mae'])
    if not all(math.isfinite(v) and v >= 0 for v in validation.values()):
        raise ValueError('nonfinite validation score')
    if sha(record['checkpoint']) != record['checkpoint_sha256']:
        raise ValueError('selected checkpoint changed')
    return dict(selected_epoch=record['best_epoch'], selection_metric=metric,
                checkpoint_sha256=record['checkpoint_sha256'], validation=validation,
                test_bank_sha256=scene['test_bank_sha256'], test_metrics_seen=False,
                development_test_previously_exposed=True,selected_checkpoint_test_metrics_seen=False, original_predictor_parameters=35427)

def paired_draws(e23, reference):
    for key in ['test_mask_bank_sha256', 'validation_mask_sha256', 'metric_space', 'split']:
        if e23[key] != reference[key]:
            raise ValueError(f'paired {key} mismatch')
    a,b = e23['test_draws'],reference['test_draws']
    if len(a) != 20 or len(b) != 20:
        raise ValueError('expected 20 paired mask positions')
    result = []
    for left,right in zip(a,b):
        for key in ['group_id','mask_seed','mask_sha256','windows','target_elements']:
            if left[key] != right[key]:
                raise ValueError(f'paired draw {key} mismatch')
        if not all(math.isfinite(x[k]) and x[k] >= 0 for x in (left,right) for k in ['mse','mae']):
            raise ValueError('nonfinite paired score')
        result.append(dict(group_id=left['group_id'], mask_seed=left['mask_seed'],
                           mse_delta=left['mse']-right['mse'], mae_delta=left['mae']-right['mae']))
    return result

def training_command(plan, scene, stage):
    work = Path(plan['work']); cell = ROOT/scene['name']/'E23-v75-mae'
    common = ['--protocol-config',plan['protocol_config'],
              '--migration-catalog',str(work.parent/'E22/migration/paper_catalog.jsonl'),
              '--mask-artifacts',str(work.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),
              '--e23-config',plan['config'],'--training-seed',str(plan['training_seed']),
              '--device',plan['device']]
    command = [sys.executable,'-m','integrations.e23.run','train-'+stage,*common]
    if stage == 'repair':
        command += ['--repair-batch-size',str(plan['stage1_batch_size']), '--output',str(cell/'stage1')]
    elif stage == 'forecast':
        command += ['--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),
                    '--selection-metric','mae','--forecast-loss',plan['forecast_loss'],
                    '--repair-retention',plan['repair_retention'],
                    '--repair-update-policy','frozen_v1','--forecast-padding',plan['forecast_padding'],
                    '--extra-mse-checkpoint','--output',str(cell/'stage2')]
    else:
        raise ValueError('unsupported training stage')
    return command

def main():
    plan = read(ROOT/'plan.json'); work = Path(plan['work'])
    assert read(ROOT/'preflight.json')['status'] == 'passed'
    frozen = read(ROOT/'frozen-definition.json')
    def verify():
        assert sha(ROOT/'plan.json') == frozen['plan_sha256']
        for p,h in frozen['protected'].items():
            assert sha(p) == h, p
        for rel,h in frozen['sources'].items():
            assert sha(work/rel) == h, rel
        for p,h in frozen['controller'].items():
            assert sha(ROOT/p) == h, p
    def announce(event, **fields):
        print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**fields)),flush=True)
    def ledger(message):
        with (ROOT/'progress.md').open('a') as f:
            f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
    def run(command, cell, stage, expected, scene):
        verify()
        if Path(expected).exists():
            announce('reuse_completed_step',scene=scene['name'],stage=stage)
            return
        command_file = cell/(stage+'-command.json')
        if command_file.exists():
            raise RuntimeError(f'partial step requires review before restarting: {command_file}')
        env = {**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        env.pop('MALR_MISSING_SCENARIO',None)
        if scene['scenario_manifest']:
            env['MALR_MISSING_SCENARIO'] = scene['scenario_manifest']
        write(command_file,dict(command=command,environment={k:env.get(k) for k in ['PYTHONPATH','MALR_MISSING_SCENARIO','OMP_NUM_THREADS','MKL_NUM_THREADS']}))
        announce('start',scene=scene['name'],stage=stage)
        ledger('Task3 start '+scene['name']+' '+stage)
        with (cell/(stage+'.log')).open('x') as log:
            subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert Path(expected).exists(),expected
        verify()
    results = []
    for scene in plan['scenarios']:
        scene_root = ROOT/scene['name']; scene_root.mkdir(exist_ok=True)
        audit = scene_root/'completed-audit.json'
        if audit.exists():
            verify(); results.append(read(audit)); continue
        primary = scene_root/'E23-v75-mae'; primary.mkdir(exist_ok=True)
        if not (primary/'plan.json').exists():
            write(primary/'plan.json',dict(label='E23-v75-seed2023-frozen-MAE',training_seed=2023,
                                         config=plan['config'],scenario=scene['name'],
                                         repair_update_policy='frozen_v1',only_change=plan['only_change']))
        s1 = primary/'stage1/runs/e23-stage1.json'
        run(training_command(plan,scene,'repair'),primary,'stage1',s1,scene)
        r1 = read(s1)
        assert (r1['epochs_completed'],r1['seed'],r1['stage1_objective'],r1['optimizer_updates']) == (100,2023,'global_mse_v1',6600)
        assert r1['validation_mask_sha256'] == scene['validation_mask_sha256']
        assert r1['stage1_gap_structure']=='long_gap_moments_v1' and r1['gap_min_length']==16
        assert r1['gap_level_weight']==r1['gap_trend_weight']==.2
        s2 = primary/'stage2/runs/e23-stage2.json'
        run(training_command(plan,scene,'forecast'),primary,'stage2',s2,scene)
        secondary = scene_root/'E23-v75-mse'; secondary.mkdir(exist_ok=True)
        if not (secondary/'stage1').exists():
            (secondary/'stage1').symlink_to(primary/'stage1',target_is_directory=True)
        if not (secondary/'stage2/runs/e23-stage2.json').exists():
            (secondary/'stage2/runs').mkdir(parents=True,exist_ok=True)
            (secondary/'stage2/runs/e23-stage2.json').write_bytes((primary/'stage2/runs/e23-stage2-mse.json').read_bytes())
        if not (secondary/'plan.json').exists():
            write(secondary/'plan.json',{**read(primary/'plan.json'),'label':'E23-v75-seed2023-frozen-MSE','cached_stage1':str(s1)})
        for cell,metric in [(primary,'mae'),(secondary,'mse')]:
            lock_path = cell/'validation-lock-before-tests.json'
            lock = stage2_lock(read(cell/'stage2/runs/e23-stage2.json'),plan,scene,metric)
            if lock_path.exists():
                assert read(lock_path) == lock
            else:
                write(lock_path,lock)
            announce('validation_locked',scene=scene['name'],metric=metric,**lock)
        t1 = read(scene['t1_evaluation']); old = read(scene['v69_evaluation'])
        choices = []
        for cell,metric in [(primary,'mae'),(secondary,'mse')]:
            evaluation = cell/'evaluation/twenty-test-draws.json'
            run([sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B',
                 '--model-label','E23-v75-seed2023-frozen-'+metric.upper(),'--training-seed','2023',
                 '--e23-config',plan['config'],'--selection',str(cell/'stage2/runs/e23-stage2.json'),
                 '--test-mask-bank',scene['test_bank'],'--output',str(cell/'evaluation'),
                 '--device',plan['device']],cell,'test',evaluation,scene)
            score = read(evaluation)
            paired = paired_draws(score,t1); paired_draws(score,old)
            item = dict(selection_metric=metric,cell=str(cell),test_mean=score['test_mean'],
                        test_std=score['test_mask_std_ddof1'],
                        paired_T1_mean_delta={k:sum(d[k+'_delta'] for d in paired)/20 for k in ['mse','mae']},
                        paired_T1_wins={k:sum(d[k+'_delta']<0 for d in paired) for k in ['mse','mae']})
            write(cell/'completed-audit.json',item); choices.append(item)
            announce('scored',scene=scene['name'],**item)
        run([sys.executable,'scripts/diagnose_e23_front_context.py','--cell',str(primary)],
            primary,'diagnostics',primary/'recovery-diagnostics.json',scene)
        item = dict(scene=scene['name'],status='complete',choices=choices,
                    t1_test_mean=t1['test_mean'],v69_test_mean=old['test_mean'],
                    test_used_for_checkpoint_selection=False,new_stage1_runs=1,new_stage2_runs=1)
        write(audit,item); results.append(item)
        ledger('Task3 complete '+scene['name']+'; both checkpoints locked before tests, frozen repair,20 paired positions.')
    verify()
    write(ROOT/'completed-audit.json',dict(status='complete',scenarios=results,
          original_v69_source_unchanged=True,protected_files_unchanged=True,
          new_stage1_runs=1,new_stage2_runs=1,new_T1_training_runs=0,test_used_for_selection=False))
    announce('complete',scenarios=[s['scene'] for s in results])

if __name__ == '__main__':
    main()
