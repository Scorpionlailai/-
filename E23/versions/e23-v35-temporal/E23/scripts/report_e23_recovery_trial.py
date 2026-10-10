"""Audited reports for a predeclared single-module recovery trial."""
import csv,json,yaml
from pathlib import Path
from scripts.report_e23_v31_trial import digest,checked_metric,validate_tests,write_csv
from integrations.forecast_compare.resources import selected_training_resources

ROOT=Path(__file__).resolve().parents[1]
PREDICTOR_SHA='18ad18b5af292a9b526233ea426f81ffb3899d1f4fc850441060843d8adc404a'


def load_candidate(root,trial):
    if json.loads((root/'code-freeze.json').read_text())['trial_sha256']!=digest(root/'trial.json'):
        raise ValueError('predeclared trial definition changed')
    cell=root/trial['condition']; paths=[cell/s/'runs'/f'e23-{s}.json' for s in ('stage1','stage2')]
    s1,s2=[json.loads(p.read_text()) for p in paths]
    for record in (s1,s2):
        expected=dict(epochs_completed=100,e23_config_sha256=digest(trial['config']),
            stage1_objective='global_mse_v1',recovery_schedule='iterative_tv_h_v1',
            recovery_rounds=1,batch_size=128,seed=2021)
        if any(record.get(k)!=v for k,v in expected.items()): raise ValueError('recovery trial training identity mismatch')
        config_model=yaml.safe_load(Path(trial['config']).read_text())['model']
        for k,v in trial['identity'].items():
            if config_model.get(k)!=v or (k in record and record[k]!=v):
                raise ValueError('recovery module identity disagrees with pinned config')
        if digest(record['checkpoint'])!=record['checkpoint_sha256']: raise ValueError('checkpoint changed')
    expected=dict(loss='forecast_mse_only',selection_metric='mae',forecast_input='explicit_v1',forecast_mixer='none',
        predictor_initial_sha256=PREDICTOR_SHA,optimizer_state_restored=False,rng_reset_seed=2021)
    if any(s2.get(k)!=v for k,v in expected.items()) or s2.get('repair_retention','none')!='none':
        raise ValueError('Stage2 differs from matched control')
    if Path(s2['stage1_manifest']).resolve()!=paths[0].resolve() or s2['stage1_manifest_sha256']!=digest(paths[0]) or s2['stage1_checkpoint_sha256']!=s1['checkpoint_sha256']:
        raise ValueError('Stage1 parent mismatch')
    replay=json.loads((cell/'validation_replay.json').read_text())
    if replay['windows']!=2688 or replay['coordinates']!=1806336 or replay['stage2_checkpoint_sha256']!=s2['checkpoint_sha256']:
        raise ValueError('validation replay mismatch')
    for m in ('mse','mae'):
        if abs(checked_metric(replay[m])-checked_metric(s2['best_val_'+m]))>1e-6: raise ValueError('validation value mismatch')
    for k in ('key','bank'):
        if replay['h_'+k+'_sha256_before']!=replay['h_'+k+'_sha256_after']: raise ValueError('H state changed')
    diag=json.loads((cell/'recovery-diagnostics.json').read_text())
    for stage,record in zip(('stage1','stage2'),(s1,s2)):
        d=diag[stage]
        if d['checkpoint_sha256']!=record['checkpoint_sha256'] or d['windows']!=2688 or d['hidden_coordinates']!=1023670 or d['causal_violations'] or not d['h_state_unchanged']:
            raise ValueError('historical diagnostic invalid')
    if diag['stage2']['predictor_parameters']!=35427: raise ValueError('head parameters changed')
    resources=selected_training_resources(paths)
    if resources['resource_status']!='measured_all_stages': raise ValueError('missing resources')
    for k in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib'): checked_metric(resources[k])
    return dict(**s2,model=trial['model'],condition=trial['condition'],selection_sha256=digest(paths[1]),
        stage1_record=s1,training_resources=resources)


def improves(record,reference):
    return record['best_val_mse']<float(reference['validation_mse']) and record['best_val_mae']<=float(reference['validation_mae'])


def report(root):
    trial=json.loads((root/'trial.json').read_text()); record=load_candidate(root,trial)
    if digest(trial['reference_csv'])!=trial['reference_csv_sha256']:
        raise ValueError('saved reference results changed')
    with Path(trial['reference_csv']).open() as f: rows=list(csv.DictReader(f))
    control=next(r for r in rows if r['model']==trial['matched_reference'])
    best=next(r for r in rows if r['model']==trial['best_reference'])
    cell=root/trial['condition']; tests=json.loads((cell/'three-test-draws.json').read_text()); validate_tests(record,tests)
    diag=json.loads((cell/'recovery-diagnostics.json').read_text()); resource=record['training_resources']
    sides=[json.loads(Path(p).read_text()) for p in resource['resource_record_paths']]
    row=dict(model=record['model'],dataset='ETTh1',scenario='P4',input_len=96,pred_len=96,training_runs=1,test_draws=3,
        validation_mse=record['best_val_mse'],validation_mae=record['best_val_mae'],selection_metric='fixed_validation_mae',
        stage1_best_epoch=record['stage1_record']['best_epoch'],stage2_best_epoch=record['best_epoch'],
        stage1_epochs_completed=100,stage2_epochs_completed=100,stage1_objective='global_mse_v1',stage2_objective='forecast_mse_only',
        historical_query_mode=record.get('historical_query_mode','window_patch_v1'),
        temporal_selection=trial['identity'].get('temporal_selection',record.get('temporal_selection','banded_v1')),
        overlap_fusion=record.get('overlap_fusion','uniform_v1'),recovery_rounds=1,
        repair_parameters=diag['stage2']['repair_parameters'],predictor_parameters=35427,
        stage1_validation_repair_mse_global=diag['stage1']['global_mse'],stage2_validation_repair_mse_global=diag['stage2']['global_mse'],
        stage1_full_workflow_seconds=sides[0]['full_workflow_seconds'],stage2_full_workflow_seconds=sides[1]['full_workflow_seconds'],
        stage1_training_seconds=sides[0]['training_stages_seconds'],stage2_training_seconds=sides[1]['training_stages_seconds'],
        stage1_batch_size=128,stage2_batch_size=128,stage1_precision='float32',stage2_precision='float32',training_seed=2021,
        gpu_uuid=sides[0]['device']['uuid'],gpu_index=sides[0]['device']['index'],
        checkpoint_sha256=record['checkpoint_sha256'],stage1_checkpoint_sha256=record['stage1_record']['checkpoint_sha256'],
        predictor_initial_sha256=record['predictor_initial_sha256'],e23_config_sha256=record['e23_config_sha256'],
        result_path=str(cell/'three-test-draws.json'),test_result_sha256=digest(cell/'three-test-draws.json'),
        resource_record_paths=';'.join(resource['resource_record_paths']),
        matched_reference=trial['matched_reference'],validation_improves_matched_control=improves(record,control),
        validation_improves_best=improves(record,best),
        diagnostic_seconds=sum(d['diagnostic_seconds'] for d in diag.values()),
        additional_test_seconds=json.loads((cell/'test-runtime.json').read_text())['seconds'],
        **{k:resource[k] for k in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib','gpu')})
    for m in ('mse','mae'):
        for s in ('mean','std'): row['test_'+m+'_'+s]=tests['test_'+s][m]
    rows.append(row)
    decision=dict(retained_model=record['model'] if improves(record,best) else trial['best_reference'],
        matched_control_passed=improves(record,control),selection_source='original fixed validation only; tests not used',
        actual_matrix_training_full_seconds=resource['full_workflow_seconds'])
    output=ROOT/'docs/experiments'; stem=trial['report_stem']
    write_csv(output/(stem+'_comparison_2026-10-04.csv'),rows)
    write_csv(output/(stem+'_test_groups_2026-10-04.csv'),[dict(model=record['model'],**g) for g in tests['test_groups']])
    write_csv(output/(stem+'_recovery_diagnostics_2026-10-04.csv'),[dict(model=record['model'],**d) for d in diag.values()])
    (root/'final-selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    return rows,decision
