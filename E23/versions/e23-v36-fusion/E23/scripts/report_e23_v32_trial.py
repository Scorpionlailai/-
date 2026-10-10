"""Validate frozen prediction-preservation trials and report measured resources."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from integrations.e23.prediction_preservation import checked_preservation
from integrations.forecast_compare.resources import selected_training_resources
from scripts.report_e23_v31_trial import digest,checked_metric,validate_tests,validate_repair_diagnostic,write_csv

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/e23_multi_expert_p4_width64_prediction_preservation_v32.yaml'
MODES={'A':'none','B':'clean_forecast_v1'}


def improves(candidate, reference):
    values=[checked_metric(r['best_val_'+k]) for r in (candidate,reference) for k in ('mse','mae')]
    return values[0]<values[2] and values[1]<=values[3]


def check_trial_identity(condition, record):
    expected=dict(loss='forecast_mse_only',selection_metric='mae',forecast_input='explicit_v1',
                  forecast_mixer='none',epochs_completed=100,optimizer_state_restored=False,rng_reset_seed=2021)
    if any(record.get(k)!=v for k,v in expected.items()) or record.get('repair_retention','none')!='none':
        raise ValueError('v32 Stage2 training identity mismatch')
    if record.get('stage1_preservation','none')!=MODES[condition]:
        raise ValueError('v32 preservation condition mismatch')
    checked_preservation(record)


def load_candidate(root,condition):
    cell=root/condition;s1path=cell/'stage1/runs/e23-stage1.json';s2path=cell/'stage2/runs/e23-stage2.json'
    s1=json.loads(s1path.read_text());s2=json.loads(s2path.read_text());check_trial_identity(condition,s2)
    for record in (s1,s2):
        if record['epochs_completed']!=100 or record['e23_config_sha256']!=digest(CONFIG) or digest(record['checkpoint'])!=record['checkpoint_sha256']:
            raise ValueError('v32 checkpoint/config/budget mismatch')
    if Path(s2['stage1_manifest']).resolve()!=s1path.resolve() or s2['stage1_manifest_sha256']!=digest(s1path) or s2['stage1_checkpoint_sha256']!=s1['checkpoint_sha256']:
        raise ValueError('v32 parent Stage1 mismatch')
    if checked_preservation(s1)!=checked_preservation(s2):raise ValueError('v32 preservation parent mismatch')
    if condition=='B':
        if digest(s1['preservation_probe_checkpoint'])!=s1['preservation_probe_checkpoint_sha256'] or s1['preservation_probe_frozen_sha256']!=s1['preservation_probe_final_sha256']:
            raise ValueError('v32 frozen auxiliary changed')
        if s1['preservation_probe_training_batches']!=1320 or s1['preservation_active_batches']!=5280 or s1['preservation_active_windows']!=675840:
            raise ValueError('v32 auxiliary coverage/budget mismatch')
    replay=json.loads((cell/'validation_replay.json').read_text())
    if replay['stage2_checkpoint_sha256']!=s2['checkpoint_sha256'] or replay['windows']!=2688 or replay['coordinates']!=1806336:raise ValueError('v32 replay ownership mismatch')
    for metric in ('mse','mae'):
        if abs(checked_metric(replay[metric])-checked_metric(s2['best_val_'+metric]))>1e-6:raise ValueError('v32 replay value mismatch')
    for name in ('key','bank'):
        if replay['h_'+name+'_sha256_before']!=replay['h_'+name+'_sha256_after']:raise ValueError('v32 historical state changed')
    resources=selected_training_resources([s1path,s2path])
    if resources['resource_status']!='measured_all_stages':raise ValueError('v32 incomplete resources')
    for key in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib'):checked_metric(resources[key])
    record=dict(**s2,model='E23_v32_preservation_'+condition,condition=condition,
                selection_sha256=digest(s2path),training_resources=resources,stage1_record=s1)
    repair=json.loads((cell/'repair_diagnostic.json').read_text());validate_repair_diagnostic(record,repair)
    return record


def report(root,output):
    records=[load_candidate(root,c) for c in MODES]
    if len({r['predictor_initial_sha256'] for r in records})!=1:raise ValueError('v32 unequal predictor initialization')
    old=list(csv.DictReader((ROOT/'docs/experiments/ETTh1_P4_E23_v31_T1_comparison_2026-10-03.csv').open()))
    rows=[r for r in old if r['model'] in ('T1','E23_v31_B')];groups=[]
    for record in records:
        cell=root/record['condition'];tests=json.loads((cell/'three-test-draws.json').read_text());validate_tests(record,tests)
        if checked_preservation(tests)!=checked_preservation(record):raise ValueError('v32 test mode mismatch')
        resources=record['training_resources'];side=[json.loads(Path(p).read_text()) for p in resources['resource_record_paths']]
        s1=record['stage1_record'];repair=json.loads((cell/'repair_diagnostic.json').read_text())
        row=dict(dataset='ETTh1',scenario='P4',input_len=96,pred_len=96,model=record['model'],condition=record['condition'],training_runs=1,test_draws=3,selection_metric='fixed_validation_mae',
            validation_mse=record['best_val_mse'],validation_mae=record['best_val_mae'],stage1_best_epoch=s1['best_epoch'],stage2_best_epoch=record['best_epoch'],stage1_epochs_completed=100,stage2_epochs_completed=100,
            stage1_preservation=record.get('stage1_preservation','none'),stage2_objective='forecast_mse_only',stage1_precision='float32',stage2_precision='float32',stage1_batch_size=128,stage2_batch_size=128,training_seed=2021,
            stage1_validation_repair_mse_global=s1['validation_repair_metrics']['global_mse'],stage1_validation_repair_mae_global=s1['validation_repair_metrics']['global_mae'],stage2_validation_repair_mse_global=repair['global_mse'],stage2_validation_repair_mae_global=repair['global_mae'],
            repair_parameters=repair['repair_parameters'],predictor_parameters=repair['predictor_parameters'],auxiliary_parameters=s1.get('preservation_probe_parameters',0),
            stage1_full_workflow_seconds=side[0]['full_workflow_seconds'],stage2_full_workflow_seconds=side[1]['full_workflow_seconds'],stage1_training_seconds=side[0]['training_stages_seconds'],stage2_training_seconds=side[1]['training_stages_seconds'],
            auxiliary_training_seconds=s1.get('preservation_probe_training_seconds',0),consistency_forward_seconds=s1.get('preservation_consistency_forward_seconds',0),auxiliary_scored_training_windows=s1.get('preservation_active_windows',0),
            auxiliary_checkpoint=s1.get('preservation_probe_checkpoint',''),stage1_selected_preservation_active=s1.get('preservation_selected_active',False),
            resource_record_paths=';'.join(resources['resource_record_paths']),gpu_uuid=side[0]['device']['uuid'],gpu_index=side[0]['device']['index'],
            predictor_initial_sha256=record['predictor_initial_sha256'],checkpoint_sha256=record['checkpoint_sha256'],stage1_checkpoint_sha256=record['stage1_checkpoint_sha256'],e23_config_sha256=record['e23_config_sha256'],result_path=str(cell/'three-test-draws.json'),test_result_sha256=digest(cell/'three-test-draws.json'),
            additional_test_seconds=json.loads((cell/'test_runtime.json').read_text())['seconds'],diagnostic_seconds=repair['diagnostic_seconds'],notes='Independent full100+100; original predictor freshly initialized; auxiliary only Stage1',
            **{k:resources[k] for k in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib','gpu')})
        for metric in ('mse','mae'):
            for stat in ('mean','std'):row['test_'+metric+'_'+stat]=tests['test_'+stat][metric]
        rows.append(row);groups.extend(dict(model=record['model'],**g) for g in tests['test_groups'])
    reference=next(r for r in rows if r['model']=='E23_v31_B')
    best_reference={'best_val_mse':float(reference['validation_mse']),'best_val_mae':float(reference['validation_mae'])}
    passed=[r for r in records if improves(r,best_reference)]
    selected=min(passed,key=lambda r:(r['best_val_mae'],r['best_val_mse'],r['training_resources']['full_workflow_seconds'],r['condition'])) if passed else None
    decision=dict(preservation_improves_control=improves(records[1],records[0]),retained_model=selected['model'] if selected else 'E23_v31_B',selection_source='fixed validation only',actual_matrix_training_full_seconds=sum(r['training_resources']['full_workflow_seconds'] for r in records))
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'ETTh1_P4_E23_v32_preservation_comparison_2026-10-03.csv',rows)
    write_csv(output/'ETTh1_P4_E23_v32_preservation_test_groups_2026-10-03.csv',groups)
    (root/'final-selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    return rows,decision
