"""Audit stage ownership and protocol, then report the fixed v33 matrix."""
import csv
import json
from pathlib import Path
from integrations.forecast_compare.resources import selected_training_resources
from scripts.report_e23_v31_trial import digest, checked_metric, validate_tests, write_csv

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = { 'C0': ROOT/'configs/e23_multi_expert_p4_width64_iterative_tv_h_r3_v29.yaml',
            'A': ROOT/'configs/e23_multi_expert_p4_round_supervision_v33.yaml',
            'B': ROOT/'configs/e23_multi_expert_p4_refreshed_rounds_v33.yaml' }
PREDICTOR_INITIAL_SHA = '18ad18b5af292a9b526233ea426f81ffb3899d1f4fc850441060843d8adc404a'


def load_candidate(root, condition):
    cell = root/condition
    paths = [cell/s/'runs'/f'e23-{s}.json' for s in ('stage1','stage2')]
    s1, s2 = [json.loads(p.read_text()) for p in paths]
    schedule = 'iterative_tv_h_refresh_v1' if condition == 'B' else 'iterative_tv_h_v1'
    for record in (s1, s2):
        expected = dict(epochs_completed=100, e23_config_sha256=digest(CONFIGS[condition]),
            stage1_objective='global_mse_v1' if condition == 'C0' else 'global_round_mse_v1',
            recovery_schedule=schedule, recovery_rounds=3, batch_size=128, seed=2021)
        if condition != 'C0': expected['stage1_round_weights'] = [.2,.3,.5]
        if any(record.get(k) != v for k,v in expected.items()):
            raise ValueError('v33 stage training/config identity mismatch')
        if digest(record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('v33 checkpoint checksum mismatch')
    expected = dict(loss='forecast_mse_only', selection_metric='mae', forecast_input='explicit_v1',
                    forecast_mixer='none', predictor_initial_sha256=PREDICTOR_INITIAL_SHA,
                    optimizer_state_restored=False, rng_reset_seed=2021)
    if any(s2.get(k) != v for k,v in expected.items()) or s2.get('repair_retention','none') != 'none':
        raise ValueError('v33 prediction head/training protocol differs from control')
    if (Path(s2['stage1_manifest']).resolve() != paths[0].resolve()
        or s2['stage1_manifest_sha256'] != digest(paths[0])
        or s2['stage1_checkpoint_sha256'] != s1['checkpoint_sha256']):
        raise ValueError('v33 Stage1 parent identity mismatch')
    replay = json.loads((cell/'validation_replay.json').read_text())
    if replay['windows'] != 2688 or replay['coordinates'] != 1806336 or replay['stage2_checkpoint_sha256'] != s2['checkpoint_sha256']:
        raise ValueError('v33 validation replay ownership/count mismatch')
    for metric in ('mse','mae'):
        if abs(replay[metric] - checked_metric(s2['best_val_'+metric])) > 1e-6:
            raise ValueError('v33 validation replay value mismatch')
    for name in ('key','bank'):
        if replay['h_'+name+'_sha256_before'] != replay['h_'+name+'_sha256_after']:
            raise ValueError('v33 historical state changed')
    diagnostic = json.loads((cell/'round_diagnostics.json').read_text())
    for stage, record in zip(('stage1','stage2'), (s1,s2)):
        d = diagnostic[stage]
        if d['checkpoint_sha256'] != record['checkpoint_sha256'] or d['windows'] != 2688 or d['hidden_coordinates'] != 1023670:
            raise ValueError('v33 diagnostic ownership/count mismatch')
        if d['h_key_sha256_before'] != d['h_key_sha256_after'] or d['h_bank_sha256_before'] != d['h_bank_sha256_after']:
            raise ValueError('v33 diagnostic historical state changed')
    if diagnostic['stage2']['predictor_parameters'] != 35427:
        raise ValueError('v33 predictor parameter count changed')
    resources = selected_training_resources(paths)
    if resources['resource_status'] != 'measured_all_stages': raise ValueError('missing v33 resources')
    return dict(**s2, model='E23_v33_'+condition, condition=condition, selection_sha256=digest(paths[1]),
                stage1_record=s1, training_resources=resources)


def improves(row, reference):
    return row['best_val_mse'] < float(reference['validation_mse']) and row['best_val_mae'] <= float(reference['validation_mae'])


def report(root, output):
    records = [load_candidate(root,c) for c in CONFIGS]
    control = next(r for r in records if r['condition']=='C0')
    with (ROOT/'docs/experiments/ETTh1_P4_E23_v32_preservation_comparison_2026-10-03.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['model'] in ('T1','E23_v31_B','E23_v32_preservation_A')]
    baseline = next(r for r in rows if r['model'] == 'E23_v32_preservation_A')
    best = next(r for r in rows if r['model'] == 'E23_v31_B')
    groups, diagnostics = [], []
    for record in records:
        cell = root/record['condition']; s1 = record['stage1_record']; resource = record['training_resources']
        tests = json.loads((cell/'three-test-draws.json').read_text()); validate_tests(record, tests)
        diag = json.loads((cell/'round_diagnostics.json').read_text())
        side = [json.loads(Path(p).read_text()) for p in resource['resource_record_paths']]
        row = dict(model=record['model'], condition=record['condition'], dataset='ETTh1', scenario='P4', input_len=96, pred_len=96,
            training_runs=1, test_draws=3, selection_metric='fixed_validation_mae', validation_mse=record['best_val_mse'],
            validation_mae=record['best_val_mae'], stage1_best_epoch=s1['best_epoch'], stage2_best_epoch=record['best_epoch'],
            stage1_epochs_completed=100, stage2_epochs_completed=100, stage1_objective=s1['stage1_objective'],
            stage1_round_weights='' if record['condition']=='C0' else '0.2;0.3;0.5',
            stage1_reused=record['condition']=='C0', recovery_schedule=record['recovery_schedule'], recovery_rounds=3,
            stage2_objective='forecast_mse_only', stage1_batch_size=128, stage2_batch_size=128,
            stage1_precision='float32', stage2_precision='float32', training_seed=2021,
            stage1_validation_repair_mse_global=diag['stage1']['rounds'][-1]['global_mse'],
            stage2_validation_repair_mse_global=diag['stage2']['rounds'][-1]['global_mse'],
            stage1_full_workflow_seconds=side[0]['full_workflow_seconds'], stage2_full_workflow_seconds=side[1]['full_workflow_seconds'],
            repair_parameters=diag['stage2']['repair_parameters'], predictor_parameters=35427,
            predictor_initial_sha256=record['predictor_initial_sha256'], checkpoint_sha256=record['checkpoint_sha256'],
            stage1_checkpoint_sha256=s1['checkpoint_sha256'], e23_config_sha256=record['e23_config_sha256'],
            resource_record_paths=';'.join(resource['resource_record_paths']), gpu_uuid=side[0]['device']['uuid'], gpu_index=side[0]['device']['index'],
            result_path=str(cell/'three-test-draws.json'), test_result_sha256=digest(cell/'three-test-draws.json'),
            validation_improves_matched_control=improves(record,baseline), validation_improves_best=improves(record,best),
            validation_improves_three_round_control=improves(record,
                dict(validation_mse=control['best_val_mse'],validation_mae=control['best_val_mae'])),
            actual_new_training_seconds=side[1]['full_workflow_seconds'] if record['condition']=='C0' else resource['full_workflow_seconds'],
            additional_test_seconds=json.loads((cell/'test_runtime.json').read_text())['seconds'],
            diagnostic_seconds=sum(diag[s]['diagnostic_seconds'] for s in ('stage1','stage2')),
            notes=('Saved v29-r3 Stage1 reused; new pure-MSE Stage2 100 epochs' if record['condition']=='C0' else
                   'Fresh100+100; Stage1-only supervision; original predictor and pure-MSE Stage2 unchanged'),
            **{k:resource[k] for k in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib','gpu')})
        for metric in ('mse','mae'):
            for stat in ('mean','std'): row['test_'+metric+'_'+stat] = tests['test_'+stat][metric]
        rows.append(row); groups.extend(dict(model=record['model'], **g) for g in tests['test_groups'])
        for stage in ('stage1','stage2'):
            diagnostics.extend(dict(model=record['model'],stage=stage,**r) for r in diag[stage]['rounds'])
    candidates = [r for r in records if improves(r,best)]
    selected = min(candidates,key=lambda r:(r['best_val_mae'],r['best_val_mse'],r['condition'])) if candidates else None
    decision = dict(retained_model=selected['model'] if selected else 'E23_v31_B',
        selection_source='original fixed validation only; tests not used',
        matched_control_passed=[r['model'] for r in records if improves(r,baseline)],
        actual_matrix_training_full_seconds=sum(r['training_resources']['full_workflow_seconds'] for r in records
            if r['condition'] != 'C0') + float(next(r for r in rows if r['model']=='E23_v33_C0')['stage2_full_workflow_seconds']))
    output.mkdir(exist_ok=True,parents=True)
    write_csv(output/'ETTh1_P4_E23_v33_comparison_2026-10-04.csv',rows)
    write_csv(output/'ETTh1_P4_E23_v33_test_groups_2026-10-04.csv',groups)
    write_csv(output/'ETTh1_P4_E23_v33_round_diagnostics_2026-10-04.csv',diagnostics)
    (root/'final-selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    return rows, decision
