"""Read locked v31 evidence, select on validation and account for shared Stage1."""
from __future__ import annotations
import csv,json,math,hashlib
from pathlib import Path
from integrations.e23.forecast_sensitive import retention_identity
from integrations.forecast_compare.resources import selected_training_resources
ROOT=Path(__file__).resolve().parents[1]
MODES={'A':'none','B':'uniform_v1','C':'forecast_gain_v1'}
CONFIG=ROOT/'configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked_metric(value):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('nonfinite metric/resource')
    return value


def validate_repair_diagnostic(record,repair):
    for key in ('checkpoint_sha256','selection_sha256','e23_config_sha256'):
        if repair.get(key)!=record[key]:raise ValueError('repair diagnostic identity mismatch')
    for key in ('source_csv_sha256','validation_mask_sha256'):
        if key in record and repair.get(key)!=record[key]:raise ValueError('repair diagnostic protocol mismatch')
    if repair.get('windows')!=2688 or repair.get('split')!='original_fixed_validation' or repair.get('metric_space')!='train_standardized' or repair.get('h_state_unchanged') is not True or repair.get('hidden_coordinates',0)<=0:raise ValueError('repair diagnostic scope mismatch')
    for key in ('global_mse','global_mae','diagnostic_seconds'):checked_metric(repair.get(key))


def choose_candidate(rows):
    passed=[]
    for row in rows:
        mse,mae=row['best_val_mse'],row['best_val_mae']
        if not all(math.isfinite(v) and v>=0 for v in (mse,mae,row['extra_training_seconds'])):
            raise ValueError('invalid validation metrics or resource time')
        if mse<.7499365438950035 and mae<=.5840456761470457:passed.append(row)
    return min(passed,key=lambda r:(r['best_val_mae'],r['best_val_mse'],r['extra_training_seconds'],list(MODES).index(r['condition']))) if passed else None


def load_candidate(root,condition):
    stage1=root/'common_stage1/runs/e23-stage1.json';selection=root/condition/'stage2/runs/e23-stage2.json'
    s1=json.loads(stage1.read_text());record=json.loads(selection.read_text())
    if Path(record['stage1_manifest']).resolve()!=stage1.resolve() or record['stage1_manifest_sha256']!=digest(stage1):raise ValueError('wrong shared Stage1')
    if record['stage1_checkpoint_sha256']!=s1['checkpoint_sha256']:raise ValueError('wrong Stage1 checkpoint')
    for item in (s1,record):
        if item['epochs_completed']!=100 or digest(item['checkpoint'])!=item['checkpoint_sha256'] or item['e23_config_sha256']!=digest(CONFIG):raise ValueError('training/checkpoint/config identity mismatch')
    expected=retention_identity(MODES[condition])
    if record.get('repair_retention','none')!=MODES[condition] or any(record.get(k)!=v for k,v in expected.items()):raise ValueError('wrong retention mode')
    if record.get('loss')!='forecast_mse_plus_0p2_mae' or record.get('selection_metric')!='mae':raise ValueError('wrong forecast objective')
    replay=json.loads((root/condition/'validation_replay.json').read_text())
    if replay['stage2_checkpoint_sha256']!=record['checkpoint_sha256'] or replay['windows']!=2688 or replay['coordinates']!=1806336:raise ValueError('wrong validation replay')
    for metric in ('mse','mae'):
        if abs(checked_metric(replay[metric])-checked_metric(record['best_val_'+metric]))>1e-6:raise ValueError('validation replay mismatch')
    for kind in ('key','bank'):
        if replay['h_'+kind+'_sha256_before']!=replay['h_'+kind+'_sha256_after']:raise ValueError('historical state changed')
    resources=selected_training_resources([stage1,selection])
    if resources['resource_status']!='measured_all_stages':raise ValueError('missing resources')
    for key in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib'):checked_metric(resources[key])
    sidecars=[json.loads(Path(p).read_text()) for p in resources['resource_record_paths']]
    return dict(condition=condition,model='E23_v31_'+condition,**record,training_resources=resources,
                shared_stage1_full_seconds=sidecars[0]['full_workflow_seconds'],stage2_full_seconds=sidecars[1]['full_workflow_seconds'],
                extra_training_seconds=checked_metric(sidecars[1]['full_workflow_seconds']),selection_sha256=digest(selection))


def validate_tests(record,result):
    from scripts.report_e23_v30_trial import MASK_BANK,MASK_BANK_SHA
    if result['checkpoint_sha256']!=record['checkpoint_sha256'] or result['selection_sha256']!=record['selection_sha256'] or result['e23_config_sha256']!=record['e23_config_sha256']:raise ValueError('test ownership mismatch')
    if result.get('repair_retention','none')!=record.get('repair_retention','none') or not result['h_state_unchanged']:raise ValueError('test training mode/state mismatch')
    if digest(MASK_BANK)!=MASK_BANK_SHA or result['test_mask_bank_sha256']!=MASK_BANK_SHA:raise ValueError('test mask bank changed')
    bank=json.loads(MASK_BANK.read_text())
    if len(result['test_groups'])!=3:raise ValueError('wrong test draw count')
    for actual,expected in zip(result['test_groups'],bank['groups']):
        if any(actual[k]!=v for k,v in expected.items()) or actual['windows']!=2688 or actual['target_elements']!=1806336:raise ValueError('test mask/count mismatch')
    for metric in ('mse','mae'):
        values=[g[metric] for g in result['test_groups']]
        if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('nonfinite test result')
        mean=sum(values)/3;std=math.sqrt(sum((v-mean)**2 for v in values)/3)
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in (result['test_mean'][metric],result['test_std'][metric])):raise ValueError('nonfinite test summary')
        if abs(result['test_mean'][metric]-mean)>1e-12 or abs(result['test_std'][metric]-std)>1e-12:raise ValueError('test summaries mismatch')
    if result['training_resources']!=record['training_resources']:raise ValueError('wrong test resources')


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def report(root,output):
    records=[load_candidate(root,c) for c in MODES]
    if len({r['predictor_initial_sha256'] for r in records})!=1:raise ValueError('different predictor initialization')
    winner=choose_candidate(records);rows=[];groups=[]
    old=list(csv.DictReader((ROOT/'docs/experiments/ETTh1_P4_E23_v28_v29_T1_comparison_2026-10-03.csv').open()))
    rows.extend(r for r in old if r['model'] in ('E23_v29_r1','T1'))
    for record in records:
        cell=root/record['condition'];result=json.loads((cell/'three-test-draws.json').read_text());validate_tests(record,result)
        repair=json.loads((cell/'repair_diagnostic.json').read_text());validate_repair_diagnostic(record,repair)
        row=dict(model=record['model'],condition=record['condition'],validation_mse=record['best_val_mse'],validation_mae=record['best_val_mae'],best_epoch=record['best_epoch'],stage2_epochs_completed=100,stage1_epochs_completed=100,
                 repair_retention=record.get('repair_retention','none'),selected_candidate=winner is not None and winner['condition']==record['condition'],shared_stage1_full_seconds=record['shared_stage1_full_seconds'],stage2_full_seconds=record['stage2_full_seconds'],
                 teacher_seconds=record.get('repair_retention_teacher_seconds',0.),repair_parameters=repair['repair_parameters'],predictor_parameters=repair['predictor_parameters'],diagnostic_seconds=repair['diagnostic_seconds'],weighted_training_windows=record.get('repair_retention_weighted_windows',0),training_windows=record.get('repair_retention_training_windows',0),predictor_initial_sha256=record['predictor_initial_sha256'],
                 stage1_checkpoint_sha256=record['stage1_checkpoint_sha256'],checkpoint_sha256=record['checkpoint_sha256'],validation_repair_mse=repair['global_mse'],validation_repair_mae=repair['global_mae'],
                 **{k:record['training_resources'][k] for k in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib','gpu')})
        for metric in ('mse','mae'):
            for stat in ('mean','std'):row['test_'+metric+'_'+stat]=result['test_'+stat][metric]
        rows.append(row);groups.extend(dict(model=record['model'],**g) for g in result['test_groups'])
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'ETTh1_P4_E23_v31_T1_comparison_2026-10-03.csv',rows);write_csv(output/'ETTh1_P4_E23_v31_test_groups_2026-10-03.csv',groups)
    decision=dict(winner=None if winner is None else winner['model'],retained_model='E23_v29_r1' if winner is None else winner['model'],selection_source='fixed validation only',actual_matrix_training_full_seconds=records[0]['shared_stage1_full_seconds']+sum(r['stage2_full_seconds'] for r in records),shared_stage1_trained_once=True)
    (root/'final-selection.json').write_text(json.dumps(decision,indent=2)+'\n');return rows,decision
