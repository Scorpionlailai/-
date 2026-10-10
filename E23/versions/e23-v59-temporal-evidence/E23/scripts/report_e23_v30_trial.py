"""Validate locked stack evidence and report accuracy and measured training costs."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections.abc import Mapping,Sequence
from pathlib import Path
from scripts.report_e23_v29_trial import _digest,_sha,_metric

ROOT=Path(__file__).resolve().parents[1]
MATRIX=((1,8),(4,1),(4,8))
REFERENCE_MSE=.7499365438950035
REFERENCE_MAE=.5840456761470457
MASK_BANK=ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json'
MASK_BANK_SHA='95fe49c78c6b884477e70c05f18a6c6c06f009e75d3afa01946275207b70e3a9'


def v30_gate_passed(record):
    return record['best_val_mse']<REFERENCE_MSE and record['best_val_mae']<=REFERENCE_MAE


def choose_v30_candidate(records:Sequence[Mapping],*,reference_mse:float=REFERENCE_MSE,reference_mae:float=REFERENCE_MAE)->Mapping|None:
    reference_mse,reference_mae=_metric(reference_mse),_metric(reference_mae)
    passed=[]
    for r in records:
        l,g=r.get('recovery_layers'),r.get('expert_groups')
        if (type(l) is not int or type(g) is not int or (l,g) not in MATRIX
            or r.get('stage')!='forecast' or r.get('recovery_schedule')!='stacked_parallel_tv_h_v1'
            or r.get('recovery_rounds')!=1 or r.get('group_width')!=64
            or r.get('group_fusion')!='gated_concat_v1' or type(r.get('activation_checkpointing')) is not bool
            or r.get('epochs_completed')!=100 or r.get('stage1_epochs_completed')!=100
            or r.get('selection_metric')!='mae' or r.get('loss')!='forecast_mse_plus_0p2_mae'
            or r.get('forecast_input')!='explicit_v1' or r.get('forecast_mixer')!='none'
            or r.get('history_adapter','none')!='none' or r.get('stage1_objective')!='global_mse_v1'
            or r.get('stage1_selection_metric_space')!='train_standardized'):
            raise ValueError('incomplete or incompatible v30 training identity')
        if not all(_sha(r.get(key)) for key in ('checkpoint_sha256','e23_config_sha256','stage1_checkpoint_sha256')):
            raise ValueError('missing checkpoint/config identity')
        mse,mae=_metric(r.get('best_val_mse')),_metric(r.get('best_val_mae'))
        parameters=r.get('repair_parameters')
        if type(parameters) is not int or parameters<1: raise ValueError('missing repair parameter count')
        replay=r.get('validation_replay')
        if (not isinstance(replay,Mapping) or replay.get('split')!='validation' or replay.get('windows')!=2688
            or replay.get('coordinates')!=1806336 or replay.get('stage2_checkpoint_sha256')!=r['checkpoint_sha256']
            or any(replay.get(key)!=r[key] for key in ('recovery_schedule','recovery_rounds','recovery_layers','expert_groups','group_width','group_fusion','activation_checkpointing'))):
            raise ValueError('validation replay identity mismatch')
        for kind in ('key','bank'):
            before,after=replay.get(f'h_{kind}_sha256_before'),replay.get(f'h_{kind}_sha256_after')
            if not _sha(before) or before!=after: raise ValueError('historical state changed during validation')
        if abs(_metric(replay.get('mse'))-mse)>1e-6 or abs(_metric(replay.get('mae'))-mae)>1e-6:
            raise ValueError('validation replay metric mismatch')
        if mse<reference_mse and mae<=reference_mae: passed.append(r)
    return min(passed,key=lambda r:(r['best_val_mae'],r['best_val_mse'],r['repair_parameters'],MATRIX.index((r['recovery_layers'],r['expert_groups'])))) if passed else None


def validate_v30_diagnostics(record:Mapping,diagnostic:Mapping)->None:
    if (diagnostic.get('split')!='original_fixed_validation' or diagnostic.get('metric_space')!='train_standardized'
        or diagnostic.get('diagnostic_only') is not True or any(diagnostic.get(key)!=record[key] for key in
        ('model','e23_config_sha256','recovery_layers','expert_groups'))):
        raise ValueError('stack diagnostic identity mismatch')
    for stage,sha in [('stage1',record['stage1_checkpoint_sha256']),('stage2',record['checkpoint_sha256'])]:
        result=diagnostic.get(stage)
        if (not isinstance(result,Mapping) or result.get('checkpoint_sha256')!=sha or result.get('windows')!=2688
            or not isinstance(result.get('layers'),list) or len(result['layers'])!=record['recovery_layers']):
            raise ValueError('stack diagnostic checkpoint or layer count mismatch')
        for kind in ('key','bank'):
            before,after=result.get(f'h_{kind}_sha256_before'),result.get(f'h_{kind}_sha256_after')
            if not _sha(before) or before!=after: raise ValueError('stack diagnostic historical state changed')
        for i,item in enumerate(result['layers'],1):
            if item.get('layer_index')!=i or len(item.get('mean_group_weights',[]))!=record['expert_groups']+1:
                raise ValueError('stack diagnostic group/layer geometry mismatch')
            _metric(item.get('global_mse'));_metric(item.get('global_mae'))
            for w in item['mean_group_weights']: _metric(w)
            if abs(sum(item['mean_group_weights'])-1)>1e-5: raise ValueError('invalid group weight sum')
            if stage=='stage2':
                for key in ('forecast_mse','forecast_mae'):
                    if key in item:_metric(item[key])


def load_candidate(root:Path,config:Path)->dict:
    from integrations.forecast_compare.resources import selected_training_resources
    root=root.resolve();config=config.resolve();s2_path=root/'stage2/runs/e23-stage2.json'
    r=json.loads(s2_path.read_text());s1_path=Path(r['stage1_manifest']);s1=json.loads(s1_path.read_text())
    if s1_path.resolve()!=root/'stage1/runs/e23-stage1.json': raise ValueError('trial requires its own fresh Stage1')
    freeze=root.parent/'code-freeze.json'
    if not freeze.is_file() or r.get('code_freeze_sha256')!=_digest(freeze):
        raise ValueError('formal training code freeze identity mismatch')
    for item in (s1,r):
        if _digest(Path(item['checkpoint']))!=item['checkpoint_sha256'] or _digest(config)!=item['e23_config_sha256']:
            raise ValueError('checkpoint/config hash changed')
    if _digest(s1_path)!=r['stage1_manifest_sha256'] or s1['checkpoint_sha256']!=r['stage1_checkpoint_sha256']:
        raise ValueError('paired Stage1 manifest identity mismatch')
    for key in ('recovery_schedule','recovery_rounds','recovery_layers','expert_groups','group_width','group_fusion','activation_checkpointing','code_freeze_sha256','key_sha256','memory_sha256'):
        if s1.get(key)!=r.get(key): raise ValueError('paired Stage1 '+key+' mismatch')
    diag=json.loads((root/'layer_diagnostics.json').read_text())
    r.update(root=str(root),config=str(config),model=f"E23_v30_l{r['recovery_layers']}_g{r['expert_groups']}",
        selection_manifest_sha256=_digest(s2_path),
        stage1_epochs_completed=s1['epochs_completed'],repair_parameters=diag['repair_parameters'],
        validation_replay=json.loads((root/'validation_replay.json').read_text()),
        training_resources=selected_training_resources([s1_path,s2_path]))
    if r['training_resources']['resource_status']!='measured_all_stages': raise ValueError('missing resource sidecars')
    validate_v30_diagnostics(r,diag);choose_v30_candidate([r]);return r


def write_selection(records,output):
    winner=choose_v30_candidate(records)
    result=dict(selection_source='original fixed validation only; test metrics never consumed',
        reference_mse=REFERENCE_MSE,reference_mae=REFERENCE_MAE,winner=None if winner is None else winner['model'],
        retained_model_if_no_winner='E23_v29_r1',candidates=records,
        original_v28_gate={r['model']:r['best_val_mse']<=.7402293330687785 and r['best_val_mae']<.5855096656989619 for r in records})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def _csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def validate_test_result(record,result):
    if result.get('checkpoint_sha256')!=record['checkpoint_sha256'] or result.get('e23_config_sha256')!=record['e23_config_sha256']:
        raise ValueError('test result checkpoint/config mismatch')
    if result.get('h_state_unchanged') is not True: raise ValueError('test changed H state')
    if (_digest(MASK_BANK)!=MASK_BANK_SHA or result.get('test_mask_bank_sha256')!=MASK_BANK_SHA
        or result.get('metric_space')!='train_standardized'
        or result.get('selection_sha256')!=record['selection_manifest_sha256']
        or any(result.get(key)!=record[key] for key in ('source_csv_sha256','validation_mask_sha256'))):
        raise ValueError('test result protocol/selection identity mismatch')
    groups=result.get('test_groups',[])
    if len(groups)!=3 or [g.get('seed') for g in groups]!=[2021,2026100201,2026100202]: raise ValueError('wrong frozen test draws')
    expected_groups=json.loads(MASK_BANK.read_text())['groups']
    for item,expected in zip(groups,expected_groups):
        if any(item.get(key)!=value for key,value in expected.items()):
            raise ValueError('test result missing-position group identity mismatch')
        if item.get('windows')!=2688 or item.get('target_elements')!=1806336: raise ValueError('test count mismatch')
        _metric(item.get('mse'));_metric(item.get('mae'))
    for metric in ('mse','mae'):
        values=[g[metric] for g in groups];mean=sum(values)/3;std=math.sqrt(sum((v-mean)**2 for v in values)/3)
        if abs(_metric(result['test_mean'][metric])-mean)>1e-12 or abs(_metric(result['test_std'][metric])-std)>1e-12: raise ValueError('test summary mismatch')
    if result.get('training_resources')!=record['training_resources']: raise ValueError('test resource ownership mismatch')


def report(root:Path,output_dir:Path):
    decision=json.loads((root/'validation_selection.json').read_text());records=decision['candidates']
    winner=choose_v30_candidate(records)
    if decision['winner']!=(None if winner is None else winner['model']):raise ValueError('saved validation decision changed')
    old=ROOT/'docs/experiments/ETTh1_P4_E23_v28_v29_T1_comparison_2026-10-03.csv'
    rows=list(csv.DictReader(old.open(newline='')))
    if {r['model'] for r in rows}!={'E23_v28','T1','E23_v29_r1','E23_v29_r3'}:raise ValueError('old comparison identity mismatch')
    groups=[];layers=[]
    for saved in records:
        r=load_candidate(Path(saved['root']),Path(saved['config']))
        if r!=saved:raise ValueError('candidate changed after locking')
        cell=Path(r['root']);diag=json.loads((cell/'layer_diagnostics.json').read_text());resources=r['training_resources']
        sidecars=[json.loads(Path(p).read_text()) for p in resources['resource_record_paths']]
        result_path=cell/'three-test-draws.json'; result=None
        if result_path.exists(): result=json.loads(result_path.read_text());validate_test_result(r,result)
        row=dict(dataset='ETTh1',scenario='P4',input_len=96,pred_len=96,model=r['model'],training_runs=1,
            test_draws=3 if result else '',selection_metric='fixed_validation_mae',validation_mse=r['best_val_mse'],
            validation_mae=r['best_val_mae'],status='completed' if result else 'locked_test_pending',
            test_mse_mean=result['test_mean']['mse'] if result else '',test_mae_mean=result['test_mean']['mae'] if result else '',
            test_mse_std=result['test_std']['mse'] if result else '',test_mae_std=result['test_std']['mae'] if result else '',
            recovery_schedule=r['recovery_schedule'],recovery_rounds=1,recovery_layers=r['recovery_layers'],expert_groups=r['expert_groups'],
            group_width=64,group_fusion=r['group_fusion'],activation_checkpointing=r['activation_checkpointing'],
            repair_parameters=r['repair_parameters'],checkpoint_sha256=r['checkpoint_sha256'],e23_config_sha256=r['e23_config_sha256'],
            stage1_epochs_completed=100,stage2_epochs_completed=100,stage1_objective=r['stage1_objective'],
            stage1_selection_metric_space='train_standardized',stage1_validation_repair_mse_global=diag['stage1']['layers'][-1]['global_mse'],
            stage1_validation_repair_mae_global=diag['stage1']['layers'][-1]['global_mae'],
            stage2_validation_repair_mse_global=diag['stage2']['layers'][-1]['global_mse'],
            stage2_validation_repair_mae_global=diag['stage2']['layers'][-1]['global_mae'],
            result_path=str(result_path.resolve()) if result else '',resource_record_paths=';'.join(resources['resource_record_paths']),
            training_seed=2021,stage1_batch_size=128,stage2_batch_size=128,stage1_precision='float32',stage2_precision='float32',
            validation_gate_passed=v30_gate_passed(r),selected_candidate=decision['winner']==r['model'],
            gpu_uuid=';'.join(sorted({str(s['device'].get('uuid',s['device'].get('gpu_uuid',''))) for s in sidecars})),
            gpu_index=sidecars[0]['device'].get('index',''),
            concurrent_conditions_path=str(cell/'formal_training_environment.json'),
            total_parameters=diag['total_parameters'],predictor_parameters=diag['predictor_parameters'],
            stage1_trainable_parameters=diag['stage1_trainable_parameters'],stage2_trainable_parameters=diag['stage2_trainable_parameters'],
            notes='Final-layer loss only; distinct parameters; fixed test positions; test/diagnostic time excluded',
            **{k:resources[k] for k in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib','gpu')})
        for i,stage in enumerate(('stage1','stage2')):
            row[f'{stage}_full_workflow_seconds']=sidecars[i]['full_workflow_seconds']
            row[f'{stage}_training_seconds']=sidecars[i]['training_stages_seconds']
            row[f'{stage}_peak_allocated_gib']=sidecars[i]['peak_allocated_bytes']/1024**3
            row[f'{stage}_peak_reserved_gib']=sidecars[i]['peak_reserved_bytes']/1024**3
            for item in diag[stage]['layers']:
                layers.append(dict(model=r['model'],stage=stage,checkpoint_sha256=diag[stage]['checkpoint_sha256'],
                    **{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in item.items()}))
        rows.append(row)
        if result:
            for item in result['test_groups']: groups.append(dict(model=r['model'],checkpoint_sha256=r['checkpoint_sha256'],**item))
    output_dir.mkdir(parents=True,exist_ok=True)
    _csv(output_dir/'ETTh1_P4_E23_v30_T1_comparison_2026-10-03.csv',rows)
    if groups:_csv(output_dir/'ETTh1_P4_E23_v30_test_groups_2026-10-03.csv',groups)
    _csv(output_dir/'ETTh1_P4_E23_v30_layer_diagnostics_2026-10-03.csv',layers)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/experiments');args=parser.parse_args()
    report(args.root,args.output_dir)

if __name__=='__main__': main()
