"""Verify selectors, frozen identities, test supports and resource deduplication."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from publish import collect,csv_write,ROOT,DOCS,PREFIX

WORK=Path('/home/xyw/paper/.e23-v42-mse/E23')
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    completed=read(ROOT/'matrix-completed.json');assert completed['status']=='complete'
    definition=read(ROOT/'matrix-definition.json')
    for rel,h in definition['sources'].items():assert digest(WORK/rel)==h,rel
    for p,h in definition['protected'].items():assert digest(p)==h,p
    assert digest(ROOT/'plan.json')==definition['plan_sha256']
    rows,draws,actual=collect();assert all(r['status']=='tested' for r in rows)
    bank=ROOT.parent/'e23_t1_five_seed_standard_20261004/test_masks/manifest.json'
    entries=read(bank)['groups'];assert digest(bank)==definition['test_bank_sha256']
    for entry in entries:
        path=Path(entry['path'])
        assert digest(path if path.is_absolute() else bank.parent/path)==entry['sha256']
    logs={};checked=[]
    for row in rows:
        cell=ROOT/row['trial'];t1=row['model']=='T1';metric=row['checkpoint_selection_metric']
        manifest=Path(row['selection_manifest']);record=read(manifest)
        assert digest(record['checkpoint'])==record['checkpoint_sha256']
        assert row['stage2_epochs']==100 and row['prediction_head_parameters']==35427
        if t1:
            assert record['training']['loss']=='forecast_mse' and record['training']['selection_metric']==metric
            first=read(record['repair_selection_manifest'])
            assert first['seed']==502 and first['epochs_completed']==136
            assert first['upstream_source_sha256']==record['upstream_source_sha256']
            log=cell/'training.log'
        else:
            assert record['loss']=='forecast_mse_only' and record.get('repair_retention','none')=='none'
            assert record['selection_metric']==metric and record['downstream_protocol']=='t1_mse_v1'
            assert record['forecast_padding']=='replicate_last_value_v1' and record['learning_rate']==1e-4
            first=read(record['stage1_manifest']);assert digest(record['stage1_manifest'])==record['stage1_manifest_sha256']
            assert row['stage1_optimizer_updates']==first.get('optimizer_updates',row['stage1_optimizer_updates'])
            assert row['stage1_optimizer_updates'] in (6600,52800)
            commands=read(cell/'training-commands.json');log=cell/f'train_{len(commands)-1}.log'
        if str(log) not in logs:
            validations=[]
            for line in log.read_text().splitlines():
                if t1 and line.startswith('{'):
                    d=json.loads(line)
                    if d.get('stage')=='forecast' and 'validation' in d:
                        validations.append((d['epoch'],d['validation']['mse'],d['validation']['mae']))
                elif not t1:
                    m=re.fullmatch(r'stage=forecast epoch=(\d+)/100 val_mse=([\d.e+-]+) val_mae=([\d.e+-]+) best_epoch=\d+',line)
                    if m:validations.append(tuple([int(m[1]),float(m[2]),float(m[3])]))
            assert [v[0] for v in validations]==list(range(1,101))
            logs[str(log)]=validations
        assert digest(first['checkpoint'])==first['checkpoint_sha256']
        index=1 if metric=='mse' else 2
        epoch=min(logs[str(log)],key=lambda v:v[index])[0]
        assert epoch==row['checkpoint_epoch'],(row['trial'],metric,epoch,row['checkpoint_epoch'])
        result=read(row['test_result']);assert result['predictor_parameters']==35427
        assert result['checkpoint_selection_metric']==metric and result['training_seed']==row['training_seed']
        assert result['checkpoint_sha256']==row['checkpoint_sha256']
        assert result['test_mask_bank_sha256']==definition['test_bank_sha256']
        lock=read(Path(row['test_result']).parent/'validation-lock.json');assert not lock['test_metrics_seen']
        assert len(result['test_draws'])==20
        for entry,d in zip(entries,result['test_draws']):
            assert (entry['seed'],entry['sha256'])==(d['mask_seed'],d['mask_sha256'])
            assert d['windows']==2688 and d['target_elements']==2688*96*7
            assert digest(d['window_metrics'])==d['window_metrics_sha256']
            with np.load(d['window_metrics']) as arrays:
                assert set(arrays.files)=={'mse','mae','target_start'}
                assert np.array_equal(arrays['target_start'],np.arange(11520,14208))
                assert np.isfinite(arrays['mse']).all() and np.isfinite(arrays['mae']).all()
                for k in ['mse','mae']:assert abs(float(arrays[k].mean())-d[k])<1e-6
        for k in ['mse','mae']:
            assert abs(np.mean([d[k] for d in result['test_draws']])-result['test_mean'][k])<1e-12
        assert result['training_resources']['resource_status']=='measured_all_stages'
        assert row['full_training_seconds']>0 and row['actual_new_training_seconds']>0
        checked.append(dict(trial=row['trial'],seed=row['training_seed'],selector=metric,checkpoint_epoch=epoch,draws=20))
    assert len(rows)==2*len(logs)
    choices=read(ROOT/'seed-validation-choice.json');plan=read(ROOT/'plan.json')
    assert set(map(int,choices['seed_cells']))==set(plan['phase3']['seeds'])
    for metric in ['mae','mse']:
        selected=min(plan['phase3']['seeds'],key=lambda s:(choices['validation_scores'][str(s)][metric][metric],
            choices['validation_scores'][str(s)][metric]['mse' if metric=='mae' else 'mae'],plan['phase3']['seeds'].index(s)))
        assert choices['selected_seeds'][metric]==selected
    assert not choices['seed_search_test_results_seen']
    for cell in sorted(set(r['trial'] for r in rows)):
        pair=[r for r in rows if r['trial']==cell];assert len(pair)==2
        assert pair[0]['full_training_seconds']==pair[1]['full_training_seconds']
        assert pair[0]['actual_new_training_seconds']==pair[1]['actual_new_training_seconds']
    audit=dict(status='passed',frozen_source_files=len(definition['sources']),protected_existing_files=len(definition['protected']),
        all_prediction_heads_original=True,all_stage2_losses_pure_mse=True,training_trajectories=len(logs),
        selection_rows=len(rows),test_draws=len(draws),unique_new_training_resource_records=len(actual),
        actual_new_training_seconds=sum(actual.values()),selected_seeds=choices['selected_seeds'],
        new_T1_stage1_training=False,new_T1_stage2_trajectories=1,checks=checked)
    (ROOT/'integrity-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (DOCS/(PREFIX+'_audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
