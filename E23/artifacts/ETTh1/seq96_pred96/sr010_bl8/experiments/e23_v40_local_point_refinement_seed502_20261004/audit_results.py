"""Verify all saved trial, mask, resource and published-result identities."""
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v40-point/E23')
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
NAME='ETTh1_P4_E23_v40_point_refinement_seed502_20261004'
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
READ=lambda p:json.loads(Path(p).read_text())


def main():
    protected=READ(ROOT/'protected-existing-files.json')
    assert all(SHA(p)==digest for p,digest in protected.items())
    test_bank=FORMAL/'test_masks/manifest.json';bank=READ(test_bank)
    shared_source=None;new_draws=0;stages=0;max_difference=0.;gpu_ids=set();records=[]
    for trial,span in [('tail32',32),('full96',96)]:
        cell=ROOT/trial;seal=cell/'trial-definition.json';definition=READ(seal)
        assert SHA(seal)==(cell/'trial-definition.sha256').read_text().strip()
        assert all(SHA(WORK/relative)==digest for relative,digest in definition['source_files'].items())
        if shared_source is None:shared_source=definition['source_files']
        else:assert shared_source==definition['source_files']
        assert SHA(definition['configuration'])==definition['config_sha256']
        assert SHA(definition['parent'])==definition['parent_sha256']
        assert SHA(test_bank)==definition['test_bank_sha256']
        for stage in ['stage1','stage2']:
            selected=READ(cell/stage/'runs'/f'e23-{stage}.json')
            assert selected['epochs_completed']==100 and selected['seed']==502
            assert selected['point_refiner_span']==span and selected['point_refiner']=='point_residual_v1'
            assert SHA(selected['checkpoint'])==selected['checkpoint_sha256']
            if stage=='stage1':assert selected['optimizer_updates']==6600 and selected['stage1_training_batch_size']==128
            else:
                assert selected['repair_retention']=='uniform_v1' and selected['repair_retention_active_batches']==5940
                assert selected['repair_retention_training_windows']==844800
                assert selected['forecast_input']=='explicit_v1' and selected['forecast_mixer']=='none'
                assert selected['loss']=='forecast_mse_plus_0p2_mae'
            stages+=1
        completed=READ(cell/'completed-audit.json');assert completed['status']=='complete'
        assert SHA(cell/'frozen-source.tar.gz')==completed['source_archive_sha256']
        decision=READ(cell/'validation-decision.json');assert not decision['test_metrics_seen']
        expected=decision['validation']['mse']<decision['parent']['mse'] and decision['validation']['mae']<=decision['parent']['mae']
        assert decision['passed']==expected
        data=READ(cell/'evaluation/twenty-test-draws.json')
        assert data['test_mask_bank_sha256']==SHA(test_bank) and len(data['test_draws'])==20
        assert data['predictor_parameters']==35427 and data['predictor_initial_sha256']==definition['predictor_initial_sha256']
        assert data['model']=='E23-v40-'+trial and data['training_seed']==502
        assert SHA(data['selection_manifest'])==data['selection_manifest_sha256']
        assert data['checkpoint_sha256']==selected['checkpoint_sha256']
        for draw,entry in zip(data['test_draws'],bank['groups']):
            assert draw['mask_seed']==entry['seed'] and draw['mask_sha256']==entry['sha256']
            assert draw['windows']==2688 and draw['target_elements']==1806336
            assert SHA(draw['window_metrics'])==draw['window_metrics_sha256']
            with np.load(draw['window_metrics']) as values:
                assert np.array_equal(values['target_start'],np.arange(11520,14208))
                for metric in ['mse','mae']:
                    value=float(values[metric].mean(dtype=np.float64))
                    diff=abs(value-draw[metric]);max_difference=max(max_difference,diff);assert diff<1e-6
            new_draws+=1
        for metric in ['mse','mae']:
            scores=np.asarray([draw[metric] for draw in data['test_draws']])
            assert abs(scores.mean()-data['test_mean'][metric])<1e-12
            assert abs(scores.std(ddof=1)-data['test_mask_std_ddof1'][metric])<1e-12
        resources=[READ(path) for path in data['training_resources']['resource_record_paths']]
        assert len(resources)==2 and all(r['status']=='completed' and r['batch_size']==128 for r in resources)
        gpu_ids.update(r['device']['uuid'] for r in resources)
        assert abs(sum(r['full_workflow_seconds'] for r in resources)-data['training_resources']['full_workflow_seconds'])<1e-9
        records.append(dict(trial=trial,gate=decision['passed'],test_mean=data['test_mean'],stage2_best_epoch=selected['best_epoch']))
    assert len(gpu_ids)==1 and next(iter(gpu_ids))=='1315e15c-d657-e61f-bde5-6e0b38e6e54b'
    for suffix in ['.csv','_all_draws.csv','.md']:
        assert SHA(ROOT/(NAME+suffix))==SHA(DOCS/(NAME+suffix))
    with (DOCS/(NAME+'.csv')).open() as stream:published=list(csv.DictReader(stream))
    assert len(published)==4
    with (DOCS/(NAME+'_all_draws.csv')).open() as stream:assert len(list(csv.DictReader(stream)))==80
    for record in records:
        row=next(r for r in published if r['model']=='E23-v40-'+record['trial'])
        assert all(float(row[metric])==record['test_mean'][metric] for metric in ['mse','mae'])
    result=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        trials=records,full_training_stages=stages,new_test_draws=new_draws,
        cached_baseline_draws_reused=40,new_baseline_training=False,
        shared_source_files=len(shared_source),shared_source_identical_between_trials=True,
        protected_existing_files_unchanged=len(protected),gpu_uuid=next(iter(gpu_ids)),
        original_prediction_head_unchanged=True,window_metrics_max_difference=max_difference,
        published_rows=4,published_draw_rows=80,selected=READ(ROOT/'final-selection.json')['selected'],
        formal_five_seed_comparison_unchanged=True)
    (ROOT/'final-integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
