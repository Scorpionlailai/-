"""Validate a completed Weather workflow and export its final comparison row."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    status=json.loads((run/'status.json').read_text())
    if status['status']!='completed':raise ValueError('workflow has not completed')
    plan=json.loads((run/'plan.json').read_text())
    for path,digest in plan['source_sha256'].items():
        if file_hash(path)!=digest:raise ValueError('sealed source changed: '+path)
    stage1=json.loads((run/'stage1/runs/e23-stage1.json').read_text())
    stage2=json.loads((run/'stage2/runs/e23-stage2.json').read_text())
    assert stage1['epochs_completed']==stage2['epochs_completed']==100
    assert stage1['optimizer_updates']==28600
    assert stage1['training_windows_per_epoch']==36608
    assert stage2['repair_update_policy']=='frozen_v1'
    assert stage2['repair_trainable_parameters']==0 and stage2['repair_training_mode']=='eval'
    assert stage2['loss']=='forecast_mse_only'
    assert stage2['selection_metric']=='mae'
    preflight=json.loads(Path(plan['preflight_acceptance']).read_text())
    assert stage2['predictor_initial_sha256']==preflight['predictor_initial_sha256']
    updates={}
    for name,tag in [('stage1','repair'),('stage2','forecast')]:
        log=(run/(name+'.log')).read_text()
        endings=re.findall(rf'stage={tag} epoch=(\d+)/100 batch=286/286 ',log)
        assert list(map(int,endings))==list(range(1,101)), (name,endings)
        updates[name]=28600
    stage_times={x['stage']:x['wall_seconds'] for x in status['stages']}
    entries=[]
    originals=[]
    for suffix in ('mae','mse'):
        path=run/f'evaluation_{suffix}/twenty-test-draws.json'
        result=json.loads(path.read_text())
        assert result['repair_frozen_verified'] and result['visible_values_exact']
        assert len(result['draws'])==20 and all(x['windows']==10444 for x in result['draws'])
        resources=result['training_resources']
        assert resources['resource_status']=='measured_all_stages'
        selected_path=Path(result['selection_manifest'])
        assert file_hash(selected_path)==result['selection_manifest_sha256']
        row=dict(dataset='Weather',model='E24-v79',protocol_id=plan['protocol_id'],
                 start_rate=.15,block_len=8,seq_len=96,pred_len=96,n_vars=21,
                 training_seed=2023,selection_metric=suffix,primary=suffix=='mae',
                 stage1_best_epoch=stage1['best_epoch'],stage2_best_epoch=result['best_epoch'],
                 stage1_epochs=100,stage2_epochs=100,stage1_optimizer_updates=updates['stage1'],
                 stage2_optimizer_updates=updates['stage2'],test_masks=20,test_windows_per_mask=10444,
                 actual_window_missing_fraction_mean=sum(x['window_missing_fraction'] for x in result['draws'])/20,
                 stage1_validation_repair_mse=stage1['best_val_repair_mse'],
                 stage1_validation_repair_mae=stage1.get('validation_repair_metrics',{}).get('global_mae'),
                 stage2_loss='pure_mse',stage2_repair_update='fully_frozen',
                 stage1_full_command_seconds=stage_times['stage1'],stage2_full_command_seconds=stage_times['stage2'],
                 full_training_workflow_seconds=stage_times['stage1']+stage_times['stage2'],
                 sidecar_full_workflow_seconds=resources['full_workflow_seconds'],
                 peak_allocated_GiB=resources['peak_allocated_gib'],peak_reserved_GiB=resources['peak_reserved_gib'],
                 gpu=resources['gpu'],training_batch_size=128,precision='FP32',
                 training_seed_std=None,drr=None,checkpoint_sha256=result['checkpoint_sha256'])
        for metric,value in result['summary'].items():
            if metric!='coordinate_weighted':
                row[metric]=value['mean'];row[metric+'_mask_std_ddof1']=value['mask_std_ddof1']
        entries.append(row)
        originals.append(dict(path=str(path),sha256=file_hash(path)))
    # Both Stage2 selections must expose exactly the same frozen repairs.
    for key in ('repair96_mse','repair96_mae','repair32_mse','repair32_mae'):
        assert abs(entries[0][key]-entries[1][key])<1e-12
    report=ROOT/'docs/experiments/Weather_sr015_bl8_E24_v79_seed2023_results.csv'
    with report.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    acceptance=dict(status='passed',workflow_status=str(run/'status.json'),
                    run=str(run),source_identity_verified=True,repair_frozen_verified=True,
                    stage1_epochs=100,stage2_epochs=100,optimizer_updates=updates,
                    evaluation_paths=originals,summary_csv=str(report),summary_csv_sha256=file_hash(report),
                    analysis_source_sha256=file_hash(Path(__file__)),
                    timing_scope='Controller child process launch to completed exit for Stage1 and Stage2, including entrypoint imports and resource-sidecar writing; excluding shared-mask-fixture generation, code adaptation/preflight, interprocess transition and final tests.',
                    sidecar_timing_scope='Inherited command resource clock starts when integrations.e23.run imports; some E24-entry imports occur before it. Retained separately; final full timing uses controller command durations.',
                    rows=entries)
    with (run/'final_acceptance.json').open('x') as f:json.dump(acceptance,f,indent=2);f.write('\n')
    print(json.dumps(acceptance),flush=True)


if __name__=='__main__':main()
