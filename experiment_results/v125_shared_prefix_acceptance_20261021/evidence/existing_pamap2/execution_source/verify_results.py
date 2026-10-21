"""Fail closed before publication: sources, actual updates, outputs and hierarchical summaries."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from integrations.v125_ablation.common import file_sha256,write_json
from calibration_worker import OUT,COARSE,check_lock
from policy import frozen_hash,tensor_hash,source_matches


def verify():
    audit=check_lock();models=[];predictions=0;strata_count=0;paired_epochs=0;resources=[]
    for arm in ('A0','A1'):
        for seed in (2023,2024,2025):
            folder=OUT/arm/f'seed_{seed}';binding=audit['sources'][f'{arm}_{seed}']
            assert file_sha256(Path(binding['coarse_checkpoint']))==binding['coarse_sha256']
            done=json.loads((folder/'evaluation_complete.json').read_text());assert done['complete'] and len(done['results'])==63
            final=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
            assert frozen_hash(final)==binding['coarse_frozen_prefix_sha256']
            provenance=json.loads((folder/'selected_provenance.json').read_text())
            assert file_sha256(folder/'selected.pt')==provenance['checkpoint_sha256'] and tensor_hash(final)==provenance['tensor_sha256']
            assert torch.equal(final['core.recent_mean_projector.weights'],torch.tensor([.5,.5],dtype=torch.float64))
            for index,width in enumerate((64,256)):
                stage=folder/f'calibration{width}';selection=json.loads((stage/'selection.json').read_text())
                assert source_matches(binding,selection['source_binding']) and selection['epochs']==100 and selection['optimizer_steps']==8500
                assert selection['frozen_state_unchanged'] and file_sha256(Path(selection['checkpoint']))==selection['checkpoint_sha256']
                bs=torch.load(selection['checkpoint'],map_location='cpu',weights_only=True)
                prefix=f'core.recent_mean_projector.members.{index}.'
                assert tensor_hash(bs)==tensor_hash({k.removeprefix(prefix):v for k,v in final.items() if k.startswith(prefix)})
                if not selection['reused']:
                    epochs=[json.loads((stage/f'epoch_{epoch:03}.json').read_text()) for epoch in range(1,101)]
                    assert min(epochs,key=lambda x:x['validation']['full96']['mae'])['epoch']==selection['selected_epoch']
                    assert all(x['optimizer_steps']==85*x['epoch'] and x['history_snapshot_hash']==binding['library_sha256'] and x['frozen_tensor_sha256']==selection['frozen_tensor_sha256'] for x in epochs)
                    payload=torch.load(stage/'resume.pt',map_location='cpu',weights_only=False)
                    assert payload['steps']==8500 and payload['epoch']==100
                    assert set(int(v['step']) for v in payload['optimizer']['state'].values())=={8500}
                    assert frozen_hash(payload['model'])==binding['coarse_frozen_prefix_sha256']
                    resources.append(dict(arm=arm,seed=seed,width=width,seconds=selection['training_seconds'],updates=8500))
                    del payload
                del bs
            for path in done['results']:
                p=Path(path);result=json.loads(p.read_text());b=result['binding'];m=result['metrics']
                assert b['arm']==arm and b['model_seed']==seed and b['coarse_checkpoint_sha256']==binding['coarse_sha256'] and b['history_snapshot_hash']==binding['library_sha256']
                assert m['state_unchanged'] and m['observed_unchanged']
                assert file_sha256(p.parent/'predictions.npy')==m['predictions_sha256'];predictions+=1
                assert len(result['strata'])==32;strata_count+=32
                old=json.loads((COARSE/arm/f'seed_{seed}'/'evaluation'/f"{b['split']}_{b['mask_seed']}"/'result.json').read_text())
                assert b['mask_sha256']==old['binding']['mask_sha256']
                for section in ('full96','recent32'):
                    assert m[section]['coordinates']==old['metrics'][section]['coordinates']
                    rows=[x for x in result['strata'] if x['section']==section];allrow=next(x for x in rows if x['family']=='all')
                    for family in ('invisible_run','history','window_visible','other_variables_visible'):
                        parts=[x for x in rows if x['family']==family]
                        assert sum(x['coordinates'] for x in parts)==allrow['coordinates']
                        for key in ('squared_error_sum','absolute_error_sum'):assert np.isclose(sum(x[key] for x in parts),allrow[key],rtol=1e-12)
            for split,seed_mask in [('validation',2021)]+[('test',s) for s in audit['protocol']['test_seeds']]:
                combination=json.loads((folder/'evaluation'/f'combination_check_{split}_{seed_mask}.json').read_text());assert combination['passed'] and combination['weights']==[.5,.5]
            models.append(dict(arm=arm,model_seed=seed,coarse_sha256=binding['coarse_sha256'],library_sha256=binding['library_sha256'],paths_verified=3,test_positions=20))
            del final
    for seed in (2024,2025):
        for width in (64,256):
            pair=[OUT/arm/f'seed_{seed}'/f'calibration{width}' for arm in ('A0','A1')]
            initials=[json.loads((p/'initialization.json').read_text()) for p in pair]
            assert initials[0]['novel_tensor_hash']==initials[1]['novel_tensor_hash']
            for epoch in range(1,101):
                records=[json.loads((p/f'epoch_{epoch:03}.json').read_text()) for p in pair]
                for key in ('train_window_order_sha256','train_actual_observation_mask_sha256','optimizer_steps'):assert records[0][key]==records[1][key]
                # The masks/order also match the corresponding frozen coarse study's original training schedule.
                coarse=json.loads((COARSE/'A0'/f'seed_{seed}'/'coarse'/f'epoch_{epoch:03}.json').read_text())
                for key in ('train_window_order_sha256','train_actual_observation_mask_sha256'):assert records[0][key]==coarse[key]
                paired_epochs+=1
    assert predictions==378 and strata_count==12096 and paired_epochs==400 and len(resources)==8
    # Independently check published hierarchical means and SD against per-mask rows.
    from export_results import build,DEST
    state=build(False);assert state['complete']
    with (DEST/'per_mask.csv').open() as f:permask=list(csv.DictReader(f))
    with (DEST/'per_seed.csv').open() as f:perseed=list(csv.DictReader(f))
    with (DEST/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    assert len(perseed)==96 and len(permask)==1008
    for row in perseed:
        take=[x for x in permask if all(x[k]==row[k] for k in ('variant','training_seed','split','phase','section'))]
        assert len(take)==(1 if row['split']=='validation' else 20)
        for key in ('mse','mae'):assert np.isclose(float(row[key]),np.mean([float(x[key]) for x in take]),rtol=1e-13)
    for row in summary:
        take=[x for x in perseed if all(x[k]==row[k] for k in ('variant','split','phase','section'))];assert len(take)==3
        for key in ('mse','mae'):
            values=[float(x[key]) for x in take]
            assert np.isclose(float(row[key+'_mean']),np.mean(values),rtol=1e-13)
            assert np.isclose(float(row[key+'_training_std_ddof1']),np.std(values,ddof=1),rtol=1e-13)
    result=dict(passed=True,verified_unix=time.time(),models=models,calibrated_prediction_hashes_verified=predictions,
        calibration_stratified_rows_verified=strata_count,total_all_path_stratified_rows=state['stratified_rows'],
        paired_epoch_inputs_verified=paired_epochs,new_training_branches=8,updates_per_branch=8500,
        training_gpu_hours=sum(x['seconds'] for x in resources)/3600,original_half_half_combination_verified=True,
        frozen_prefix_and_library_hashes_verified=True,validation_only_selection_verified=True,aggregation_independently_recomputed=True,
        no_stage2_or_other_variants=True,source_audit_sha256=file_sha256(OUT/'source_audit.json'))
    write_json(OUT/'final_verification.json',result);return result
if __name__=='__main__':print(json.dumps(verify()))
