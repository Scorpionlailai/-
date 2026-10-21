"""Final read-only asset/update/input barrier checks plus independent summary checks."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from policy3 import OUT,COARSE,CAL,BASE,training_modules
from run3 import check_lock,require_actual_barrier
from integrations.v125_ablation.common import file_sha256,write_json
training_modules()
from policy import frozen_hash,tensor_hash


def verify():
    audit=check_lock();barrier=require_actual_barrier();sources=json.loads((OUT/'calibration_sources.json').read_text());predictions=0;strata=0;paired_epochs=0
    base=list(csv.DictReader(open(OUT/'baseline_tables/per_mask.csv')))
    index={(int(x['training_seed']),x['split'],int(x['mask_seed']),x['phase'],x['section']):x for x in base if x['variant']=='A0'}
    for seed in (2023,2024,2025):
        folder=OUT/'A4'/f'seed_{seed}';done=json.loads((folder/'evaluation_complete.json').read_text());assert done['complete'] and len(done['results'])==84
        binding=sources['sources'][f'A4_{seed}'];final=barrier['details'][f'A4_{seed}_ensemble']
        state=torch.load(final['checkpoint'],map_location='cpu',weights_only=True)
        assert frozen_hash(state)==binding['coarse_frozen_prefix_sha256']
        assert torch.equal(state['core.recent_mean_projector.weights'],torch.tensor([.5,.5],dtype=torch.float64))
        for phase in ('coarse','calibration64','calibration256'):
            s=barrier['details'][f'A4_{seed}_{phase}'];assert s['epochs']==100 and s['optimizer_steps']==8500
            assert file_sha256(Path(s['checkpoint']))==s['checkpoint_sha256']
            if phase!='coarse':
                member_index=0 if phase=='calibration64' else 1
                prefix=f'core.recent_mean_projector.members.{member_index}.'
                member=torch.load(s['checkpoint'],map_location='cpu',weights_only=True)
                assert tensor_hash(member)==tensor_hash({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)})
                del member
            if seed==2023:continue
            stage=folder/phase
            init=json.loads((folder/'initialization.json').read_text())
            cs=torch.load(folder/'coarse/selected.pt',map_location='cpu',weights_only=True)
            for name,digest in init.items():
                if name.startswith(('core.temporal.','core.router.residual.temporal.','core.router.scales.temporal.','core.point_router.quality.temporal.','core.point_candidate_readout.quality.temporal.')):
                    import hashlib
                    assert hashlib.sha256(cs[name].numpy().tobytes()).hexdigest()==digest
            if phase!='coarse':
                payload=torch.load(stage/'resume.pt',map_location='cpu',weights_only=False)
                assert payload['steps']==8500 and payload['epoch']==100 and set(int(x['step']) for x in payload['optimizer']['state'].values())=={8500}
                assert frozen_hash(payload['model'])==binding['coarse_frozen_prefix_sha256'];del payload
            reference=COARSE/'A0'/f'seed_{seed}'/phase if phase=='coarse' else CAL/'A0'/f'seed_{seed}'/phase
            epochs=[json.loads((stage/f'epoch_{epoch:03}.json').read_text()) for epoch in range(1,101)]
            metric='mse' if phase=='coarse' else 'mae';assert min(epochs,key=lambda x:x['validation']['full96'][metric])['epoch']==s['selected_epoch']
            for row in epochs:
                old=json.loads((reference/f"epoch_{row['epoch']:03}.json").read_text())
                for key in ('train_window_order_sha256','train_actual_observation_mask_sha256','optimizer_steps'):assert row[key]==old[key]
                if phase!='coarse':assert row['history_snapshot_hash']==binding['library_sha256'] and row['frozen_tensor_sha256']==s['frozen_tensor_sha256']
                paired_epochs+=1
            if phase!='coarse':
                newinit=json.loads((stage/'initialization.json').read_text())
                oldinit=json.loads((CAL/'A0'/f'seed_{seed}'/phase/'initialization.json').read_text())
                assert newinit['novel_tensor_hash']==oldinit['novel_tensor_hash']
        for entry in done['results']:
            p=Path(entry);r=json.loads(p.read_text());b,m=r['binding'],r['metrics']
            assert b['all_checkpoint_lock_sha256']==file_sha256(OUT/'all_checkpoint_lock.json')
            assert p.stat().st_mtime>=barrier['locked_unix'],'evaluation before global checkpoint lock'
            assert b['coarse_checkpoint_sha256']==binding['coarse_sha256'] and b['history_snapshot_hash']==binding['library_sha256']
            assert file_sha256(p.parent/'predictions.npy')==m['predictions_sha256'];predictions+=1
            assert m['observed_unchanged'] and m['state_unchanged'] and len(r['strata'])==32;strata+=32
            for section in ('full96','recent32'):
                old=index[seed,b['split'],b['mask_seed'],b['phase'],section]
                assert old['mask_sha256']==b['mask_sha256'] and int(old['coordinates'])==m[section]['coordinates']
                rows=[x for x in r['strata'] if x['section']==section];allrow=next(x for x in rows if x['family']=='all')
                for family in ('invisible_run','history','window_visible','other_variables_visible'):
                    parts=[x for x in rows if x['family']==family];assert sum(x['coordinates'] for x in parts)==allrow['coordinates']
                    for k in ('absolute_error_sum','squared_error_sum'):assert np.isclose(sum(x[k] for x in parts),allrow[k],rtol=1e-12)
    assert predictions==252 and strata==8064 and paired_epochs==600
    capacity=json.loads((OUT/'capacity.json').read_text());assert len(capacity)==24 and json.loads((OUT/'capacity_complete.json').read_text())['complete']
    assert len({(x['resources']['gpu'],x['resources']['cuda_visible_devices'],x['resources']['cpu_threads']) for x in capacity})==1
    for x in capacity:assert len(x['forward_seconds'])==5 and x['batch_size']==128 and x['allocated_parameters']>=x['inference_invoked_parameters']
    from export3 import build,DEST
    status=build();assert status['complete']
    mask=list(csv.DictReader(open(DEST/'per_mask.csv')));seeds=list(csv.DictReader(open(DEST/'per_seed.csv')));summary=list(csv.DictReader(open(DEST/'summary.csv')))
    assert len(mask)==1512 and len(seeds)==144
    for x in seeds:
        if x['variant'] not in ('A0','A4'):continue
        take=[y for y in mask if all(x[k]==y[k] for k in ('variant','training_seed','split','phase','section'))];assert len(take)==(1 if x['split']=='validation' else 20)
        for m in ('mse','mae'):assert abs(float(x[m])-np.mean([float(y[m]) for y in take]))<1e-13
    for x in summary:
        if x['variant'] not in ('A0','A4'):continue
        take=[y for y in seeds if all(x[k]==y[k] for k in ('variant','split','phase','section'))];assert len(take)==3
        for m in ('mse','mae'):
            values=[float(y[m]) for y in take];assert abs(float(x[m+'_mean'])-np.mean(values))<1e-13 and abs(float(x[m+'_training_std_ddof1'])-np.std(values,ddof=1))<1e-13
    pairs=list(csv.DictReader(open(DEST/'round3_paired_differences.csv')))
    import itertools
    expected_pairs=set(itertools.product((2023,2024,2025),('validation','test'),('coarse','calibration64','calibration256','ensemble'),('full96','recent32'),('mse','mae')))
    actual_pairs=[(int(x['training_seed']),x['split'],x['phase'],x['section'],x['metric']) for x in pairs]
    assert len(pairs)==len(expected_pairs)==96 and set(actual_pairs)==expected_pairs and len(set(actual_pairs))==len(actual_pairs)
    for x in pairs:assert abs(float(x['A4_minus_A0'])-(float(x['A4_error'])-float(x['A0_error'])))<1e-13
    # Baseline A0/A1 numeric observations remain identical after the append.
    published={(x['variant'],x['training_seed'],x['split'],x['mask_seed'],x['phase'],x['section']):x for x in mask}
    for old in base:
        current=published[old['variant'],old['training_seed'],old['split'],old['mask_seed'],old['phase'],old['section']]
        for m in ('mse','mae','coordinates','checkpoint_sha256','library_sha256'):assert current[m]==old[m]
    result=dict(passed=True,verified_unix=time.time(),baseline_commit=BASE,A4_prediction_hashes_verified=predictions,A4_stratified_rows_verified=strata,
        A0_A4_actual_paired_epochs_verified=paired_epochs,all_checkpoints_before_new_test_verified=True,
        historical_A4_seed2023_audit_passed=True,baseline_A0_A1_results_unchanged=True,own_frozen_prefixes_libraries_verified=True,
        source_initializations_and_T_disabled_parameters_verified=True,selection_and_actual_updates_verified=True,
        aggregation_and_A4minusA0_sign_independently_verified=True,inference_capacity_paths_verified=24,
        no_Stage2_or_other_variants=True,audit_sha256=file_sha256(OUT/'audit.json'),all_checkpoint_lock_sha256=file_sha256(OUT/'all_checkpoint_lock.json'))
    write_json(OUT/'final_verification.json',result);return result
if __name__=='__main__':print(json.dumps(verify()))
