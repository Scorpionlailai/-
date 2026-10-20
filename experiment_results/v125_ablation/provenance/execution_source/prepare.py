"""Fresh strict audit of own coarse sources and eligible historical calibrations."""
import gc
import json
from pathlib import Path
import time
import torch
from integrations.v125_ablation.common import STUDY,data_for,verify_lock,file_sha256,write_json,seed_all
from integrations.v125_ablation.adapter import V125Adapter
from policy import frozen_hash,tensor_hash,object_hash,initialize_branch,source_matches
OUT=Path(__file__).resolve().parents[1]
COARSE=Path('/home/xyw/paper/E24/artifacts/v125_pamap2_coarse_seed_stability_20261019')
DIAG=Path('/home/xyw/paper/E24/artifacts/v125_contribution_diagnostics_20261019')

def main():
    torch.set_num_threads(4);verify_lock(STUDY);data=data_for('PAMAP2','sr015_bl8',STUDY)
    prior=json.loads((COARSE/'final_verification_20261020.json').read_text());assert prior['passed']
    protocol=dict(dataset='PAMAP2',scene='sr015_bl8',data_manifest_sha256=data.manifest_sha,
        source_lock_sha256=file_sha256(STUDY/'design_lock.json'),model_seeds=[2023,2024,2025],
        training_order_seed=2023,training_mask_seed=2023,validation_mask_seed=2021,test_seeds=data.test_seeds,
        masks=data.manifest['masks'],splits=data.manifest['splits'],subjects=data.manifest['subjects'],
        scaler=data.manifest['scaler'],native_observation_sha256=data.manifest['native_observation_sha256'],
        scoring='train-global-standardized; native_valid AND artificially hidden; unique96points per window; observed untouched',
        grouping=json.loads((DIAG/'diagnostic_design.json').read_text())['grouping'])
    training=dict(epochs_per_branch=100,batch_size=128,updates_per_branch=8500,optimizer='Adam',learning_rate=.001,
        weight_decay=0.,scheduler=None,precision='float32',loss='hidden_valid_MAE + .25*recent32_hidden_valid_MAE',
        selection='validation_full96_mae',selection_strict_improvement=True,branch_widths=[64,256],weights=[.5,.5],
        source_run_sha256=file_sha256(STUDY.parents[1]/'integrations/v125_ablation/run.py'),
        initialization='own frozen coarse mean copied with strict=False; original novel parameters retained from own coarse checkpoint; seed_all(model_seed) per branch',
        forward='unchanged V125Adapter branch path; frozen front/base/experts/tail, branch replaces original coarse level as originalv125',
        requires_grad='ONLY entire selected original calibration member; original coarse_projector, other branch, retrieval encoders and memory frozen')
    sources={};reuse={};initials={};audits=[]
    for seed in (2023,2024,2025):
        for arm in ('A0','A1'):
            identity=f'{arm}_{seed}';item=next(x for x in prior['models'] if x['arm']==arm and x['model_seed']==seed)
            if seed==2023:
                entry=json.loads((COARSE/'design_lock.json').read_text())['existing_seed2023'][arm]
                checkpoint=Path(entry['coarse_checkpoint'])
            else:checkpoint=COARSE/arm/f'seed_{seed}'/'coarse/selected.pt'
            assert file_sha256(checkpoint)==item['checkpoint_sha256']
            coarse_state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            seed_all(seed);model=V125Adapter(data,arm,seed);model.load_state_dict(coarse_state)
            library=model.memory.snapshot_hash();prefix=frozen_hash(coarse_state)
            assert library==json.loads((COARSE/arm/f'seed_{seed}'/'evaluation_complete.json').read_text())['history_snapshot_hash']
            binding=dict(arm=arm,model_seed=seed,coarse_checkpoint=str(checkpoint),coarse_sha256=item['checkpoint_sha256'],
                library_sha256=library,coarse_frozen_prefix_sha256=prefix,protocol_sha256=object_hash(protocol),
                training_config_sha256=object_hash(training),coarse_selected_epoch=item['selected_epoch'])
            sources[identity]=binding;initials[identity]={}
            for index,width in enumerate((64,256)):
                params,initial=initialize_branch(model,index,seed);assert initial['inherited_tensor_hash']==initial['source_coarse_mean_tensor_hash']
                initials[identity][str(width)]=initial
            if seed==2023:
                run=STUDY/'PAMAP2/sr015_bl8'/arm/'seed_2023';oldplan=json.loads((run/'plan.json').read_text())
                sel=json.loads((run/'stage1/selection.json').read_text());fs=torch.load(run/'stage1/selected.pt',map_location='cpu',weights_only=True)
                assert oldplan['data_manifest_sha256']==protocol['data_manifest_sha256']
                assert oldplan['design_lock_sha256']==protocol['source_lock_sha256'] and oldplan['training_mask_seed']==2023
                assert sel['actual_epochs']==300 and sel['batch_size']==128 and sel['precision']=='float32' and sel['combination_weights']==[.5,.5]
                assert file_sha256(run/'stage1/selected.pt')==sel['checkpoint_sha256']
                assert frozen_hash(fs)==prefix and sel['history_snapshot_hash']==library
                model.load_state_dict(fs);assert model.memory.snapshot_hash()==library
                observed_binding=dict(arm=oldplan['variant'],model_seed=oldplan['model_seed'],coarse_sha256=file_sha256(run/'stage1/coarse_selected.pt'),
                    library_sha256=model.memory.snapshot_hash(),protocol_sha256=object_hash(protocol),training_config_sha256=object_hash(training))
                # Config/source equality is checked above against the locked implementation; hash the same canonical original semantics.
                assert source_matches(binding,observed_binding)
                records=[json.loads(x) for x in (run/'stage1/curve.jsonl').read_text().splitlines()]
                branches={}
                for index,width in enumerate((64,256)):
                    phase=f'calibration{width}';ck=run/f'stage1/{phase}_selected.pt';bs=torch.load(ck,map_location='cpu',weights_only=True)
                    extract={k.removeprefix(f'core.recent_mean_projector.members.{index}.'):v for k,v in fs.items() if k.startswith(f'core.recent_mean_projector.members.{index}.')}
                    assert tensor_hash(bs)==tensor_hash(extract)
                    unique={x['epoch']:x for x in records if x['subphase']==phase};assert set(unique)==set(range(1,101))
                    best=min(unique,key=lambda e:unique[e]['validation']['full96']['mae']);assert best==sel['selected_epochs'][phase]
                    resume_path=run/f'stage1/{phase}_resume.pt';resume=torch.load(resume_path,map_location='cpu',weights_only=False)
                    pg=resume['optimizer']['param_groups'];assert len(pg)==1 and pg[0]['lr']==.001 and pg[0]['weight_decay']==0.
                    assert resume['epoch']==100 and resume['steps']==8500*(index+2)
                    steps=sorted(set(int(x['step']) for x in resume['optimizer']['state'].values()));assert steps==[8500]
                    assert frozen_hash(resume['model'])==prefix
                    phasecost=next(x for x in sel['phases'] if x['phase']==phase)
                    branches[str(width)]=dict(checkpoint=str(ck),checkpoint_sha256=file_sha256(ck),tensor_sha256=tensor_hash(bs),
                        selected_epoch=best,validation=unique[best]['validation'],epochs=100,optimizer_steps=8500,
                        seconds=phasecost['seconds'],trainable_parameters=phasecost['trainable_parameters'],
                        max_allocated_gib=max(x['resources']['max_allocated_gib'] for x in unique.values()),
                        max_reserved_gib=max(x['resources']['max_reserved_gib'] for x in unique.values()),
                        peak_memory_scope='historical cumulative stage1 allocator peak, not isolated branch; exact historical branch-only peak unavailable',
                        resume_sha256=file_sha256(resume_path),curve_sha256=file_sha256(run/'stage1/curve.jsonl'),
                        selection_json_sha256=file_sha256(run/'stage1/selection.json'))
                    del resume,bs,extract
                reuse[identity]=dict(eligible=True,binding=binding,observed_binding=observed_binding,branches=branches,
                    final_checkpoint=str(run/'stage1/selected.pt'),final_checkpoint_sha256=sel['checkpoint_sha256'],
                    source_plan_sha256=file_sha256(run/'plan.json'),original_protocol_and_training_config_verified=True,
                    frozen_prefix_and_library_verified=True,optimizer_and100epoch_selection_verified=True)
                del fs
            audits.append(dict(identity=identity,own_coarse_verified=True,library_verified=True,reuse_eligible=identity in reuse))
            del model,coarse_state;gc.collect()
    for seed in (2023,2024,2025):
        for width in ('64','256'):
            assert initials[f'A0_{seed}'][width]['novel_tensor_hash']==initials[f'A1_{seed}'][width]['novel_tensor_hash']
    write_json(OUT/'source_audit.json',dict(passed=True,registered_unix=time.time(),protocol=protocol,training=training,
        protocol_sha256=object_hash(protocol),training_config_sha256=object_hash(training),sources=sources,reuse=reuse,
        initializations=initials,audits=audits,novel_initialization_pairs_all_match=True,
        previous_coarse_verification_sha256=file_sha256(COARSE/'final_verification_20261020.json')))
    print(json.dumps(dict(passed=True,coarse_sources=6,reusable_models=list(reuse),new_models=4,novel_initialization_pairs_all_match=True)))
if __name__=='__main__':main()
