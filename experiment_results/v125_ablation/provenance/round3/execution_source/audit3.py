"""Fresh third-round source/config/update audit and runtime no-T controls."""
from contextlib import ExitStack
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
from policy3 import OUT,REPO,BASE,COARSE,CAL,DIAG,training_modules
from integrations.v125_ablation.common import STUDY,verify_lock,data_for,file_sha256,write_json,seed_all,parameter_hashes,state_sha
from integrations.v125_ablation.adapter import V125Adapter
training_modules()
from policy import frozen_hash,tensor_hash,object_hash


def isolation(model,data,device,path):
    model.inference_path=path;model.requires_grad_(False).eval()
    truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device)
    before=state_sha(model);calls={};states={}
    def record(name,original):
        def f(*args,**kwargs):
            calls[name]=calls.get(name,0)+1
            if name in ('V_prepare','V_message'):states[name]=args[0].base_latents.detach().clone()
            return original(*args,**kwargs)
        return f
    with torch.no_grad(),ExitStack() as stack:
        for name in ('prepare_sources','from_cache','forward'):
            stack.enter_context(patch.object(model.core.temporal,name,side_effect=AssertionError('disabled T method called: '+name)))
        for label,obj,name in [('front',model.core.front_context,'forward'),('H_prepare',model.core.historical,'prepare_candidates'),
                ('H_message',model.core.historical,'from_cache'),('V_prepare',model.core.variable,'prepare_sources'),('V_message',model.core.variable,'from_cache')]:
            stack.enter_context(patch.object(obj,name,side_effect=record(label,getattr(obj,name))))
        result,_,_=model.output(torch.where(obs,truth,0.),obs,starts)
        t=result.evidence['temporal'];assert not t.message.any() and not t.quality.any() and not t.available.any()
        assert not t.candidates.valid.any() and not t.candidates.weights.any() and not t.candidates.messages.any()
        assert not result.router.deltas['temporal'].any() and not result.router.available[...,3].any()
        attention=result.point_candidate_attention;offset=1+sum(attention['candidate_counts'][:2])
        assert not attention['weights'][...,offset:].any() and not attention['valid'][...,sum(attention['candidate_counts'][:2]):].any()
        for iteration in result.iterations:
            assert not iteration.preview_gates[...,0].any() and not iteration.allowed[...,2].any()
        assert torch.equal(states['V_prepare'],states['V_message']),'V source state still changed through T'
        assert all(calls.get(k,0)>0 for k in ('front','H_prepare','H_message','V_prepare','V_message'))
        assert torch.equal(result.explicit.sequence_observed,obs)
        # Poison all T-specific learned parameters: complete output must remain identical.
        saved={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith('core.temporal.') or
            name.startswith(('core.router.residual.temporal.','core.router.scales.temporal.','core.point_router.quality.temporal.','core.point_candidate_readout.quality.temporal.'))}
        try:
            for name,p in model.named_parameters():
                if name in saved:p.fill_(123.)
            poisoned,_,_=model.output(torch.where(obs,truth,0.),obs,starts)
            assert torch.equal(poisoned.explicit.standardized_sequence,result.explicit.standardized_sequence)
        finally:
            for name,p in model.named_parameters():
                if name in saved:p.copy_(saved[name])
    assert before==state_sha(model)
    return dict(path=path,passed=True,T_methods_never_called=True,T_messages_quality_candidates_deltas_zero=True,
        serial_T_update_zero_and_V_state_unchanged=True,T_point_candidate_attention_exactly_zero=True,public_front_and_HV_calls=calls,
        poisoned_T_parameters_output_unchanged=True,checkpoint_state_unchanged=True,validation_batch=128)


def main():
    torch.set_num_threads(4);verify_lock(STUDY)
    assert subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()==BASE
    base=OUT/'baseline_tables';base.mkdir(exist_ok=True)
    names=['README.md','protocol.json','summary.csv','per_seed.csv','per_mask.csv','stratified.csv','resources.csv',
        'position_dispersion.csv','contributions_per_seed.csv','contributions_summary.csv','stratified_per_seed.csv','stratified_summary.csv','stratified_contributions_per_seed.csv']
    files={}
    for name in names:
        b=subprocess.run(['git','show',BASE+':experiment_results/v125_ablation/'+name],cwd=REPO,capture_output=True,check=True).stdout
        (base/name).write_bytes(b);files[name]=file_sha256(base/name)
    data=data_for('PAMAP2','sr015_bl8',STUDY);old_audit=json.loads((DIAG/'asset_and_training_audit.json').read_text())
    previous=next(x for x in old_audit['arms'] if x['dataset']=='PAMAP2' and x['arm']=='A4');assert previous['status']=='passed'
    run=STUDY/'PAMAP2/sr015_bl8/A4/seed_2023';plan=json.loads((run/'plan.json').read_text());sel=json.loads((run/'stage1/selection.json').read_text())
    assert plan['variant']=='A4' and plan['model_seed']==2023 and plan['training_mask_seed']==2023
    assert plan['data_manifest_sha256']==data.manifest_sha and plan['design_lock_sha256']==file_sha256(STUDY/'design_lock.json')
    assert sel['actual_epochs']==300 and sel['optimizer_steps']==25500 and sel['batch_size']==128 and sel['precision']=='float32' and sel['combination_weights']==[.5,.5]
    cs=torch.load(run/'stage1/coarse_selected.pt',map_location='cpu',weights_only=True);fs=torch.load(run/'stage1/selected.pt',map_location='cpu',weights_only=True)
    assert file_sha256(run/'stage1/coarse_selected.pt')==previous['coarse_sha256']
    assert file_sha256(run/'stage1/selected.pt')==sel['checkpoint_sha256']==previous['checkpoint_sha256']
    assert frozen_hash(cs)==frozen_hash(fs)==previous['shared_coarse_and_memory_tensor_hash']
    raw=[json.loads(x) for x in (run/'stage1/curve.jsonl').read_text().splitlines()];phases={}
    for index,phase in enumerate(('coarse','calibration64','calibration256')):
        rows={x['epoch']:x for x in raw if x['subphase']==phase};assert set(rows)==set(range(1,101))
        metric='mse' if index==0 else 'mae';best=min(rows,key=lambda e:rows[e]['validation']['full96'][metric]);assert best==sel['selected_epochs'][phase]
        ck=run/f'stage1/{phase}_selected.pt';resume_path=run/f'stage1/{phase}_resume.pt'
        payload=torch.load(resume_path,map_location='cpu',weights_only=False)
        assert payload['epoch']==100 and payload['steps']==8500*(index+1)
        assert all(x['lr']==.001 and x['weight_decay']==0. for x in payload['optimizer']['param_groups'])
        assert set(int(x['step']) for x in payload['optimizer']['state'].values()) <= {8415,8500} if index==0 else set(int(x['step']) for x in payload['optimizer']['state'].values())=={8500}
        if index:
            branch=torch.load(ck,map_location='cpu',weights_only=True);prefix=f'core.recent_mean_projector.members.{index-1}.'
            assert tensor_hash(branch)==tensor_hash({k.removeprefix(prefix):v for k,v in fs.items() if k.startswith(prefix)})
            assert frozen_hash(payload['model'])==frozen_hash(cs)
        cost=next(x for x in sel['phases'] if x['phase']==phase)
        phases[phase]=dict(checkpoint=str(ck),checkpoint_sha256=file_sha256(ck),selected_epoch=best,selection_rule='validation_full96_'+metric,
            validation=rows[best]['validation'],epochs=100,optimizer_steps=8500,training_seconds=cost['seconds'],trainable_parameters=cost['trainable_parameters'],
            resources=dict(max_allocated_gib=max(x['resources']['max_allocated_gib'] for x in rows.values()),max_reserved_gib=max(x['resources']['max_reserved_gib'] for x in rows.values()),
                peak_memory_scope='historical cumulative stage1 allocator; independent phase peak unavailable'),
            resume_sha256=file_sha256(resume_path),curve_sha256=file_sha256(run/'stage1/curve.jsonl'))
        del payload;gc.collect()
    cal_audit=json.loads((CAL/'source_audit.json').read_text());a0={};initials={}
    for seed in (2023,2024,2025):
        identity=f'A0_{seed}';binding=cal_audit['sources'][identity]
        assert file_sha256(Path(binding['coarse_checkpoint']))==binding['coarse_sha256']
        f=CAL/'A0'/f'seed_{seed}';prov=json.loads((f/'selected_provenance.json').read_text())
        assert file_sha256(f/'selected.pt')==prov['checkpoint_sha256']
        members={}
        for width in (64,256):
            s=json.loads((f/f'calibration{width}/selection.json').read_text());assert file_sha256(Path(s['checkpoint']))==s['checkpoint_sha256'];members[str(width)]=s
        a0[str(seed)]=dict(source_binding=binding,selected_checkpoint=str(f/'selected.pt'),selected_checkpoint_sha256=prov['checkpoint_sha256'],branches=members,
            coarse_evaluation=str(COARSE/'A0'/f'seed_{seed}'/'evaluation'),calibration_evaluation=str(f/'evaluation'))
        seed_all(seed);model=V125Adapter(data,'A4',seed);initial=parameter_hashes(model)
        a0_initial=json.loads(((STUDY/'PAMAP2/sr015_bl8/A0/seed_2023/plan.json') if seed==2023 else (COARSE/'A0'/f'seed_{seed}'/'initialization.json')).read_text())
        if seed==2023:a0_initial=a0_initial['initialization'];assert initial==plan['initialization']
        assert initial==a0_initial;initials[str(seed)]=initial
        assert model.disabled==frozenset(('temporal',))
        assert model.core.front_context is not None and model.core.iterative is not None
        assert model.core.temporal_evidence_refiner is None,'unaccounted T-specific post refiner'
        assert model.core.iterative.parallel==False and model.core.iterative.neutral_quality==False and model.core.point_candidate_readout.patch_shared==False
        del model;gc.collect()
    device=torch.device('cuda:0');seed_all(2023);m=V125Adapter(data,'A4',2023).to(device)
    runtime=[];m.load_state_dict(cs);m.epoch=phases['coarse']['selected_epoch']
    assert m.memory.snapshot_hash()==sel['history_snapshot_hash'];runtime.append(isolation(m,data,device,'coarse'))
    m.load_state_dict(fs);m.epoch=0
    for path in ('calibration64','calibration256','ensemble'):runtime.append(isolation(m,data,device,path))
    assert m.memory.snapshot_hash()==sel['history_snapshot_hash']
    # Confirm initialization and gradient/update isolation on one real training batch (discarded; not a formal epoch).
    del m;gc.collect();torch.cuda.empty_cache();seed_all(2024);m=V125Adapter(data,'A4',2024).to(device)
    from integrations.v125_ablation.run import configure_phase
    params=configure_phase(m,'coarse');optimizer=torch.optim.Adam(params,lr=.001,weight_decay=0.)
    truth,obs,starts=next(data.batches('train',128,epoch=0,seed=2023));truth,obs=truth.to(device),obs.to(device);m.epoch=1;m.batch_index=1
    m.core.historical.retrieval.begin_epoch(1);m.train()
    with ExitStack() as stack:
        for name in ('prepare_sources','from_cache','forward'):stack.enter_context(patch.object(m.core.temporal,name,side_effect=AssertionError('T training method called')))
        loss=m.loss(truth,obs,starts);assert torch.isfinite(loss);loss.backward()
        assert not any(p.grad is not None for p in m.core.temporal.parameters())
        for module in (m.core.front_context,m.core.variable,m.core.historical):assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
        optimizer.step();m.after_step()
    write_json(OUT/'audit.json',dict(passed=True,registered_unix=time.time(),baseline_commit=BASE,baseline_tables_sha256=files,
        data_manifest_sha256=data.manifest_sha,protocol=json.loads((base/'protocol.json').read_text()),
        original_source_lock_sha256=file_sha256(STUDY/'design_lock.json'),A0=a0,initial_parameter_hashes=initials,
        A4_seed2023=dict(eligible=True,run=str(run),phases=phases,final_checkpoint=str(run/'stage1/selected.pt'),final_checkpoint_sha256=sel['checkpoint_sha256'],
            coarse_frozen_prefix_sha256=frozen_hash(cs),history_snapshot_hash=sel['history_snapshot_hash'],plan_sha256=file_sha256(run/'plan.json'),
            original_protocol_config_verified=True,coarse_and_calibration_frozen_prefix_verified=True),runtime_isolation=runtime,
        real_training_update=dict(passed=True,batch_size=128,temporal_gradients_absent=True,public_front_HV_gradients_present=True,loss=float(loss.detach())),
        previous_readonly_audit_sha256=file_sha256(DIAG/'asset_and_training_audit.json'),
        historical_input_pairing_evidence='2023 source/config identical and shared data generator; historical actual input fingerprints were not recorded;2024/2025 require per-epoch actual fingerprints',
        deletion_capacity_note='modules remain allocated to preserve random initialization; T routes disabled/frozen, trainable and invoked capacity/time reported; no matched-capacity causal claim'))
    print(json.dumps(dict(passed=True,A0_models_locked=3,A4_2023_reusable_phases=list(phases),T_isolation_paths=4,paired_initializations=3,real_batch128_HV_front_update_passed=True)))
if __name__=='__main__':main()
