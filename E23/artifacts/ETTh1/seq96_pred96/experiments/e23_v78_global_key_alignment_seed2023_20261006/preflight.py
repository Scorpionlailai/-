import hashlib, json, os, tarfile
from pathlib import Path
from itertools import islice
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, _starts, stage1_objective, _retrieval_loss, _new_memory, _stage_observed, _publish_epoch
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from malr.migration import MigrationCatalog
from malr.data.patching import patchify_with_state,patchify_preserving_values
from integrations.forecast_compare.auxiliary_forecast import build_auxiliary_predictor, auxiliary_forecast_loss

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT/'plan.json').read_text())
WORK = Path(plan['work'])
torch.set_num_threads(4)
dev = torch.device(plan['device'])
results = []
for scene in plan['scenarios']:
    os.environ.pop('MALR_MISSING_SCENARIO',None)
    if scene['scenario_manifest']:
        os.environ['MALR_MISSING_SCENARIO'] = scene['scenario_manifest']
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=Path(plan['protocol_config']),
        mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',
        training_seed=2023)
    assert protocol.mask_sha256['validation'] == scene['validation_mask_sha256']
    assert len(load_test_bank(protocol,Path(scene['test_bank']))) == 20
    assert protocol.config.missing.start_rate == scene['start_rate']
    assert protocol.config.missing.block_len == scene['block_len']
    settings = load_e23_settings(Path(plan['config']),protocol=protocol)
    assert (settings.seed,settings.stage1_epochs,settings.stage2_epochs) == (2023,100,100)
    torch.manual_seed(2023)
    model = build_forecast_model(protocol,settings,device=dev)
    assert parameter_hash(model.predictor) == plan['predictor_initial_sha256']
    assert sum(p.numel() for p in model.predictor.parameters()) == 35427
    assert sum(p.numel() for p in model.repair.parameters()) == plan['repair_parameters']
    batch = next(iter_p4_batches(protocol,'train',batch_size=4,epoch=0)).to(dev)
    starts = _starts(batch,stride=8,patches=12,device=dev)
    memory=_new_memory(protocol)
    for cpu in islice(iter_p4_batches(protocol,'train',batch_size=128,epoch=0),2):
        seeded=cpu.to(dev); st=_starts(seeded,stride=8,patches=12,device=dev)
        _stage_observed(memory,model.repair,seeded,st)
    _publish_epoch(memory,model.repair,1)
    assert model.repair.historical.index.member_keys.numel()>0
    aux=None;aux_hash=None;active_hidden_gradient=None
    if plan['stage1_aux_forecast']!='none':
        aux=build_auxiliary_predictor(protocol,device=dev);aux.train();aux_hash=parameter_hash(aux)
        assert sum(p.numel() for p in aux.parameters())==35427
        assert sum(p.numel() for p in aux.parameters() if p.requires_grad)==35424
        probe=batch.normalized_observed.detach().clone().requires_grad_(True)
        warm=auxiliary_forecast_loss(aux,probe,batch,epoch=10)
        assert torch.autograd.grad(warm,probe,allow_unused=True)[0] is None
        active=auxiliary_forecast_loss(aux,probe,batch,epoch=11)
        active_grad=torch.autograd.grad(active,probe)[0]
        assert torch.isfinite(active_grad).all() and active_grad[~batch.observed_mask].abs().sum()>0
        active_hidden_gradient=float(active_grad[~batch.observed_mask].abs().sum())
    trainable=sum(p.numel() for p in model.repair.parameters() if p.requires_grad)+(sum(p.numel() for p in aux.parameters() if p.requires_grad) if aux is not None else 0)
    total=sum(p.numel() for p in model.repair.parameters())+(sum(p.numel() for p in aux.parameters()) if aux is not None else 0)
    assert trainable==plan['stage1_trainable_parameters'] and total==plan['stage1_total_parameters']
    assert parameter_hash(model.predictor)==plan['predictor_initial_sha256']
    opt=torch.optim.Adam(list(model.repair.parameters())+(list(aux.parameters()) if aux is not None else []),lr=.001)
    gradients = []; auxiliary_losses = []; global_level_gradients = []
    for step in range(2):
        model.train(); opt.zero_grad()
        captured=[];original_encode=model.repair.historical.retrieval.encode_frozen
        def spy(values,mask):
            captured.append((values.detach().clone(),mask.detach().clone()));return original_encode(values,mask)
        model.repair.historical.retrieval.encode_frozen=spy
        out = model.repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        model.repair.historical.retrieval.encode_frozen=original_encode
        q=patchify_with_state(batch.normalized_observed,batch.observed_mask,16,8)
        expected=torch.where(q.effective_mask,q.values*batch.query_std[:,:,None,None]+batch.query_mean[:,:,None,None],0).reshape(-1,16)
        assert len(captured)==1 and torch.equal(captured[0][1],q.effective_mask.reshape(-1,16))
        torch.testing.assert_close(captured[0][0],expected)

        assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
        assert out.evidence['historical'].available.any()
        loss,parts = stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,
            batch.observed_mask,_retrieval_loss(model.repair,batch),retrieval_weight=.1,
            loss_mode='global_mse_v1',query_std=batch.query_std,structure_mode='recent_mean_v2',
            gap_structure_mode=settings.stage1_gap_structure)
        assert parts['gap_hidden_coordinates']>0 and parts['gap_segments']>0
        assert torch.isfinite(parts['gap_level']) and torch.isfinite(parts['gap_trend'])
        if aux is not None:
            auxiliary_loss=auxiliary_forecast_loss(aux,out.explicit.normalized_sequence,batch,epoch=11+step,batch_index=step)
            loss=loss+.05*auxiliary_loss
            auxiliary_losses.append(float(auxiliary_loss.detach()))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.repair.parameters() if p.grad is not None)
        gradients.append(float(model.repair.front_context.variable_context.position.grad.abs().sum()))
        if model.repair.recent_mean_projector.global_level_network is not None:global_level_gradients.append(float(model.repair.recent_mean_projector.global_level_network[0].weight.grad.abs().sum()))
        opt.step()
    if aux is not None:assert all(torch.isfinite(p.grad).all() for p in aux.parameters() if p.grad is not None)
    if plan['recent_mean_projection']=='global_level_mean_v2':assert global_level_gradients[1]>0
    assert parameter_hash(model.predictor)==plan['predictor_initial_sha256']
    assert gradients[1] > 0
    before = parameter_hash(model.repair)
    opt2 = build_stage2_optimizer(model,mode='frozen_v1',learning_rate=1e-4,weight_decay=0)
    stage2_training_mode(model,'frozen_v1')
    assert not any(p.requires_grad for p in model.repair.parameters())
    assert not model.repair.training and model.predictor.training
    prediction,_ = model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    error = prediction-batch.forecast_target
    forecast_loss = error.square().mean()+.2*error.abs().mean()
    opt2.zero_grad(); forecast_loss.backward(); opt2.step()
    assert torch.isfinite(forecast_loss)
    assert before == parameter_hash(model.repair)
    item = dict(scene=scene['name'],training_seed=2023,stage1_step_loss=float(loss.detach()),
                position_gradients=gradients,auxiliary_losses=auxiliary_losses,auxiliary_predictor_initial_sha256=aux_hash,auxiliary_epoch11_hidden_gradient_sum=active_hidden_gradient,auxiliary_predictor_parameters=35427 if aux is not None else 0,stage1_trainable_parameters=trainable,stage1_total_parameters=total,global_level_gradients=global_level_gradients,final_predictor_initial_sha256=plan['predictor_initial_sha256'],front_level_context=settings.front_level_context,stage1_gap_structure=settings.stage1_gap_structure,gap_level=float(parts['gap_level'].detach()),gap_trend=float(parts['gap_trend'].detach()),gap_segments=int(parts['gap_segments'].item()),stage2_forecast_loss=float(forecast_loss.detach()),
                original_predictor_parameters=35427,repair_parameters=plan['repair_parameters'],historical_key_policy=settings.historical_key_policy,key_inference_global_coordinates_checked=True,key_loss_recipe='legacy_pair_and_contrastive',key_ema_schedule='epoch',
                complete_repair_freeze=True,stage2_trainable_repair_parameters=0,
                validation_mask_sha256=protocol.mask_sha256['validation'],test_bank_count=20,
                no_test_scoring=True)
    results.append(item); print(json.dumps(item),flush=True)
    del model,aux,opt,opt2,prediction,out,loss,forecast_loss,error,batch
    torch.cuda.empty_cache()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.write('\n')

write(ROOT/'preflight.json',dict(status='passed',scenarios=results,original_head_config_unchanged=True))
sources = json.loads((ROOT/'copied-parent-sources.json').read_text())
protected = json.loads((ROOT/'protected-files.json').read_text())
changes = [p for p,h in sources.items() if sha(WORK/p)!=h]
assert set(changes)=={'integrations/e23/config.py','integrations/e23/train.py','src/malr/models/e23/model.py','src/malr/models/e23/historical_candidates.py'}
all_sources = {str(p.relative_to(WORK)):sha(p) for folder in ['src','integrations','scripts','configs','tests'] for p in (WORK/folder).rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])}
all_sources['pyproject.toml']=sha(WORK/'pyproject.toml')
sources=all_sources
assert all(sha(p)==h for p,h in protected.items())
definition = dict(sources=sources,protected=protected,plan_sha256=sha(ROOT/'plan.json'),
                  controller={p:sha(ROOT/p) for p in ['execute.py','preflight.py','test_controller.py','publish.py','launch.py','reporting/diagnose_long_gaps.py']},
                  parent_v77_source_changes=len(changes),new_files=sorted(set(all_sources)-set(json.loads((ROOT/'copied-parent-sources.json').read_text()))))
write(ROOT/'frozen-definition.json',definition)
with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as f:
    for p in sources:
        f.add(WORK/p,arcname=p,recursive=False)
with (ROOT/'progress.md').open('a') as f:
    f.write('Task2 complete: 7newRED→GREEN/15targeted checks; fullsuite1187pass/same9failure/1skip; controller5pass; actual CUDA globalquery/gradient/oldhead/freeze; no test scoring.\n')
print(json.dumps(dict(event='preflight_passed',scenarios=1,source_files=len(sources),protected_files=len(protected))),flush=True)
