import hashlib, json, os, tarfile
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, _starts, stage1_objective, _retrieval_loss, _tv_query_loss
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from malr.migration import MigrationCatalog

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
    assert sum(p.numel() for p in model.repair.parameters()) == 425389
    batch = next(iter_p4_batches(protocol,'train',batch_size=4,epoch=0)).to(dev)
    starts = _starts(batch,stride=8,patches=12,device=dev)
    opt = torch.optim.Adam(model.repair.parameters(),lr=.001)
    gradients = []
    for step in range(2):
        model.train(); opt.zero_grad()
        out = model.repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
        loss,parts = stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,
            batch.observed_mask,_retrieval_loss(model.repair,batch),retrieval_weight=.1,
            loss_mode='global_mse_v1',query_std=batch.query_std,structure_mode='recent_mean_v2')
        query_loss = _tv_query_loss(model.repair,batch,out)
        assert torch.isfinite(query_loss) and query_loss > 0
        loss = loss + .1*query_loss
        loss.backward()
        assert model.repair.historical.tv_query.project.weight.grad.abs().sum() > 0
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.repair.parameters() if p.grad is not None)
        gradients.append(float(model.repair.front_context.variable_context.position.grad.abs().sum()))
        opt.step()
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
                position_gradients=gradients,tv_query_alignment_loss=float(query_loss.detach()),stage2_forecast_loss=float(forecast_loss.detach()),
                original_predictor_parameters=35427,repair_parameters=425389,
                complete_repair_freeze=True,stage2_trainable_repair_parameters=0,
                validation_mask_sha256=protocol.mask_sha256['validation'],test_bank_count=20,
                no_test_scoring=True)
    results.append(item); print(json.dumps(item),flush=True)
    del model,opt,opt2,prediction,out,loss,forecast_loss,error,batch
    torch.cuda.empty_cache()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.write('\n')

write(ROOT/'preflight.json',dict(status='passed',scenarios=results,original_head_config_unchanged=True))
sources = json.loads((ROOT/'copied-v69-sources.json').read_text())
protected = json.loads((ROOT/'protected-files.json').read_text())
changes = [p for p,h in sources.items() if sha(WORK/p)!=h]
assert set(changes) == {'integrations/e23/config.py','integrations/e23/train.py','src/malr/models/e23/model.py','src/malr/models/e23/iterative.py','src/malr/models/e23/historical.py','src/malr/models/e23/historical_candidates.py'}
all_sources = {str(p.relative_to(WORK)):sha(p) for folder in ['src','integrations','scripts','configs','tests'] for p in (WORK/folder).rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])}
all_sources['pyproject.toml']=sha(WORK/'pyproject.toml')
sources=all_sources
assert all(sha(p)==h for p,h in protected.items())
definition = dict(sources=sources,protected=protected,plan_sha256=sha(ROOT/'plan.json'),
                  controller={p:sha(ROOT/p) for p in ['execute.py','preflight.py','test_controller.py']},
                  original_v69_source_changes=len(changes),new_files=sorted(set(all_sources)-set(json.loads((ROOT/'copied-v69-sources.json').read_text()))))
write(ROOT/'frozen-definition.json',definition)
with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as f:
    for p in sources:
        f.add(WORK/p,arcname=p,recursive=False)
with (ROOT/'progress.md').open('a') as f:
    f.write('Task2 complete: four orchestration checks RED→GREEN; two actual CUDA scenario preflights passed, identical head, finite losses, full repair freeze; no test scoring.\n')
print(json.dumps(dict(event='preflight_passed',scenarios=2,source_files=len(sources),protected_files=len(protected))),flush=True)
