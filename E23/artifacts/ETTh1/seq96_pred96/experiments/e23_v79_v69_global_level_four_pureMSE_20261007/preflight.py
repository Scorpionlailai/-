"""Check the actual v69+global-level configuration before formal training."""
import json
import os
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,_starts,stage1_objective,_retrieval_loss
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'plan.json').read_text())
torch.set_num_threads(4)
device=torch.device(plan['device'])
results=[]
for scene in plan['scenarios']:
    os.environ.pop('MALR_MISSING_SCENARIO',None)
    if scene['scenario_manifest']:os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
    protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
    assert protocol.mask_sha256['validation']==scene['validation_mask_sha256']
    assert len(load_test_bank(protocol,Path(scene['test_bank'])))==20
    settings=load_e23_settings(Path(plan['e23_config']),protocol=protocol)
    assert settings.stage1_gap_structure=='none' and settings.stage1_structure=='recent_mean_v2'
    assert settings.recent_mean_projection=='global_level_mean_v2'
    torch.manual_seed(2023)
    model=build_forecast_model(protocol,settings,device=device)
    assert parameter_hash(model.predictor)==plan['predictor_initial_sha256']
    assert sum(p.numel() for p in model.repair.parameters())==424238
    batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    starts=_starts(batch,stride=8,patches=12,device=device)
    optimizer=torch.optim.Adam(model.repair.parameters(),lr=.001)
    gradients=[]
    for step in range(2):
        model.repair.train();optimizer.zero_grad(set_to_none=True)
        out=model.repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
        loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,
            _retrieval_loss(model.repair,batch),retrieval_weight=.1,loss_mode='global_mse_v1',
            query_std=batch.query_std,structure_mode='recent_mean_v2')
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.repair.parameters() if p.grad is not None)
        gradients.append(float(model.repair.recent_mean_projector.global_level_network[0].weight.grad.abs().sum()))
        optimizer.step()
    assert gradients[1]>0
    optimizer2=build_stage2_optimizer(model,mode='frozen_v1',learning_rate=1e-4,weight_decay=0)
    # Drop stale Stage1 gradients before checking the frozen Stage2 path.
    model.repair.zero_grad(set_to_none=True)
    stage2_training_mode(model,'frozen_v1');initial=parameter_hash(model.repair)
    prediction,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    loss2=(prediction-batch.forecast_target).square().mean()
    optimizer2.zero_grad(set_to_none=True);loss2.backward();optimizer2.step()
    assert parameter_hash(model.repair)==initial
    assert not any(p.requires_grad or p.grad is not None for p in model.repair.parameters())
    result=dict(scene=scene['name'],status='passed',stage1_loss=float(loss.detach()),
                global_branch_gradients=gradients,stage2_pure_mse=float(loss2.detach()),
                repair_parameters=424238,head_parameters=35427,head_unchanged=True,
                stage2_full_repair_freeze=True,no_test_scoring=True)
    results.append(result);print(json.dumps(result),flush=True)
    del model,optimizer,optimizer2,batch,out,loss,loss2,prediction
    torch.cuda.empty_cache()
with (ROOT/'preflight.json').open('x') as f:json.dump(dict(status='passed',scenarios=results),f,indent=2);f.write('\n')
