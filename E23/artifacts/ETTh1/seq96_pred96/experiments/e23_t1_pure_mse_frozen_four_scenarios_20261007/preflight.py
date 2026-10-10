"""Check cached repair integrity and pure-MSE head updates in each scene."""
import json
import os
from pathlib import Path
import torch

from scripts.run_t1_frozen_stage2 import read, write, preflight as t1_preflight
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (build_forecast_model, load_stage1_selection, _new_memory,
                                     _starts, stage2_objective)
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parent
plan=read(ROOT/'plan.json')
torch.set_num_threads(4)
device=torch.device(plan['device'])
results=[]
for scene in plan['scenarios']:
    t1_preflight(read(ROOT/'t1-plan.json'),scene)
    os.environ.pop('MALR_MISSING_SCENARIO',None)
    if scene['scenario_manifest']:os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
    protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
    assert len(load_test_bank(protocol,Path(scene['test_bank'])))==20
    assert protocol.mask_sha256['validation']==scene['validation_mask_sha256']
    settings=load_e23_settings(Path(plan['e23_config']),protocol=protocol)
    payload=load_stage1_selection(Path(scene['e23_stage1']),protocol,settings,device=device)
    torch.manual_seed(2023)
    model=build_forecast_model(protocol,settings,device=device)
    assert parameter_hash(model.predictor)==plan['predictor_initial_sha256']
    model.repair.load_state_dict(payload['model_state'],strict=True)
    model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state'])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    optimizer=build_stage2_optimizer(model,mode='frozen_v1',learning_rate=1e-4,weight_decay=0)
    stage2_training_mode(model,'frozen_v1')
    initial=parameter_hash(model.repair)
    batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    starts=_starts(batch,stride=8,patches=12,device=device)
    optimizer.zero_grad(set_to_none=True)
    prediction,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    loss=stage2_objective(prediction,batch.forecast_target,loss_mode='mse')
    assert torch.equal(loss,(prediction-batch.forecast_target).square().mean())
    assert torch.isfinite(loss)
    loss.backward();optimizer.step()
    assert parameter_hash(model.repair)==initial
    assert not any(p.requires_grad or p.grad is not None for p in model.repair.parameters())
    assert not model.repair.training and model.predictor.training
    assert sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)==35424
    result=dict(scene=scene['name'],status='passed',pure_mse=True,frozen_repair=True,
                loss=float(loss.detach()),test_scoring=False,head_initial_sha256=plan['predictor_initial_sha256'])
    results.append(result);print(json.dumps(result),flush=True)
    del model,optimizer,prediction,loss,batch,payload,memory
    torch.cuda.empty_cache()
write(ROOT/'preflight.json',dict(status='passed',four_scenes=results,models=2,no_test_scoring=True))
