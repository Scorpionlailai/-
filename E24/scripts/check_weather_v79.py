"""Check dimension migration, complete evaluation windows and frozen Stage2."""
import json
from pathlib import Path
from time import perf_counter
from dataclasses import replace

import numpy as np
import torch

from integrations.e24.data import load_dataset_protocol, read_manifest
from integrations.e23.config import load_e23_settings, load_e23_p4_protocol
from integrations.e23.train import (build_forecast_model, _starts, stage1_objective,
    _new_memory, _key_sha256, load_stage2_selection, _validation_forecast_metrics)
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches, prepare_p4_batch
from malr.migration import MigrationCatalog
from malr.evaluation.report import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    manifest = ROOT/'artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json'
    record = read_manifest(manifest)
    protocol = load_dataset_protocol(manifest, config_path=Path(record['config_path']), artifacts=manifest.parent)
    settings = load_e23_settings(ROOT/'configs/e24_v79.yaml', protocol=protocol)
    assert {k:len(v) for k,v in protocol.refs.items()} == dict(train=36696,validation=5175,test=10444)
    counts = {}
    for split in ('validation','test'):
        counts[split] = sum(len(b.refs) for b in iter_p4_batches(protocol,split))
        assert counts[split] == len(protocol.refs[split])
        for ref in protocol.refs[split]:
            bounds = protocol.bounds.bounds_for(split)
            assert bounds.start <= ref.target_start < ref.target_end <= bounds.end
    device = torch.device('cuda:1')
    torch.manual_seed(2023)
    model = build_forecast_model(protocol,settings,device=device)
    head_sha = parameter_hash(model.predictor)
    parent = ROOT.parent/'E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007'
    parent_plan = json.loads((parent/'plan.json').read_text())
    assert head_sha == parent_plan['predictor_initial_sha256']
    batch = next(iter_p4_batches(protocol,'train',epoch=0)).to(device)
    starts = _starts(batch,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
    torch.cuda.reset_peak_memory_stats(device)
    t = perf_counter()
    model.repair.train()
    output = model.repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    loss,_ = stage1_objective(output.explicit.normalized_sequence,batch.normalized_target,
                              batch.observed_mask,output.explicit.normalized_sequence.sum()*0,
                              retrieval_weight=.1,loss_mode='global_mse_v1',query_std=batch.query_std,
                              structure_mode='recent_mean_v2')
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.repair.parameters() if p.grad is not None)
    stage1_peak = torch.cuda.max_memory_allocated(device)/2**30
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        a = model.repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        assert torch.equal(a.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
        # Different hidden truth/future labels must leave actual repair inputs unchanged.
        cpu = next(iter_p4_batches(protocol,'train',epoch=0))
        clean = cpu.normalized_target*cpu.query_std[:,None]+cpu.query_mean[:,None]
        damaged = torch.where(cpu.observed_mask,clean,clean+123.)
        b = prepare_p4_batch(damaged,cpu.observed_mask,cpu.forecast_target+999.,protocol.fallback,refs=cpu.refs)
        assert torch.allclose(cpu.normalized_observed,b.normalized_observed,atol=2e-5,rtol=2e-5)
        assert torch.allclose(cpu.query_mean,b.query_mean,atol=2e-6,rtol=2e-6)
        assert torch.allclose(cpu.query_std,b.query_std,atol=2e-6,rtol=2e-6)
    del output,a,loss
    optimizer = build_stage2_optimizer(model,mode='frozen_v1',learning_rate=1e-4,weight_decay=0.)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.predictor.parameters() if p.requires_grad}
    stage2_training_mode(model,'frozen_v1')
    assert all(not m.training for m in model.repair.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.repair.parameters())
    before = parameter_hash(model.repair), _key_sha256(model.repair.historical.retrieval.state_dict())
    prediction,_ = model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    torch.nn.functional.mse_loss(prediction,batch.forecast_target).backward()
    optimizer.step()
    assert before == (parameter_hash(model.repair),_key_sha256(model.repair.historical.retrieval.state_dict()))
    result = dict(status='passed',dataset='Weather',protocol_id=record['protocol_id'],
                  windows={k:len(v) for k,v in protocol.refs.items()},evaluated_windows=counts,
                  train_batches_per_epoch=286,optimizer_updates_per_stage=28600,
                  predictor_initial_sha256=head_sha,predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
                  repair_parameters=sum(p.numel() for p in model.repair.parameters()),
                  batch128_forward_backward_peak_allocated_GiB=stage1_peak,
                  preflight_seconds=perf_counter()-t,preflight_optimizer_updates=1,
                  hidden_truth_excluded=True,visible_values_exact=True,optimizer_only_predictor=True,
                  repair_state_unchanged=True,head_source_sha256=file_sha256(ROOT/'src/malr/models/patchtst.py'))
    del model,optimizer,prediction,batch
    torch.cuda.empty_cache()
    # Recheck one ETTh1 parent's complete validation after generic dimension changes.
    catalog = MigrationCatalog.load(ROOT/'migration/paper_catalog.jsonl',project='E23')
    legacy = load_e23_p4_protocol(catalog=catalog,protocol_config=ROOT/'configs/etth1_seq96_pred96.yaml',
               mask_artifacts=Path(parent_plan['mask_artifacts']),training_seed=2023)
    old_settings = load_e23_settings(ROOT/'configs/e24_v79.yaml',protocol=legacy)
    selection = parent/'sr010_bl8/E23-v79/stage2/runs/e23-stage2.json'
    selected = json.loads(selection.read_text())
    legacy = replace(legacy,config_path=Path(selected['protocol_config_path']))
    payload = load_stage2_selection(selection,legacy,old_settings,device=device)
    old_model = build_forecast_model(legacy,old_settings,device=device)
    old_model.load_state_dict(payload['model_state'],strict=True)
    old_model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(legacy);memory.load_state_dict(payload['memory_state'])
    old_model.repair.historical.refresh_snapshot(memory.snapshot())
    metrics = _validation_forecast_metrics(old_model,legacy,old_settings,device=device)
    assert all(abs(metrics[k]-selected['best_val_'+k])<1e-6 for k in ('mse','mae'))
    result['etth1_regression_validation'] = metrics
    result['etth1_regression_passed'] = True
    out = manifest.parent/'preflight_acceptance.json'
    with out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
