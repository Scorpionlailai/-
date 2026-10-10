"""Actual CUDA batch checks for a frozen, single-module recovery candidate."""
import argparse,ast,hashlib,json
from dataclasses import replace
from pathlib import Path
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.report_e23_recovery_trial import ROOT,PREDICTOR_SHA
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (build_forecast_model,_new_memory,_starts,_stage_observed,
    _publish_epoch,_verify_bank_key_pair,_key_sha256,_retrieval_loss,stage1_objective)
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches


def node(path,name):
    source=Path(path).read_text();parsed=ast.parse(source)
    return ast.dump(next(n for n in parsed.body if getattr(n,'name',None)==name),include_attributes=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cuda:1');a=p.parse_args();root=a.root
    trial=json.loads((root/'trial.json').read_text());baseline=Path(trial['baseline_workspace'])
    for path,name in [('integrations/e23/train.py','train_forecast'),
                      ('integrations/e23/train.py','build_forecast_model'),
                      ('src/malr/models/e23/model.py','E23ForecastModel')]:
        assert node(ROOT/path,name)==node(baseline/path,name),(path,name)
    hashes=json.loads((root/'baseline-source-files.json').read_text())['files']
    head_files=[n for n in hashes if n.startswith('src/malr/vendor/patchtst/') or
                n in ('src/malr/models/patchtst.py','src/malr/models/repaired_forecast.py')]
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==hashes[n] for n in head_files)
    protocol=load_protocol();settings=load_e23_settings(Path(trial['config']),protocol=protocol)
    device=torch.device(a.device);torch.set_num_threads(4)
    batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    torch.manual_seed(2021);model=build_forecast_model(protocol,settings,device=device);model.eval()
    assert parameter_hash(model.predictor)==PREDICTOR_SHA
    repair=model.repair; memory=_new_memory(protocol)
    starts=_starts(batch,stride=8,patches=12,device=device)
    repair.historical.retrieval.begin_epoch(1)
    assert _stage_observed(memory,repair,batch,starts,set())>0
    _publish_epoch(memory,repair,1);_verify_bank_key_pair(memory,repair)
    repair.historical.refresh_snapshot(memory.snapshot())
    before=(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
    with torch.inference_mode():
        initial,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    # New modules initialize to the original function under the same H policy.
    control_settings=replace(settings,**trial['control_setting_overrides'])
    torch.manual_seed(2021);control=build_forecast_model(protocol,control_settings,device=device);control.eval()
    control.repair.historical.retrieval.load_state_dict(repair.historical.retrieval.state_dict())
    control.repair.historical.refresh_snapshot(memory.snapshot())
    with torch.inference_mode():
        original,_=control(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    assert torch.equal(initial,original)
    del control
    point=getattr(repair.temporal,'point_context',None)
    if point is not None: torch.nn.init.normal_(point.projection.weight,std=.02)
    forecast,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    retrieval=_retrieval_loss(repair,batch)
    loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,retrieval,
        retrieval_weight=.1,loss_mode='global_mse_v1',query_std=batch.query_std)
    loss=loss+(forecast-batch.forecast_target).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if point is not None: assert point.stem[0].weight.grad.norm()>0
    assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
    assert torch.equal(out.explicit.sequence_observed,batch.observed_mask)
    with torch.no_grad():
        altered=batch.normalized_observed.clone();altered[~batch.observed_mask]=12345
        other,_=model(altered,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        assert torch.equal(forecast,other)
    assert before==(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
    record=dict(predictor_initial_sha256=PREDICTOR_SHA,predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
        repair_parameters=sum(p.numel() for p in repair.parameters()),stage2_ast_unchanged=True,head_sources_unchanged=head_files,
        initial_function_matches_control=True,active_module_gradients_finite=True,hidden_inputs_ignored=True,
        h_state_unchanged=True,formal_reuses_preflight_weights=False)
    (root/'cuda-preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
