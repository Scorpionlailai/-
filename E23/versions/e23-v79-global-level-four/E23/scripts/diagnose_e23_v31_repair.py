"""Replay selected v31 repair error on the original validation split."""
import argparse,json
from time import perf_counter
from scripts.report_e23_v31_trial import digest
from pathlib import Path
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.report_e23_v31_trial import CONFIG
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage2_selection,build_forecast_model,_new_memory,_key_sha256,_starts
from integrations.forecast_compare.protocol import iter_p4_batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=perf_counter()
    torch.set_num_threads(4);protocol=load_protocol();settings=load_e23_settings(CONFIG,protocol=protocol);device=torch.device(a.device)
    payload=load_stage2_selection(a.selection,protocol,settings,device=device);model=build_forecast_model(protocol,settings,device=device);model.load_state_dict(payload['model_state']);model.eval();repair=model.repair
    repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
    before=(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash());sq=ab=count=windows=0
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
            b=cpu.to(device);starts=_starts(b,stride=8,patches=12,device=device);out=repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            error=((out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None,:])[~b.observed_mask]
            sq+=error.square().double().sum().item();ab+=error.abs().double().sum().item();count+=error.numel();windows+=len(b.normalized_observed)
    assert before==(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
    record=json.loads(a.selection.read_text())
    a.output.write_text(json.dumps(dict(checkpoint_sha256=record['checkpoint_sha256'],selection_sha256=digest(a.selection),e23_config_sha256=record['e23_config_sha256'],source_csv_sha256=protocol.source_sha256,validation_mask_sha256=protocol.mask_sha256['validation'],repair_parameters=sum(p.numel() for p in model.repair.parameters()),predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),diagnostic_seconds=perf_counter()-start,global_mse=sq/count,global_mae=ab/count,hidden_coordinates=count,windows=windows,split='original_fixed_validation',metric_space='train_standardized',h_state_unchanged=True),indent=2)+'\n')

if __name__=='__main__':main()
