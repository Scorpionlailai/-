"""Strictly load saved E23 models and replay the mapped single-group v29 reference."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (build_repair_model,build_forecast_model,load_stage1_selection,load_stage2_selection,
    _new_memory,_key_sha256,_starts)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.models.e23.stacked_reference import copy_v29_r1_into_stack_reference
from scripts.diagnose_e23_v30_stack import load_protocol,ROOT

EXPERIMENTS=ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);device=torch.device(args.device);protocol=load_protocol();records=[];source=None
    entries=[('v3','e23_multi_expert_p4_width64.yaml',EXPERIMENTS/'e23_multi_expert_width64_mae_v3/stage2_p4/runs/e23-stage2.json'),
        ('v12','e23_multi_expert_p4_width64_window_v12.yaml',ROOT/'artifacts/author_p4_20261002/E23_v12/stage2/runs/e23-stage2.json'),
        ('v28','e23_multi_expert_p4_width64_global_stage1_v28.yaml',EXPERIMENTS/'e23_v28_global_stage1/stage2/runs/e23-stage2.json'),
        ('v29_r1','e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml',EXPERIMENTS/'e23_v29_iterative_tv_h_r1/stage2/runs/e23-stage2.json'),
        ('v29_r3','e23_multi_expert_p4_width64_iterative_tv_h_r3_v29.yaml',EXPERIMENTS/'e23_v29_iterative_tv_h_r3/stage2/runs/e23-stage2.json')]
    for name,config,s2 in entries:
        settings=load_e23_settings(ROOT/'configs'/config,protocol=protocol);record=json.loads(s2.read_text());s1=Path(record['stage1_manifest'])
        payload=load_stage1_selection(s1,protocol,settings,device=device);repair=build_repair_model(protocol,settings,device=device)
        repair.load_state_dict(payload['model_state'],strict=True);del repair,payload
        payload=load_stage2_selection(s2,protocol,settings,device=device);model=build_forecast_model(protocol,settings,device=device)
        model.load_state_dict(payload['model_state'],strict=True)
        records.append(dict(model=name,stage1='strict_load_passed',stage2='strict_load_passed',checkpoint_sha256=record['checkpoint_sha256']))
        if name=='v29_r1':
            source=model;source_record=record;source_settings=settings;source_payload=payload
        else:del model,payload
    source.repair.historical.retrieval.load_state_dict(source_payload['retrieval_state']);memory=_new_memory(protocol)
    memory.load_state_dict(source_payload['memory_state']);source.repair.historical.refresh_snapshot(memory.snapshot())
    target_settings=replace(source_settings,recovery_schedule='stacked_parallel_tv_h_v1',recovery_layers=1,expert_groups=1)
    target=build_forecast_model(protocol,target_settings,device=device)
    copy_v29_r1_into_stack_reference(source.repair,target.repair)
    target.predictor.load_state_dict(source.predictor.state_dict(),strict=True)
    source.eval();target.eval();before=_key_sha256(source.repair.historical.retrieval.state_dict());bank=memory.snapshot_hash()
    sums=[dict(squared=0.,absolute=0.) for _ in range(2)];count=windows=0;max_difference=0.
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
            batch=cpu.to(device);starts=_starts(batch,stride=source.repair.stride,patches=source.repair.patch_count,device=device)
            predictions=[]
            for model in (source,target):
                prediction,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts);predictions.append(prediction)
            max_difference=max(max_difference,(predictions[0]-predictions[1]).abs().max().item())
            # Grouped GEMMs change FP32 reduction order; metrics below retain
            # the required absolute 1e-6 gate on the complete validation set.
            for acc,prediction in zip(sums,predictions):
                error=prediction-batch.forecast_target;acc['squared']+=error.square().sum().item();acc['absolute']+=error.abs().sum().item()
            count+=predictions[0].numel();windows+=len(batch.normalized_observed)
    after=_key_sha256(source.repair.historical.retrieval.state_dict());bank_after=memory.snapshot_hash()
    if before!=after or before!=_key_sha256(target.repair.historical.retrieval.state_dict()) or bank!=bank_after:raise ValueError('historical state changed')
    replay=[dict(mse=s['squared']/count,mae=s['absolute']/count) for s in sums]
    if max_difference>1e-5: raise ValueError('mapped FP32 pointwise prediction difference exceeds 1e-5')
    for metrics in replay:
        for k in ('mse','mae'):
            if abs(metrics[k]-source_record['best_val_'+k])>1e-6:raise ValueError('v29 mapped replay mismatch')
    result=dict(strict_loads=records,windows=windows,coordinates=count,reference_replay=replay[0],mapped_stack_replay=replay[1],
        max_prediction_difference=max_difference,h_key_sha256_before=before,h_key_sha256_after=after,h_bank_sha256_before=bank,
        h_bank_sha256_after=bank_after,mapping_used_for_formal_initialization=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
