"""Throwaway CUDA measurements at the formal batch size; no optimizer updates."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
import torch
import yaml
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (build_forecast_model,_new_memory,_stage_observed,_publish_epoch,_starts,
    _retrieval_loss,stage1_objective,stage2_objective)
from integrations.forecast_compare.protocol import iter_p4_batches
from scripts.diagnose_e23_v30_stack import load_protocol
from malr.evaluation.report import file_sha256


def measure(config,protocol,device):
    settings=load_e23_settings(config,protocol=protocol);torch.manual_seed(settings.seed)
    model=build_forecast_model(protocol,settings,device=device);repair=model.repair
    repair.historical.retrieval.begin_epoch(1);memory=_new_memory(protocol)
    batches=iter(iter_p4_batches(protocol,'train',batch_size=128,epoch=0));known=set()
    for _ in range(4):
        batch=next(batches).to(device);starts=_starts(batch,stride=repair.stride,patches=repair.patch_count,device=device)
        _stage_observed(memory,repair,batch,starts,known_identities=known)
    _publish_epoch(memory,repair,1);batch=next(batches).to(device)
    starts=_starts(batch,stride=repair.stride,patches=repair.patch_count,device=device)
    row=dict(config=str(config.resolve()),config_sha256=file_sha256(config),layers=settings.recovery_layers,
        groups=settings.expert_groups,activation_checkpointing=settings.activation_checkpointing,
        repair_parameters=sum(p.numel() for p in repair.parameters()),forecast_parameters=sum(p.numel() for p in model.parameters()),
        parameter_counts={name:sum(p.numel() for p in module.parameters()) for name,module in repair.named_children()},
        temporary_bank_members=memory.snapshot().member_count,stages={})
    for stage in ('repair','forecast'):
        if stage=='forecast':
            for parameter in repair.historical.key_encoder.parameters():parameter.requires_grad_(False)
        model.train();model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        calls=[];original_query=repair.historical.index.query
        def counted(*args,**kwargs):calls.append(1);return original_query(*args,**kwargs)
        repair.historical.index.query=counted
        forward=backward=0.
        try:
            for i in range(14):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize(device);started=perf_counter()
                if stage=='repair':
                    out=repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,
                        epoch=1,explore_variable=True,specialist_exposure=.2,specialist_seed=settings.seed)
                    loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,
                        _retrieval_loss(repair,batch),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=batch.query_std)
                else:
                    prediction,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
                    loss=stage2_objective(prediction,batch.forecast_target,loss_mode='mse_plus_0p2_mae')
                torch.cuda.synchronize(device);forwarded=perf_counter()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite preflight loss')
                loss.backward();torch.cuda.synchronize(device);finished=perf_counter()
                if i>=4:forward+=forwarded-started;backward+=finished-forwarded
                if i==3:torch.cuda.reset_peak_memory_stats(device)
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise FloatingPointError('nonfinite preflight gradient')
        finally:repair.historical.index.query=original_query
        if len(calls)!=14:raise RuntimeError('each forward must issue exactly one H query')
        row['stages'][stage]=dict(forward_ms=forward*100,backward_ms=backward*100,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,h_queries=len(calls),
            layer_group_expert_route_available=[item.routes[...,1:].gt(0).float().mean((0,1,2)).tolist() for item in out.stack_layers])
    del model,repair,memory,batch,out,loss,original_query
    gc.collect();torch.cuda.empty_cache();return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--configs',nargs=3,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);device=torch.device(args.device)
    if device.type!='cuda':raise ValueError('CUDA profiling requires CUDA')
    props=torch.cuda.get_device_properties(device);protocol=load_protocol()
    result=dict(purpose='throwaway preflight; no optimizer steps or formal initialization',batch_size=128,precision='float32',
        warmup_steps=4,measured_steps=10,gpu=props.name,gpu_uuid=str(props.uuid),device=str(device),models=[])
    for config in args.configs:
        oom=False
        try:row=measure(config,protocol,device)
        except torch.cuda.OutOfMemoryError:oom=True
        if oom:
            gc.collect();torch.cuda.empty_cache()
            document=yaml.safe_load(config.read_text())
            if document['model']['activation_checkpointing']:raise RuntimeError('batch128 OOM even with checkpointing')
            document['model']['activation_checkpointing']=True;config.write_text(yaml.safe_dump(document,sort_keys=False))
            row=measure(config,protocol,device);row['normal_path_oom']=True
        result['models'].append(row)
        print(json.dumps({'profiled':row['config'],'stages':row['stages']}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':main()
