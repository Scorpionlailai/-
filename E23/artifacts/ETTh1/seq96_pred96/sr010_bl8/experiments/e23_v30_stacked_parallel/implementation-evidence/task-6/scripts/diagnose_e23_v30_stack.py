"""Read-only validation diagnostics: layer prefixes and independent group ablations."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import (_key_sha256,_new_memory,_starts,build_repair_model,build_forecast_model,
    load_stage1_selection,load_stage2_selection)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parents[1]


def decode_layers(model,output,values,mask,mean,std,*,valid_mask=None):
    if output.stack_layers is None:raise ValueError('layer diagnostics require stacked recovery')
    patches=patchify_with_state(values,mask,model.patch_length,model.stride,validity=valid_mask)
    original=patches.values.masked_fill(~patches.effective_mask,0)
    result=[]
    for item in output.stack_layers:
        explicit=build_explicit_repair(model.decoder(item.latents),original,patches.effective_mask,patches.validity,
            mean,std,window_len=model.window_length,patch_len=model.patch_length,stride=model.stride)
        result.append(replace(explicit,repaired_patches=explicit.repaired_patches.masked_fill(~explicit.validity_patches,0)))
    return tuple(result)


def load_protocol():
    return load_e23_p4_protocol(catalog=MigrationCatalog.load(ROOT.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')


@torch.inference_mode()
def diagnose(model,protocol,settings,device,stage):
    model.eval();repair=model if stage=='stage1' else model.repair
    sums=[dict(squared=0.,absolute=0.,forecast_squared=0.,forecast_absolute=0.,update=0.,
        route=torch.zeros(settings.expert_groups,4,dtype=torch.double),gate=torch.zeros(settings.expert_groups+1,dtype=torch.double),
        preview=torch.zeros(settings.expert_groups,2,dtype=torch.double),difference=0.) for _ in range(settings.recovery_layers)]
    hidden_count=targets=windows=forecast_count=0;ablation_batches=[]
    for cpu in iter_p4_batches(protocol,'validation',batch_size=settings.batch_size):
        batch=cpu.to(device);starts=_starts(batch,stride=repair.stride,patches=repair.patch_count,device=device)
        output=repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        decoded=decode_layers(repair,output,batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std)
        torch.testing.assert_close(decoded[-1].normalized_sequence,output.explicit.normalized_sequence,rtol=0,atol=0)
        damaged=(~output.explicit.observation_patches & output.explicit.validity_patches).any(-1);hidden=~batch.observed_mask
        for acc,item,explicit in zip(sums,output.stack_layers,decoded):
            error=(explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:,None,:]
            acc['squared']+=error.square()[hidden].sum().item();acc['absolute']+=error.abs()[hidden].sum().item()
            acc['update']+=item.update_rms[damaged].sum().item()
            acc['route']+=item.routes[damaged].double().sum(0).cpu();acc['gate']+=item.group_weights[damaged].double().sum(0).cpu()
            acc['preview']+=item.preview_gates[damaged].double().sum(0).cpu()
            if settings.expert_groups>1:
                pair=item.group_deltas[..., :,None,:]-item.group_deltas[...,None,:,:]
                pair_mask=torch.triu(torch.ones(settings.expert_groups,settings.expert_groups,device=device,dtype=torch.bool),diagonal=1)
                acc['difference']+=pair.square().mean(-1).add(1e-12).sqrt()[damaged][:,pair_mask].mean(-1).sum().item()
            if stage=='stage2':
                prediction=model.predictor(explicit.repaired_patches,explicit.observation_patches,batch.query_mean,batch.query_std,patch_validity=explicit.validity_patches)
                forecast_error=prediction-batch.forecast_target
                acc['forecast_squared']+=forecast_error.square().sum().item();acc['forecast_absolute']+=forecast_error.abs().sum().item()
        hidden_count+=hidden.sum().item();targets+=damaged.sum().item();windows+=len(batch.normalized_observed)
        forecast_count+=batch.forecast_target.numel()
        if len(ablation_batches)<2:ablation_batches.append(cpu)
    layers=[]
    for i,acc in enumerate(sums,1):
        row=dict(layer_index=i,global_mse=acc['squared']/hidden_count,global_mae=acc['absolute']/hidden_count,
            mean_update_rms=acc['update']/targets,mean_route_weights=(acc['route']/targets).tolist(),
            mean_group_weights=(acc['gate']/targets).tolist(),mean_preview_gates=(acc['preview']/targets).tolist(),
            mean_weighted_projected_group_pairwise_rms=acc['difference']/targets)
        if stage=='stage2':row.update(forecast_mse=acc['forecast_squared']/forecast_count,forecast_mae=acc['forecast_absolute']/forecast_count)
        layers.append(row)
    ablations=[]
    def score(disabled):
        squared=absolute=count=0
        for cpu in ablation_batches:
            batch=cpu.to(device);starts=_starts(batch,stride=repair.stride,patches=repair.patch_count,device=device)
            output=repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,disabled_groups=disabled)
            if stage=='stage2':
                x=output.explicit
                pred=model.predictor(x.repaired_patches,x.observation_patches,batch.query_mean,batch.query_std,patch_validity=x.validity_patches)
                error=pred-batch.forecast_target
            else:
                hidden=~batch.observed_mask
                error=((output.explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:,None,:])[hidden]
            squared+=error.square().sum().item();absolute+=error.abs().sum().item();count+=error.numel()
        return dict(mse=squared/count,mae=absolute/count,coordinates=count)
    reference=score(frozenset())
    for l in range(settings.recovery_layers):
        for g in range(settings.expert_groups):
            ablations.append(dict(disabled_layer=l,disabled_group=g,**score(frozenset([(l,g)]))))
    return dict(windows=windows,hidden_coordinates=hidden_count,damaged_patch_targets=targets,layers=layers,
        group_ablation_scope='first two fixed validation batches only; no selection',group_ablation_windows=sum(len(b.normalized_observed) for b in ablation_batches),
        group_ablation_reference=reference,group_ablations=ablations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('e23-config','stage1-manifest','stage2-manifest','output'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);protocol=load_protocol();settings=load_e23_settings(args.e23_config,protocol=protocol);device=torch.device(args.device)
    result=dict(split='original_fixed_validation',metric_space='train_standardized',diagnostic_only=True,
        model=f'E23_v30_l{settings.recovery_layers}_g{settings.expert_groups}',e23_config_sha256=settings.config_sha256,
        recovery_layers=settings.recovery_layers,expert_groups=settings.expert_groups,
        routes_are_descriptive_not_calibrated_confidence=True)
    for stage,manifest in [('stage1',args.stage1_manifest),('stage2',args.stage2_manifest)]:
        record=json.loads(manifest.read_text());loader=load_stage1_selection if stage=='stage1' else load_stage2_selection
        payload=loader(manifest,protocol,settings,device=device)
        model=(build_repair_model if stage=='stage1' else build_forecast_model)(protocol,settings,device=device)
        model.load_state_dict(payload['model_state'],strict=True);repair=model if stage=='stage1' else model.repair
        repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol)
        memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
        before,bank=_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash()
        metrics=diagnose(model,protocol,settings,device,stage)
        after,bank_after=_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash()
        if before!=after or bank!=bank_after or before!=record['key_sha256']:raise ValueError('historical state changed during diagnosis')
        if stage=='stage1' and abs(metrics['layers'][-1]['global_mse']-record['best_val_repair_mse'])>1e-6:raise ValueError('Stage1 final layer replay mismatch')
        if stage=='stage2' and any(abs(metrics['layers'][-1]['forecast_'+k]-record['best_val_'+k])>1e-6 for k in ('mse','mae')):
            raise ValueError('Stage2 final layer forecast replay mismatch')
        metrics.update(checkpoint_sha256=record['checkpoint_sha256'],selected_epoch=record['best_epoch'],h_key_sha256_before=before,
            h_key_sha256_after=after,h_bank_sha256_before=bank,h_bank_sha256_after=bank_after)
        result[stage]=metrics;result['repair_parameters']=sum(p.numel() for p in repair.parameters())
        del model,repair,memory,payload
        if device.type=='cuda':torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'output':str(args.output),'layers':settings.recovery_layers,'groups':settings.expert_groups}))

if __name__=='__main__':main()
