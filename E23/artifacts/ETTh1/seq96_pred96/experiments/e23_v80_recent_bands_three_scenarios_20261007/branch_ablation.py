"""Disable only the new branch at inference on locked v80; no training or test selection."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,load_stage2_selection,_starts,_new_memory,_key_sha256
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_preserving_values
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parent
PARENT=ROOT

def affine_error(error,hidden):
    t=torch.linspace(-1,1,32,device=error.device,dtype=error.dtype)[None,:,None]
    m=hidden.to(error.dtype);n=m.sum(1).clamp_min(1)
    mean=(error*m).sum(1)/n;phase=(t*m).sum(1)/n
    centered=t-phase[:,None];den=(centered.square()*m).sum(1)
    trend=(centered*error*m).sum(1)/den.clamp_min(1e-6)
    trend=trend.masked_fill(den<1e-6,0)
    return mean[:,None].expand_as(error),trend[:,None]*centered

def main():
    plan=json.loads((PARENT/'plan.json').read_text());torch.set_num_threads(4);device=torch.device('cuda:1');results=[]
    for scene in plan['scenarios']:
        os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
        protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),
            protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
        settings=load_e23_settings(Path(plan['e23_config']),protocol=protocol)
        selection=PARENT/scene['name']/'E23-v80/stage2/runs/e23-stage2.json'
        record=json.loads(selection.read_text());before_selection=file_sha256(selection)
        payload=load_stage2_selection(selection,protocol,settings,device=device)
        model=build_forecast_model(protocol,settings,device=device)
        model.load_state_dict(payload['model_state'],strict=True)
        model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot())
        model.eval().requires_grad_(False);before=parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict())
        labels=['base','disabled_band']
        totals={k:[0.,0.,0] for k in labels}
        repair={label:{k:[0.,0.,0] for k in ['all_hidden','last32_hidden','last16_hidden']} for label in labels}
        del payload
        with torch.inference_mode():
            for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
                b=cpu.to(device);starts=_starts(b,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
                out=model.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                e=out.explicit;values=e.normalized_sequence;hidden=~b.observed_mask & e.sequence_validity
                target=b.normalized_target
                assert torch.equal(values[b.observed_mask],b.normalized_observed[b.observed_mask])
                branch=model.repair.recent_mean_projector.recent_band_network
                hook=branch.register_forward_hook(lambda module,args,value:torch.zeros_like(value))
                try:
                    ablated=model.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                finally:
                    hook.remove()
                variants={'base':values,'disabled_band':ablated.explicit.normalized_sequence}
                for label,v in variants.items():
                    assert torch.equal(v[b.observed_mask],values[b.observed_mask])
                    patches=patchify_preserving_values(v,b.observed_mask,16,8,validity=e.sequence_validity)
                    prediction=model.predictor(patches.values.masked_fill(~patches.validity,0),patches.effective_mask,
                        b.query_mean,b.query_std,patch_validity=patches.validity)
                    residual=(prediction-b.forecast_target).double();assert torch.isfinite(residual).all()
                    totals[label][0]+=residual.square().sum().item();totals[label][1]+=residual.abs().sum().item();totals[label][2]+=residual.numel()
                for label,v in variants.items():
                    error=(target-v)*b.query_std[:,None]
                    for region,err,mask in [('all_hidden',error,hidden),('last32_hidden',error[:,-32:],hidden[:,-32:]),('last16_hidden',error[:,-16:],hidden[:,-16:])]:
                        residual=err[mask].double()
                        repair[label][region][0]+=residual.square().sum().item()
                        repair[label][region][1]+=residual.abs().sum().item()
                        repair[label][region][2]+=residual.numel()
        assert before==(parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict()))
        assert file_sha256(selection)==before_selection and file_sha256(Path(record['checkpoint']))==record['checkpoint_sha256']
        metrics={k:dict(mse=s/n,mae=a/n,target_elements=n) for k,(s,a,n) in totals.items()}
        assert all(abs(metrics['base'][k]-record[f'best_val_{k}'])<1e-6 for k in ['mse','mae'])
        result=dict(scene=scene['name'],split='fixed_validation_only',forecast_interventions=metrics,
            repair_regions={label:{k:dict(mse=s/n,mae=a/n,hidden_coordinates=n) for k,(s,a,n) in regions.items()} for label,regions in repair.items()},
            checkpoint_sha256=record['checkpoint_sha256'],selection_manifest=str(selection),selection_sha256=before_selection,
            interpretation='disable v80 new band only, same frozen trained head/masks/statistics; not retrained parent v79',
            training_performed=False,model_state_unchanged=True)
        results.append(result);print(json.dumps(result),flush=True)
        del model,memory,out,b;torch.cuda.empty_cache()
    with (ROOT/'branch-ablation-validation.json').open('x') as f:json.dump(results,f,indent=2);f.write('\n')

if __name__=='__main__':main()
