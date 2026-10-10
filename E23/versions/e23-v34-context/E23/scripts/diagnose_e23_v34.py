"""Validation-only repair, H coverage, and context provenance diagnostics."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.report_e23_v31_trial import digest
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (load_stage1_selection,load_stage2_selection,
    build_repair_model,build_forecast_model,_new_memory,_starts,_key_sha256)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_with_state


def diagnose(config,cell,device):
    protocol=load_protocol(); settings=load_e23_settings(config,protocol=protocol); rows={}
    for stage in ('stage1','stage2'):
        started=perf_counter(); selection=cell/stage/'runs'/f'e23-{stage}.json'
        record=json.loads(selection.read_text())
        payload=(load_stage1_selection if stage=='stage1' else load_stage2_selection)(selection,protocol,settings,device=device)
        model=(build_repair_model if stage=='stage1' else build_forecast_model)(protocol,settings,device=device)
        model.load_state_dict(payload['model_state']); repair=model if stage=='stage1' else model.repair
        repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory=_new_memory(protocol); memory.load_state_dict(payload['memory_state'])
        repair.historical.refresh_snapshot(memory.snapshot()); model.eval()
        before=(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
        squared=absolute=0.; hidden=windows=damaged=fully_hidden=h_available=full_h_available=0
        h_weight=0.; causal_violations=0
        members=[m for groups in memory.snapshot().groups for group in groups for m in group.members]
        tables=[]
        for v in range(7):
            selected=sorted([m for m in members if m.variable_id==v],key=lambda m:m.start)
            tables.append((torch.tensor([m.start for m in selected],device=device),
                           torch.tensor([m.evidence_end for m in selected],device=device)))
        with torch.inference_mode():
            for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
                b=cpu.to(device); starts=_starts(b,stride=8,patches=12,device=device)
                out=repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                err=((out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None])[~b.observed_mask]
                squared+=err.double().square().sum().item(); absolute+=err.double().abs().sum().item()
                hidden+=err.numel(); windows+=len(b.normalized_observed)
                patches=patchify_with_state(b.normalized_observed,b.observed_mask,16,8)
                d=patches.rho.lt(1); full=~patches.effective_mask.any(-1)&patches.validity.any(-1)
                h=out.evidence['historical']; damaged+=d.sum().item(); fully_hidden+=full.sum().item()
                h_available+=(h.available&d).sum().item(); full_h_available+=(h.available&full).sum().item()
                h_weight+=(out.router.weights[...,1]*d).double().sum().item()
                candidate_starts=h.provenance['candidate_starts']; valid=h.provenance['candidate_valid']
                for v,(member_starts,ends) in enumerate(tables):
                    if not len(member_starts): continue
                    ix=torch.searchsorted(member_starts,candidate_starts[:,v]).clamp_max(len(member_starts)-1)
                    if not torch.equal(member_starts[ix][valid[:,v]],candidate_starts[:,v][valid[:,v]]):
                        raise ValueError('retrieved candidate does not exist in saved bank')
                    causal_violations+=((ends[ix]>starts[:,:,None])&valid[:,v]).sum().item()
        assert before==(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
        assert windows==2688 and hidden==1023670 and causal_violations==0
        global_mse=squared/hidden
        if stage=='stage1': assert abs(global_mse-record['best_val_repair_mse'])<1e-6
        rows[stage]=dict(stage=stage,checkpoint_sha256=record['checkpoint_sha256'],selection_sha256=digest(selection),
            global_mse=global_mse,global_mae=absolute/hidden,hidden_coordinates=hidden,windows=windows,
            damaged_patches=damaged,fully_hidden_patches=fully_hidden,h_available_patches=h_available,
            h_available_fraction=h_available/damaged,fully_hidden_h_available_patches=full_h_available,
            fully_hidden_h_available_fraction=full_h_available/fully_hidden if fully_hidden else 0,
            mean_h_route_weight=h_weight/damaged,causal_violations=causal_violations,
            h_key_sha256=before[0],h_bank_sha256=before[1],h_state_unchanged=True,
            bank_members=len(members),bank_groups=sum(len(g) for g in memory.snapshot().groups),
            mean_bank_context_visible_fraction=(sum(m.context_mask.mean() for m in members)/len(members)
                if members and members[0].context_mask is not None else None),
            repair_parameters=sum(p.numel() for p in repair.parameters()),
            predictor_parameters=0 if stage=='stage1' else sum(p.numel() for p in model.predictor.parameters()),
            split='original_fixed_validation',metric_space='train_standardized',diagnostic_seconds=perf_counter()-started)
        del model,repair,memory,out,payload; torch.cuda.empty_cache()
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cell',type=Path,required=True); p.add_argument('--device',default='cuda:1')
    a=p.parse_args(); torch.set_num_threads(4)
    output=a.cell/'historical-diagnostics.json'
    if output.exists(): raise FileExistsError(output)
    result=diagnose(a.config,a.cell,torch.device(a.device))
    output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
