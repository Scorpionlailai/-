"""Fixed-v41 predictor response to history-only oracle/Stage1 interventions."""
from pathlib import Path
import json,hashlib
from time import perf_counter
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_repair_model,build_forecast_model,load_stage1_selection,load_stage2_selection,_new_memory,_starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_preserving_values
from malr.migration import MigrationCatalog

OUT=Path(__file__).resolve().parent;W=Path('/home/xyw/paper/.e23-v43-nonoverlap/E23');EXPS=OUT.parent

def main():
    assert not (OUT/'interventions.json').exists()
    torch.set_num_threads(4);started=perf_counter();device=torch.device('cuda:1')
    protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load(W.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=W/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=W.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
    settings=load_e23_settings(EXPS/'e23_v40_local_point_refinement_seed502_20261004/tail32/e23_seed502.yaml',protocol=protocol)
    first_path=EXPS/'e23_v40_local_point_refinement_seed502_20261004/tail32/stage1/runs/e23-stage1.json'
    final_path=EXPS/'e23_v41_repair_drift_seed502_20261004/tail_slow/stage2/runs/e23-stage2.json'
    p1=load_stage1_selection(first_path,protocol,settings,device=device);p2=load_stage2_selection(final_path,protocol,settings,device=device)
    first=build_repair_model(protocol,settings,device=device);first.load_state_dict(p1['model_state'])
    model=build_forecast_model(protocol,settings,device=device);model.load_state_dict(p2['model_state'])
    for repair,payload in [(first,p1),(model.repair,p2)]:
        repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
    first.eval();model.eval();before=parameter_hash(first),parameter_hash(model)
    names=['baseline','restore_stage1_last32','oracle_half_last32','oracle_full_last32','oracle_level_last32','oracle_level_slope_last32','oracle_all96']
    totals={n:[0.,0.,0] for n in names};segments={n:[[0.,0.,0] for _ in range(3)] for n in names}
    pos=torch.arange(96,device=device)[None,:,None];t=torch.arange(32,device=device,dtype=torch.float64)-15.5
    with torch.inference_mode():
        for index,cpu in enumerate(iter_p4_batches(protocol,'validation',batch_size=128)):
            b=cpu.to(device);starts=_starts(b,stride=8,patches=12,device=device)
            _,out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            old=first(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts).explicit.normalized_sequence
            seq=out.explicit.normalized_sequence;hidden=~b.observed_mask;recent=hidden&(pos>=64)
            full=torch.where(recent,b.normalized_target,seq)
            level=seq.clone();mean_mask=hidden[:,64:].double();error=(seq[:,64:]-b.normalized_target[:,64:]).double()
            correction=-(error*mean_mask).sum(1)/mean_mask.sum(1).clamp_min(1)
            level[:,64:]+=(mean_mask*correction[:,None]).to(seq.dtype)
            basis=torch.stack((torch.ones_like(t),t),-1)
            gram=torch.einsum('btc,ti,tj->bcij',mean_mask,basis,basis)
            rhs=-torch.einsum('btc,ti->bci',error*mean_mask,basis)
            coefficients=(torch.linalg.pinv(gram,hermitian=True)@rhs[...,None]).squeeze(-1)
            joint=seq.clone();joint[:,64:]+=(mean_mask*torch.einsum('bci,ti->btc',coefficients,basis)).to(seq.dtype)
            cases={'baseline':seq,'restore_stage1_last32':torch.where(recent,old,seq),
                'oracle_half_last32':seq+.5*(full-seq),'oracle_full_last32':full,'oracle_level_last32':level,
                'oracle_level_slope_last32':joint,'oracle_all96':torch.where(hidden,b.normalized_target,seq)}
            for n,x in cases.items():
                assert torch.equal(x[b.observed_mask],seq[b.observed_mask])
                p=patchify_preserving_values(x,b.observed_mask,16,8,validity=out.explicit.sequence_validity)
                prediction=model.predictor(p.values.masked_fill(~p.validity,0),p.effective_mask,b.query_mean,b.query_std,patch_validity=p.validity)
                e=(prediction-b.forecast_target).double()
                v=[e.square().sum().item(),e.abs().sum().item(),e.numel()]
                for i in range(3):totals[n][i]+=v[i];segments[n][min(index//7,2)][i]+=v[i]
    assert before==(parameter_hash(first),parameter_hash(model))
    rows={n:dict(mse=v[0]/v[2],mae=v[1]/v[2]) for n,v in totals.items()}
    assert abs(rows['baseline']['mse']-.7379902117376695)<1e-6
    result=dict(model='E23-v41-tail32-slow',split='original_fixed_validation',training_seed=502,
        prediction_metrics=rows,three_contiguous_validation_segments={n:[dict(mse=v[0]/v[2],mae=v[1]/v[2]) for v in ss] for n,ss in segments.items()},
        input_observations_mask_query_stats_and_all_weights_fixed=True,no_training=True,no_test_evaluation=True,no_checkpoint_selection=True,
        oracle_is_not_deployable=True,level_slope_intervention='minimum-norm additive correction restricted to last32 hidden coordinates matching full last32 level/slope',
        model_intervention_not_physical_causality=True,seconds=perf_counter()-started)
    (OUT/'interventions.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
