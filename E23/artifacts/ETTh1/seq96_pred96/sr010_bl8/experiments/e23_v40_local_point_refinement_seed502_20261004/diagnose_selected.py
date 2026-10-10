"""After both trainings: paired pre/post refinement on fixed validation only."""
from dataclasses import replace
import csv
import json
from pathlib import Path
from time import perf_counter
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (_new_memory,_key_sha256,_starts,build_repair_model,
    build_forecast_model,load_stage1_selection,load_stage2_selection)
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed,iter_p4_batches
from malr.models.explicit_repair import build_explicit_repair
from malr.data.patching import patchify_with_state

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v40-point/E23')
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
VARIABLES=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']


@torch.inference_mode()
def main():
    output=ROOT/'selected-repair-diagnostics.json'
    if output.exists():raise FileExistsError(output)
    assert all((ROOT/s/'completed-audit.json').exists() for s in ['tail32','full96'])
    torch.set_num_threads(4);device=torch.device('cuda:1');start=perf_counter()
    protocol=with_training_seed(load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),502)
    result={};region_rows=[]
    definitions=[('E23-v31-B',FORMAL/'configs/e23_seed_502.yaml',FORMAL/'seed_502/E23-v31-B')]
    definitions += [('E23-v40-'+s,ROOT/s/'e23_seed502.yaml',ROOT/s) for s in ['tail32','full96']]
    for label,config,cell in definitions:
        settings=load_e23_settings(config,protocol=protocol);result[label]={}
        for stage in ['stage1','stage2']:
            manifest=cell/stage/'runs'/f'e23-{stage}.json';record=json.loads(manifest.read_text())
            loader=load_stage1_selection if stage=='stage1' else load_stage2_selection
            payload=loader(manifest,protocol,settings,device=device)
            model=(build_repair_model if stage=='stage1' else build_forecast_model)(protocol,settings,device=device)
            model.load_state_dict(payload['model_state'],strict=True);model.eval().requires_grad_(False)
            repair=model if stage=='stage1' else model.repair
            repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
            memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
            before=(parameter_hash(model),_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
            stats={kind:torch.zeros(42,3,dtype=torch.float64,device=device) for kind in ['with_refiner','without_refiner']}
            forecast_sums={kind:[0.,0.,0] for kind in stats}
            correction_sum=correction_count=0
            for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
                b=cpu.to(device);out=repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,
                    _starts(b,stride=8,patches=12,device=device))
                if repair.point_refiner is not None:
                    patches=patchify_with_state(b.normalized_observed,b.observed_mask,16,8)
                    base=build_explicit_repair(repair.decoder(out.router.latents),
                        patches.values.masked_fill(~patches.validity,0),patches.effective_mask,patches.validity,
                        b.query_mean,b.query_std,window_len=96,patch_len=16,stride=8)
                    base=replace(base,repaired_patches=base.repaired_patches.masked_fill(~base.validity_patches,0))
                    check=repair.point_refiner(base,out.router,b.query_mean,b.query_std)
                    torch.testing.assert_close(check.normalized_sequence,out.explicit.normalized_sequence,rtol=0,atol=0)
                else:base=out.explicit
                for kind,x in [('with_refiner',out.explicit),('without_refiner',base)]:
                    error=(x.normalized_sequence-b.normalized_target)*b.query_std[:,None]
                    for band in range(6):
                        for c in range(7):
                            e=error[:,band*16:(band+1)*16,c][~b.observed_mask[:,band*16:(band+1)*16,c]]
                            stats[kind][band*7+c]+=torch.stack((e.square().double().sum(),e.abs().double().sum(),e.new_tensor(e.numel(),dtype=torch.float64)))
                    if stage=='stage2':
                        pred=model.predictor(x.repaired_patches,x.observation_patches,b.query_mean,b.query_std,patch_validity=x.validity_patches)
                        error=pred-b.forecast_target
                        forecast_sums[kind][0]+=error.square().double().sum().item()
                        forecast_sums[kind][1]+=error.abs().double().sum().item()
                        forecast_sums[kind][2]+=error.numel()
                delta=((out.explicit.normalized_sequence-base.normalized_sequence)*b.query_std[:,None])[~b.observed_mask]
                correction_sum+=delta.square().double().sum().item();correction_count+=delta.numel()
            arrays={kind:v.cpu().numpy() for kind,v in stats.items()}
            with_s=arrays['with_refiner'];no_s=arrays['without_refiner']
            metrics=dict(mse=float(with_s[:,0].sum()/with_s[:,2].sum()),mae=float(with_s[:,1].sum()/with_s[:,2].sum()),
                last32_mse=float(with_s[28:,0].sum()/with_s[28:,2].sum()),before64_mse=float(with_s[:28,0].sum()/with_s[:28,2].sum()),
                without_refiner_repair_mse=float(no_s[:,0].sum()/no_s[:,2].sum()),
                without_refiner_last32_repair_mse=float(no_s[28:,0].sum()/no_s[28:,2].sum()),
                global_correction_mean_square=correction_sum/correction_count,checkpoint_sha256=record['checkpoint_sha256'],
                model_memory_key_hashes_unchanged=True)
            if stage=='stage2':
                for kind,sums in forecast_sums.items():
                    metrics[kind+'_forecast_mse']=sums[0]/sums[2];metrics[kind+'_forecast_mae']=sums[1]/sums[2]
                assert abs(metrics['with_refiner_forecast_mse']-record['best_val_mse'])<1e-6
                assert abs(metrics['with_refiner_forecast_mae']-record['best_val_mae'])<1e-6
            else:assert abs(metrics['mse']-record['best_val_repair_mse'])<1e-6
            assert before==(parameter_hash(model),_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
            result[label][stage]=metrics
            for kind,s in arrays.items():
                for g in range(42):
                    region_rows.append(dict(model=label,stage=stage,intervention=kind,variable=VARIABLES[g%7],
                        history_start=g//7*16,history_end_exclusive=(g//7+1)*16,mse=float(s[g,0]/s[g,2]),
                        mae=float(s[g,1]/s[g,2]),hidden_coordinates=int(s[g,2])))
            print(json.dumps({'model':label,'stage':stage,'metrics':metrics}),flush=True)
            del model,repair,payload;torch.cuda.empty_cache()
    output.write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'selected-repair-regions.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(region_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(region_rows)
    (ROOT/'diagnostic-resources.json').write_text(json.dumps(dict(diagnostic_seconds=perf_counter()-start,
        device=str(device),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,new_test_evaluation=False,new_training=False),indent=2)+'\n')


if __name__=='__main__':main()
