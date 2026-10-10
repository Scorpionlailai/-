"""Fixed validation repair error and output drift; no training or extra tests."""
import csv
import json
from pathlib import Path
from time import perf_counter
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (build_repair_model,build_forecast_model,load_stage1_selection,
                                    load_stage2_selection,_new_memory,_starts,_key_sha256)
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed,iter_p4_batches

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v41-drift/E23')
VARIABLES=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']


def read(path): return json.loads(Path(path).read_text())


def attach(model,payload,protocol):
    model.load_state_dict(payload['model_state'],strict=True)
    repair=model.repair if hasattr(model,'repair') else model
    repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state'])
    repair.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False)
    return repair,memory


@torch.inference_mode()
def main():
    assert read(ROOT/'matrix-completed.json')['status']=='complete'
    torch.set_num_threads(4);device=torch.device('cuda:1');start=perf_counter()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    protocol=with_training_seed(load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),502)
    results={};rows=[]
    for trial in read(ROOT/'plan.json')['trials']:
        settings=load_e23_settings(Path(trial['config']),protocol=protocol)
        first=load_stage1_selection(Path(trial['stage1']),protocol,settings,device=device)
        before=build_repair_model(protocol,settings,device=device);_,memory1=attach(before,first,protocol)
        path=ROOT/trial['trial']/'stage2/runs/e23-stage2.json';record=read(path)
        selected=load_stage2_selection(path,protocol,settings,device=device)
        model=build_forecast_model(protocol,settings,device=device);repair,memory2=attach(model,selected,protocol)
        hashes=(parameter_hash(before),parameter_hash(model),memory1.snapshot_hash(),memory2.snapshot_hash(),
            _key_sha256(repair.historical.retrieval.state_dict()))
        stats={kind:torch.zeros(42,4,dtype=torch.float64,device=device) for kind in ['stage1','stage2']}
        delta_sum=0.;delta_count=0;forecast=[0.,0.,0]
        for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
            b=cpu.to(device);starts=_starts(b,stride=8,patches=12,device=device)
            old=before(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            prediction,out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            for kind,x in [('stage1',old.explicit.normalized_sequence),('stage2',out.explicit.normalized_sequence)]:
                error=(x-b.normalized_target)*b.query_std[:,None]
                for g in range(42):
                    band=g//7;c=g%7;hidden=~b.observed_mask[:,band*16:(band+1)*16,c]
                    e=error[:,band*16:(band+1)*16,c][hidden]
                    d=((x-old.explicit.normalized_sequence)*b.query_std[:,None])[:,band*16:(band+1)*16,c][hidden]
                    stats[kind][g]+=torch.stack((e.square().double().sum(),e.abs().double().sum(),
                        e.new_tensor(e.numel(),dtype=torch.float64),d.square().double().sum()))
                assert torch.equal(x[b.observed_mask],b.normalized_observed[b.observed_mask])
            delta=((out.explicit.normalized_sequence-old.explicit.normalized_sequence)*b.query_std[:,None])[~b.observed_mask]
            delta_sum+=delta.square().double().sum().item();delta_count+=delta.numel()
            e=prediction-b.forecast_target
            forecast[0]+=e.square().double().sum().item();forecast[1]+=e.abs().double().sum().item();forecast[2]+=e.numel()
        arrays={k:v.cpu().numpy() for k,v in stats.items()};metrics={}
        for kind,s in arrays.items():
            metrics[kind]=dict(mse=float(s[:,0].sum()/s[:,2].sum()),mae=float(s[:,1].sum()/s[:,2].sum()),
                last32_mse=float(s[28:,0].sum()/s[28:,2].sum()),before64_mse=float(s[:28,0].sum()/s[:28,2].sum()),
                last32_output_drift_mse=float(s[28:,3].sum()/s[28:,2].sum()))
            for g in range(42):rows.append(dict(model=trial['label'],stage=kind,variable=VARIABLES[g%7],
                history_start=g//7*16,history_end_exclusive=(g//7+1)*16,mse=float(s[g,0]/s[g,2]),
                mae=float(s[g,1]/s[g,2]),stage1_output_drift_mse=float(s[g,3]/s[g,2]),hidden_coordinates=int(s[g,2])))
        metrics.update(output_drift_mse=delta_sum/delta_count,validation_forecast_mse=forecast[0]/forecast[2],
            validation_forecast_mae=forecast[1]/forecast[2],checkpoint_sha256=record['checkpoint_sha256'],
            repair_state_identical=parameter_hash(before)==parameter_hash(repair))
        assert abs(metrics['validation_forecast_mse']-record['best_val_mse'])<1e-6
        assert abs(metrics['validation_forecast_mae']-record['best_val_mae'])<1e-6
        if trial['policy']=='frozen_v1': assert metrics['output_drift_mse']==0 and metrics['repair_state_identical']
        assert hashes==(parameter_hash(before),parameter_hash(model),memory1.snapshot_hash(),memory2.snapshot_hash(),
            _key_sha256(repair.historical.retrieval.state_dict()))
        results[trial['label']]=metrics;print(json.dumps(dict(model=trial['label'],metrics=metrics)),flush=True)
        del model,repair,before,first,selected;torch.cuda.empty_cache()
    with (ROOT/'selected-repair-diagnostics.json').open('x') as stream:json.dump(results,stream,indent=2)
    with (ROOT/'selected-repair-regions.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    torch.cuda.synchronize(device)
    with (ROOT/'diagnostic-resources.json').open('x') as stream:json.dump(dict(diagnostic_seconds=perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,new_training=False,new_test=False),stream,indent=2)


if __name__=='__main__':main()
