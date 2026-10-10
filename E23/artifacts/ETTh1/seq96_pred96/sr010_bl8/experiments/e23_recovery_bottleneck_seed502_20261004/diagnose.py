"""Read-only diagnosis of existing checkpoints on fixed validation histories."""
from pathlib import Path
import csv,json,hashlib
from time import perf_counter
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_repair_model,build_forecast_model,load_stage1_selection,load_stage2_selection,_new_memory,_starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.author_profiles import build_author_model,load_author_profiles
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history
from scripts.run_author_aligned_p4 import _autocast
from malr.migration import MigrationCatalog

OUT=Path(__file__).resolve().parent
W=Path('/home/xyw/paper/.e23-v43-nonoverlap/E23')
EXPS=OUT.parent

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (OUT/'summary.json').exists(),'Refusing duplicate diagnosis'
    started=perf_counter();torch.set_num_threads(4);device=torch.device('cuda:1')
    protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load(W.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=W/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=W.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
    settings=load_e23_settings(EXPS/'e23_v40_local_point_refinement_seed502_20261004/tail32/e23_seed502.yaml',protocol=protocol)
    paths={'E23_stage1':EXPS/'e23_v40_local_point_refinement_seed502_20261004/tail32/stage1/runs/e23-stage1.json',
           'E23_v41_slow_stage2':EXPS/'e23_v41_repair_drift_seed502_20261004/tail_slow/stage2/runs/e23-stage2.json',
           'T1_stage1':EXPS/'e23_t1_five_seed_standard_20261004/seed_502/T1/training/repair-selection.json'}
    models={};identities={}
    for name,path in paths.items():
        record=read(path);identities[str(path)]=digest(path);identities[record['checkpoint']]=digest(record['checkpoint'])
        if name.startswith('E23'):
            if name=='E23_stage1':
                payload=load_stage1_selection(path,protocol,settings,device=device);repair=build_repair_model(protocol,settings,device=device);repair.load_state_dict(payload['model_state'])
            else:
                payload=load_stage2_selection(path,protocol,settings,device=device);model=build_forecast_model(protocol,settings,device=device);model.load_state_dict(payload['model_state']);repair=model.repair
            repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
            memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot());models[name]=repair.eval()
        else:
            profiles=load_author_profiles(EXPS/'e23_t1_five_seed_standard_20261004/configs/author_seed_502.json');cfg=profiles['models']['T1']
            model=build_author_model('T1',cfg,W/cfg['root'],device);model.load_state_dict(torch.load(record['checkpoint'],map_location=device,weights_only=False)['model_state']);models[name]=model.eval()
    hashes={n:parameter_hash(m) for n,m in models.items()}
    groups=['all','gap_lt8','gap_8to15','gap_ge16','all_covering_patches_fully_hidden','no_H_in_any_covering_patch','last32','before64']
    totals={n:{g:[0.,0.,0] for g in groups} for n in models}
    trends={n:[0.,0.,0] for n in models};variables={n:torch.zeros(7,2,dtype=torch.float64) for n in models}
    patch_counts=dict(damaged=0,fully_hidden=0,fully_hidden_H_available=0)
    windows=0
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
            b=cpu.to(device);hidden=~b.observed_mask;pos=torch.arange(96,device=device)[None,:,None]
            prev=torch.where(b.observed_mask,pos,-1).cummax(1).values
            nex=torch.where(b.observed_mask,pos,96).flip(1).cummin(1).values.flip(1);gap=nex-prev-1
            errors={};evidence={}
            for n,m in models.items():
                if n.startswith('E23'):
                    starts=_starts(b,stride=8,patches=12,device=device)
                    out=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                    errors[n]=(out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None]
                    evidence[n]=out
                else:
                    value,mask=t1_inputs(b)
                    with _autocast(cfg['training'],device):candidate=m(value,mask=mask)
                    errors[n]=candidate.float()-t1_clean_history(b)
            base=evidence['E23_stage1'];B,C,N,_=base.router.latents.shape
            empty=~base.explicit.observation_patches.any(-1)
            unavailable=~base.evidence['historical'].available
            assert torch.equal(unavailable,~evidence['E23_v41_slow_stage2'].evidence['historical'].available)
            ref=models['E23_stage1'].point_refiner
            def fold(v):
                return ref._points(v[...,None].float(),base.explicit.validity_patches,base.explicit.coverage).reshape(B,C,96).transpose(1,2)
            masks={'all':hidden,'gap_lt8':hidden&(gap<8),'gap_8to15':hidden&(gap>=8)&(gap<16),
                'gap_ge16':hidden&(gap>=16),'all_covering_patches_fully_hidden':hidden&fold(empty).eq(1),
                'no_H_in_any_covering_patch':hidden&fold(unavailable).eq(1),'last32':hidden&(pos>=64),'before64':hidden&(pos<64)}
            patch_counts['damaged']+=base.explicit.observation_patches.ne(base.explicit.validity_patches).any(-1).sum().item()
            patch_counts['fully_hidden']+=empty.sum().item()
            patch_counts['fully_hidden_H_available']+=(empty&base.evidence['historical'].available).sum().item()
            for n,error in errors.items():
                for g,mask in masks.items():
                    e=error[mask].double();totals[n][g][0]+=e.square().sum().item();totals[n][g][1]+=e.abs().sum().item();totals[n][g][2]+=e.numel()
                e=error.masked_fill(~hidden,0).double();variables[n][:,0]+=e.square().sum((0,1)).cpu();variables[n][:,1]+=hidden.sum((0,1)).cpu()
                recent=e[:,64:];t=torch.arange(32,device=device,dtype=torch.float64)-15.5
                trends[n][0]+=recent.mean(1).square().sum().item()
                trends[n][1]+=(recent*t[None,:,None]).sum(1).div(t.square().sum()).square().sum().item()
                trends[n][2]+=recent.shape[0]*recent.shape[2]
            windows+=B
    assert windows==2688
    expected={'E23_stage1':.2774709515418482,'E23_v41_slow_stage2':.3586186865083281,'T1_stage1':.1521003861546415}
    rows=[]
    for n,data in totals.items():
        total_error=data['all'][0];count=data['all'][2]
        assert abs(total_error/count-expected[n])<1e-6,(n,total_error/count)
        for g,v in data.items():rows.append(dict(model=n,region=g,mse=v[0]/v[2],mae=v[1]/v[2],hidden_coordinates=v[2],
            hidden_coordinate_fraction=v[2]/count,squared_error_fraction=v[0]/total_error))
    assert hashes=={n:parameter_hash(m) for n,m in models.items()}
    assert all(digest(p)==h for p,h in identities.items())
    report=dict(status='complete',split='original_fixed_validation',training_seed=502,windows=windows,metric_space='train_standardized',
        rows=rows,patch_counts=patch_counts,trend_diagnostics={n:dict(last32_level_mse=v[0]/v[2],last32_slope_mse=v[1]/v[2]) for n,v in trends.items()},
        per_variable={n:{k:dict(mse=v[i,0].item()/v[i,1].item(),error_fraction=v[i,0].item()/totals[n]['all'][0]) for i,k in enumerate(['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'])} for n,v in variables.items()},
        no_training=True,no_test_evaluation=True,no_checkpoint_selection=True,weights_unchanged=True,seconds=perf_counter()-started,
        limitations='Long gaps and H unavailability co-occur; these groups do not prove an H intervention benefit. Group regions overlap.',identities=identities)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'groups.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:report[k] for k in ['patch_counts','trend_diagnostics','per_variable','seconds']}))
    for row in rows:print(json.dumps(row))

if __name__=='__main__':main()
