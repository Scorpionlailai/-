"""Saved-checkpoint recovery diagnostics on unique validation coordinates."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import (_starts,_new_memory,_key_sha256,build_repair_model,
    build_forecast_model,load_stage1_selection,load_stage2_selection)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog

WORK=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda:1')
    spec=json.loads((args.cell/'plan.json').read_text())
    protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
    settings=load_e23_settings(Path(spec['config']),protocol=protocol)
    stage1=Path(spec['cached_stage1']) if spec.get('cached_stage1') else args.cell/'stage1/runs/e23-stage1.json'
    paths={'stage1':stage1,'stage2_mae':args.cell/'stage2/runs/e23-stage2.json'}
    rows=[]
    for stage,path in paths.items():
        started=perf_counter();record=json.loads(path.read_text())
        if stage=='stage1':
            payload=load_stage1_selection(path,protocol,settings,device=device)
            repair=build_repair_model(protocol,settings,device=device);repair.load_state_dict(payload['model_state'])
        else:
            payload=load_stage2_selection(path,protocol,settings,device=device)
            model=build_forecast_model(protocol,settings,device=device);model.load_state_dict(payload['model_state']);repair=model.repair
        repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
        before=memory.snapshot_hash(),_key_sha256(repair.historical.retrieval.state_dict())
        repair.eval();sums={name:[0.,0.,0] for name in ['all96','before64','last32','block_boundaries']}
        level=slope=0.;trend_count=0;h_available=h_damaged=h_empty_available=0
        point_deviation=point_variance=copy_deviation=0.;point_count=copy_count=0
        candidate_entropy=candidate_variance=candidate_null=0.;candidate_count=0
        time_entropy=time_null=time_distance=time_visible_mass=0.;time_count=0
        with torch.inference_mode():
            for cpu_batch in iter_p4_batches(protocol,'validation',batch_size=128):
                b=cpu_batch.to(device);starts=_starts(b,stride=repair.stride,patches=repair.patch_count,device=device)
                out=repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                assert torch.equal(out.explicit.sequence_observed,b.observed_mask)
                assert torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
                if repair.stride==repair.patch_length:assert out.explicit.coverage.eq(1).all()
                patch_hidden=out.explicit.validity_patches & ~out.explicit.observation_patches
                if out.temporal_evidence_attention is not None:
                    tw=out.temporal_evidence_attention["weights"]
                    tc=(~b.observed_mask[:,64:]).transpose(1,2).reshape(-1,32)[:,None,None].expand_as(tw[...,0])
                    forbidden=~out.temporal_evidence_attention["source_observed"][:,None,None,None].expand_as(tw[...,1:])
                    assert tw[...,1:][forbidden].eq(0).all()
                    time_entropy+=(-(tw*tw.clamp_min(1e-8).log()).sum(-1))[tc].double().sum().item()
                    time_null+=tw[...,0][tc].double().sum().item()
                    distance=(torch.arange(96,device=device)[None]-torch.arange(64,96,device=device)[:,None]).abs()
                    time_distance+=(tw[...,1:]*distance).sum(-1)[tc].double().sum().item()
                    time_visible_mass+=(1-tw[...,0])[tc].double().sum().item();time_count+=tc.sum().item()
                if out.point_candidate_attention is not None:
                    a=out.point_candidate_attention['weights']
                    chosen=patch_hidden[...,None,:].expand_as(a[...,0])
                    candidate_entropy+=(-(a*a.clamp_min(1e-8).log()).sum(-1))[chosen].double().sum().item()
                    candidate_null+=a[...,0][chosen].double().sum().item()
                    m=chosen[...,None].to(a.dtype)
                    mean=(a*m).sum(-2,keepdim=True)/m.sum(-2,keepdim=True).clamp_min(1)
                    candidate_variance+=((a-mean).square()*m).sum(-1).double().sum().item()
                    candidate_count+=chosen.sum().item()
                if out.router.point_weights is not None:
                    w=out.router.point_weights;prior=out.router.weights[...,None,:].expand_as(w)
                    point_deviation+=(w-prior).abs()[patch_hidden].double().sum().item()
                    m=patch_hidden[...,None].to(w.dtype)
                    average=(w*m).sum(-2,keepdim=True)/m.sum(-2,keepdim=True).clamp_min(1)
                    point_variance+=((w-average).square()*m).double().sum().item()
                    point_count+=patch_hidden.sum().item()*4
                if out.explicit.aggregation_weights is not None:
                    copy_deviation+=(out.explicit.aggregation_weights[patch_hidden]-1).abs().double().sum().item()
                    copy_count+=patch_hidden.sum().item()
                error=(out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None]
                hidden=~b.observed_mask
                position=torch.arange(96,device=device)[None,:,None]
                boundary=((position%repair.patch_length==repair.patch_length-1)&(position<95))|((position%repair.patch_length==0)&(position>0))
                e=error.masked_fill(~hidden,0).double()[:,64:]
                centered=torch.arange(32,device=device,dtype=torch.float64)-15.5
                level+=e.mean(1).square().sum().item()
                slope+=(e*centered[None,:,None]).sum(1).div(centered.square().sum()).square().sum().item()
                trend_count+=e.shape[0]*e.shape[2]
                damaged=out.explicit.observation_patches.ne(out.explicit.validity_patches).any(-1)
                empty=~out.explicit.observation_patches.any(-1)
                available=out.evidence['historical'].available
                h_available+=(available&damaged).sum().item();h_damaged+=damaged.sum().item()
                h_empty_available+=(available&empty).sum().item()
                for name,mask in [('all96',hidden),('before64',hidden&(position<64)),('last32',hidden&(position>=64)),('block_boundaries',hidden&boundary)]:
                    e=error[mask].double();sums[name][0]+=e.square().sum().item();sums[name][1]+=e.abs().sum().item();sums[name][2]+=e.numel()
        assert before==(memory.snapshot_hash(),_key_sha256(repair.historical.retrieval.state_dict()))
        rows.append(dict(stage=stage,selection_manifest=str(path),selection_sha256=file_sha256(path),checkpoint_sha256=record['checkpoint_sha256'],
            selected_epoch=record['best_epoch'],metric_space='train_standardized',
            metrics={name:dict(mse=v[0]/v[2],mae=v[1]/v[2],hidden_coordinates=v[2]) for name,v in sums.items()},
            repair_stride=repair.stride,repair_patch_count=repair.patch_count,repair_parameters=sum(p.numel() for p in repair.parameters()),
            history_sampling_stride=memory.patch_stride,historical_query_length=getattr(repair.historical,'query_length',repair.patch_length),
            repair_patch_length=repair.patch_length,last32_level_mse=level/trend_count,last32_slope_mse=slope/trend_count,
            h_available_damaged_patches=h_available,damaged_patches=h_damaged,h_available_fully_hidden_patches=h_empty_available,
            point_weight_absolute_deviation=None if not point_count else point_deviation/point_count,
            point_weight_within_patch_variance=None if not point_count else point_variance/point_count,
            overlap_weight_absolute_deviation=None if not copy_count else copy_deviation/copy_count,
            point_candidate_entropy=None if not candidate_count else candidate_entropy/candidate_count,
            point_candidate_null_mass=None if not candidate_count else candidate_null/candidate_count,
            point_candidate_within_patch_variance=None if not candidate_count else candidate_variance/candidate_count,
            temporal_evidence_entropy=None if not time_count else time_entropy/time_count,
            temporal_evidence_null_mass=None if not time_count else time_null/time_count,
            temporal_evidence_visible_mean_distance=None if not time_visible_mass else time_distance/time_visible_mass,
            bank_state_unchanged=True,diagnostic_seconds=perf_counter()-started))
    output=args.cell/'recovery-diagnostics.json'
    with output.open('x') as f:json.dump(rows,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
