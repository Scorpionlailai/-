"""Check fixed predictor, cropped history, input isolation, gradients and dense residuals."""
import hashlib,json
from pathlib import Path
import torch

from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,_new_memory,_starts,_retrieval_loss,stage1_objective
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v45_plain_granularity_seed502_20261004')

def main():
    torch.set_num_threads(4);device=torch.device('cuda:1')
    plan=json.loads((ROOT/'plan.json').read_text())
    protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
    batch=next(iter_p4_batches(protocol,'validation',batch_size=4)).to(device)
    reference_path=ROOT.parent/'e23_v43_nonoverlap_seed502_20261004/nonoverlap_slow/stage1/runs/e23-stage1.json'
    reference=json.loads(reference_path.read_text());reference_hash=hashlib.sha256(Path(reference['checkpoint']).read_bytes()).hexdigest()
    old=torch.load(reference['checkpoint'],map_location=device,weights_only=False)
    rows=[];model_hashes={};head_file=WORK/'src/malr/models/patchtst.py'
    head_reference=Path('/home/xyw/paper/.e23-v43-nonoverlap/E23/src/malr/models/patchtst.py')
    assert head_file.read_bytes()==head_reference.read_bytes()
    for spec in plan['cells']:
        settings=load_e23_settings(Path(spec['config']),protocol=protocol)
        torch.manual_seed(502);model=build_forecast_model(protocol,settings,device=device)
        assert parameter_hash(model.predictor)==plan['predictor_initial_sha256']
        assert sum(p.numel() for p in model.predictor.parameters())==35427
        model_hashes[spec['name']]=parameter_hash(model.repair)
        h=model.repair.historical;h.retrieval.load_state_dict(old['retrieval_state'])
        memory=_new_memory(protocol);memory.load_state_dict(old['memory_state']);h.refresh_snapshot(memory.snapshot())
        starts=_starts(batch,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
        model.eval()
        with torch.no_grad():
            forecast,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            assert forecast.shape==(4,7,96) and torch.isfinite(forecast).all()
            assert torch.equal(out.explicit.sequence_observed,batch.observed_mask)
            assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
            assert out.explicit.coverage.eq(1).all()
            polluted=torch.where(batch.observed_mask,batch.normalized_observed,torch.full_like(batch.normalized_observed,12345))
            f2,o2=model(polluted,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            assert torch.equal(forecast,f2) and torch.equal(out.explicit.normalized_sequence,o2.explicit.normalized_sequence)
            assert model.repair.point_refiner is None
            assert not any(name.startswith("point_refiner.") for name,_ in model.repair.named_parameters())
            ev=out.evidence['historical'];valid=ev.provenance['candidate_valid']
            if valid.any():
                parent_starts=ev.provenance['parent_query_starts'][:,None,:,None].expand_as(valid)
                assert ((ev.provenance['parent_candidate_starts']+16<=parent_starts)|~valid).all()
        model.train();model.zero_grad(set_to_none=True)
        forecast,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,
                           specialist_exposure=.2,specialist_seed=503)
        loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,
            _retrieval_loss(model.repair,batch),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=batch.query_std)
        loss=loss+(forecast-batch.forecast_target).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        rows.append(dict(name=spec['name'],repair_parameters=sum(p.numel() for p in model.repair.parameters()),
            predictor_parameters=35427,repair_patch_count=model.repair.patch_count,
            historical_query_length=16,predictor_initial_sha256=plan["predictor_initial_sha256"],
            finite_loss=float(loss.detach()),h_available=int(ev.available.sum()),h_valid=int(valid.sum())))
    assert hashlib.sha256(Path(reference['checkpoint']).read_bytes()).hexdigest()==reference_hash
    result=dict(status='passed',cells=rows,no_final_cnn_or_attention=True,
                original_predictor_source_identical=True,hidden_inputs_isolated=True,
                no_test_evaluation=True,no_training_runs=True)
    with (ROOT/'preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
