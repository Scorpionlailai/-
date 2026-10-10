"""Check all possible sequential configurations before freezing the study."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, load_stage1_selection, _new_memory, _starts, stage1_objective
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v52_variable_calibration_sr015_bl8_seed502_20261005')


def main():
    torch.set_num_threads(4);device=torch.device('cuda:1');plan=json.loads((ROOT/'plan.json').read_text())
    p=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
    original=load_e23_settings(Path(plan['base_config']),protocol=p)
    parent_record=json.loads((Path(plan['initial_parent']['root'])/'stage2/runs/e23-stage2.json').read_text())
    checkpoint=load_stage1_selection(Path(parent_record['stage1_manifest']),p,original,device=device)
    assert (WORK/'src/malr/models/patchtst.py').read_bytes()==Path('/home/xyw/paper/.e23-v47-front-freeze/E23/src/malr/models/patchtst.py').read_bytes()
    b=next(iter_p4_batches(p,'validation',batch_size=4)).to(device)
    torch.manual_seed(502);base=build_forecast_model(p,original,device=device).eval()
    base.repair.load_state_dict(checkpoint['model_state'])
    memory=_new_memory(p);memory.load_state_dict(checkpoint['memory_state'])
    base.repair.historical.retrieval.load_state_dict(checkpoint['retrieval_state']);base.repair.historical.refresh_snapshot(memory.snapshot())
    starts=_starts(b,stride=8,patches=12,device=device)
    with torch.no_grad():reference,ref=base(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    rows=[]
    for key,path in plan['configs'].items():
        cfg=load_e23_settings(Path(path),protocol=p);torch.manual_seed(502)
        m=build_forecast_model(p,cfg,device=device)
        assert parameter_hash(m.predictor)==plan['predictor_initial_sha256']
        result=m.repair.load_state_dict(checkpoint['model_state'],strict=False)
        assert not result.unexpected_keys and all(k.startswith(('variable.calibration.',)) for k in result.missing_keys)
        m.repair.historical.retrieval.load_state_dict(checkpoint['retrieval_state']);m.repair.historical.refresh_snapshot(memory.snapshot());m.eval()
        with torch.no_grad():
            pred,out=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            assert torch.equal(reference,pred) and torch.equal(ref.explicit.normalized_sequence,out.explicit.normalized_sequence)
            if m.repair.point_router is not None:m.repair.point_router.score[-1].weight.normal_(std=.02)
            if m.repair.variable.calibration is not None:m.repair.variable.calibration[-1].weight.normal_(std=.02)
            pred,out=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            canary=torch.where(b.observed_mask,b.normalized_observed,torch.full_like(b.normalized_observed,1e9))
            pred2,_=m(canary,b.observed_mask,b.query_mean,b.query_std,starts)
            assert torch.equal(pred,pred2)
            assert torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
            if out.router.point_weights is not None:
                assert out.router.point_weights.masked_select(~out.router.available[...,None,:].expand_as(out.router.point_weights)).eq(0).all()
        m.repair.train();m.zero_grad(set_to_none=True)
        out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts,specialist_exposure=.2,specialist_seed=503)
        loss,_=stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.,device=device),
            retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std,structure_mode=cfg.stage1_structure)
        loss.backward();assert torch.isfinite(loss)
        assert all(torch.isfinite(x.grad).all() for x in m.repair.parameters() if x.grad is not None)
        if m.repair.variable.calibration is not None:assert m.repair.variable.calibration[-1].weight.grad.abs().sum()>0
        if m.repair.overlap_confidence is not None:assert m.repair.overlap_confidence.score[-1].weight.grad.abs().sum()>0
        for x in m.repair.historical.key_encoder.parameters():x.requires_grad_(False)
        opt=build_stage2_optimizer(m,mode='freeze_front_v1',learning_rate=1e-4,weight_decay=0)
        stage2_training_mode(m,'freeze_front_v1')
        assert not m.repair.front_context.training and not any(x.requires_grad for x in m.repair.front_context.parameters())
        assert m.repair.point_router is None or all(x.requires_grad for x in m.repair.point_router.parameters())
        rows.append(dict(config_key=key,repair_parameters=sum(x.numel() for x in m.repair.parameters()),
            stage2_trainable_repair_parameters=sum(x.numel() for x in m.repair.parameters() if x.requires_grad),
            predictor_initial_sha256=plan['predictor_initial_sha256'],finite_loss=float(loss.detach())))
        del m,opt
    result=dict(status='passed',configurations=rows,original_initial_function_identical=True,
        hidden_input_canary_independent=True,visible_values_preserved=True,expert_eligibility_preserved=True,
        frontend_frozen_in_stage2=True,original_predictor_source_identical=True,predictor_parameters=35427,
        old_bank_used_for_preflight_only=True,fresh_training_required=True,no_test_evaluation=True)
    with (ROOT/'preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
