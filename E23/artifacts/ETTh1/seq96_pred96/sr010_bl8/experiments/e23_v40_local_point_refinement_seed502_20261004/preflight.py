"""Verify new repair capacity against original initialization and GPU gradients."""
import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import build_forecast_model,_starts
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed,iter_p4_batches

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v40-point/E23')
ORIGINAL=Path('/home/xyw/paper/.e23-t1-standard/E23')


def main():
    torch.set_num_threads(4);device=torch.device('cuda:1')
    protocol=with_training_seed(load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),502)
    settings=load_e23_settings(ROOT/'tail32/e23_seed502.yaml',protocol=protocol)
    torch.manual_seed(502);base=build_forecast_model(protocol,replace(settings,point_refiner_span=0),device=device).eval()
    expected='26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c'
    assert parameter_hash(base.predictor)==expected
    b=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    starts=_starts(b,stride=8,patches=12,device=device)
    records=[];branch_hashes=set()
    for label in ['tail32','full96']:
        settings=load_e23_settings(ROOT/label/'e23_seed502.yaml',protocol=protocol)
        torch.manual_seed(502);model=build_forecast_model(protocol,settings,device=device).eval()
        assert parameter_hash(model.predictor)==expected
        assert sum(p.numel() for p in model.predictor.parameters())==35427
        assert all(torch.equal(model.repair.state_dict()[k],v) for k,v in base.repair.state_dict().items())
        branch_hashes.add(parameter_hash(model.repair.point_refiner))
        with torch.no_grad():
            original,old=base(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            forecast,new=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            torch.testing.assert_close(original,forecast,rtol=0,atol=0)
            torch.testing.assert_close(old.explicit.normalized_sequence,new.explicit.normalized_sequence,rtol=0,atol=0)
        model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for step in range(3):
            optimizer.zero_grad()
            forecast,out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            error=(out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None]
            loss=error[~b.observed_mask].square().mean()
            if step==2:loss=loss+(forecast-b.forecast_target).square().mean()+.2*(forecast-b.forecast_target).abs().mean()
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert model.repair.point_refiner.output.weight.grad.abs().sum()>0
            if step:assert model.repair.point_refiner.input[0].weight.grad.abs().sum()>0
            optimizer.step()
        model.eval()
        with torch.no_grad():
            forecast,out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
            canary=b.normalized_observed.clone();canary[~b.observed_mask]=1e6
            other,other_out=model(canary,b.observed_mask,b.query_mean,b.query_std,starts)
            torch.testing.assert_close(forecast,other,rtol=0,atol=0)
            assert torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
            assert out.explicit.repaired_patches[~out.explicit.validity_patches].eq(0).all()
        records.append(dict(label=label,repair_parameters=sum(p.numel() for p in model.repair.parameters()),
            refiner_parameters=sum(p.numel() for p in model.repair.point_refiner.parameters()),
            forecast_parameters=35427,head_initial_sha256=expected,zero_start_forecast_exact=True,
            hidden_input_canary_unchanged=True,finite_gpu_gradients=True))
        del model;torch.cuda.empty_cache()
    assert len(branch_hashes)==1
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before=functions(ORIGINAL/'integrations/e23/train.py');after=functions(WORK/'integrations/e23/train.py')
    untouched=['build_forecast_model','stage2_objective','train_forecast']
    assert all(before[name]==after[name] for name in untouched)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    unchanged_files=['src/malr/models/patchtst.py','src/malr/models/repaired_forecast.py',
        'src/malr/models/e23/historical.py','src/malr/models/e23/temporal.py',
        'src/malr/models/e23/variable.py','src/malr/models/e23/router.py','src/malr/models/e23/iterative.py',
        'integrations/e23/forecast_sensitive.py']
    assert all(sha(WORK/p)==sha(ORIGINAL/p) for p in unchanged_files)
    record=dict(status='passed',trials=records,equal_branch_initial_sha256=next(iter(branch_hashes)),
        original_stage2_functions_unchanged=untouched,original_experts_and_head_files_unchanged=unchanged_files,
        note='three optimizer steps are discarded preflight checks; formal training starts fresh')
    output=ROOT/'preflight.json';assert not output.exists();output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
