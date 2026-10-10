"""Discarded GPU optimizer checks on each cached Stage1, before any scored trial."""
import ast
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.e23.train import build_forecast_model,load_stage1_selection,_new_memory,_starts
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed,iter_p4_batches

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v41-drift/E23')
ORIGINAL=Path('/home/xyw/paper/.e23-t1-standard/E23')


def main():
    torch.set_num_threads(4);device=torch.device('cuda:1')
    protocol=with_training_seed(load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),502)
    batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    records=[]
    for trial in json.loads((ROOT/'plan.json').read_text())['trials']:
        settings=load_e23_settings(Path(trial['config']),protocol=protocol)
        selected=load_stage1_selection(Path(trial['stage1']),protocol,settings,device=device)
        torch.manual_seed(502);model=build_forecast_model(protocol,settings,device=device)
        assert parameter_hash(model.predictor)=='26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c'
        assert sum(p.numel() for p in model.predictor.parameters())==35427
        model.repair.load_state_dict(selected['model_state'],strict=True)
        h=model.repair.historical;h.retrieval.load_state_dict(selected['retrieval_state'])
        h.key_encoder.requires_grad_(False)
        memory=_new_memory(protocol);memory.load_state_dict(selected['memory_state']);h.refresh_snapshot(memory.snapshot())
        optimizer=build_stage2_optimizer(model,mode=trial['policy'],learning_rate=.0001,weight_decay=0)
        starts=_starts(batch,stride=8,patches=12,device=device)
        model.eval()
        with torch.no_grad():
            _,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            original=out.explicit.normalized_sequence.clone()
        state_before=parameter_hash(model.repair);head_before=parameter_hash(model.predictor)
        for _ in range(3):
            stage2_training_mode(model,trial['policy']);optimizer.zero_grad()
            prediction,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            loss=(prediction-batch.forecast_target).square().mean()+.2*(prediction-batch.forecast_target).abs().mean()
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert sum(p.grad.abs().sum().item() for p in model.predictor.parameters() if p.grad is not None)>0
            if trial['policy']=='frozen_v1':
                assert not model.repair.training and all(p.grad is None for p in model.repair.parameters())
                assert torch.equal(original,out.explicit.normalized_sequence)
            optimizer.step()
        assert parameter_hash(model.predictor)!=head_before
        frozen=parameter_hash(model.repair)==state_before
        assert frozen==(trial['policy']=='frozen_v1')
        records.append(dict(trial=trial['trial'],repair_state_exactly_preserved=frozen,head_learns=True,
            finite_gradients=True,optimizer_learning_rates=[g['lr'] for g in optimizer.param_groups],
            repair_trainable_parameters=sum(p.numel() for p in model.repair.parameters() if p.requires_grad)))
        del model,selected;torch.cuda.empty_cache()
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old=functions(ORIGINAL/'integrations/e23/train.py');new=functions(WORK/'integrations/e23/train.py')
    for name in ['build_forecast_model','stage2_objective']: assert old[name]==new[name]
    for rel in ['src/malr/models/patchtst.py']:
        assert (WORK/rel).read_bytes()==(ORIGINAL/rel).read_bytes()
    result=dict(status='passed',records=records,discarded_optimizer_steps=9,new_stage1_training=False,
        original_head_build_and_forecast_objective_unchanged=True)
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__': main()
