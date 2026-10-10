"""Real training batch, observed-only context bank and unchanged-head checks."""
import ast
import hashlib
import json
from pathlib import Path
from dataclasses import replace
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (build_forecast_model, _new_memory, _starts,
    _stage_observed, _publish_epoch, _verify_bank_key_pair, _retrieval_loss,
    _key_sha256, _memory_state_sha256, stage1_objective)
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = dict(A=ROOT/'configs/e23_multi_expert_p4_global_patch_keys_v34.yaml',
               B=ROOT/'configs/e23_multi_expert_p4_contextual_memory_v34.yaml')
PREDICTOR_SHA = '18ad18b5af292a9b526233ea426f81ffb3899d1f4fc850441060843d8adc404a'


def source_node(path,name):
    source=Path(path).read_text()
    node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
    return ast.dump(node,include_attributes=False)


def main():
    root=ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v34_contextual_memory'
    baseline=Path('/home/xyw/paper/.e23-v32-preservation/E23')
    for path,name in [('integrations/e23/train.py','train_forecast'),
                      ('integrations/e23/train.py','build_forecast_model'),
                      ('src/malr/models/e23/model.py','E23ForecastModel')]:
        assert source_node(ROOT/path,name)==source_node(baseline/path,name),(path,name)
    hashes=json.loads((root/'baseline-source-files.json').read_text())
    head_files=[n for n in hashes if n.startswith('src/malr/vendor/patchtst/') or
                n in ('src/malr/models/patchtst.py','src/malr/models/repaired_forecast.py')]
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==hashes[n] for n in head_files)
    protocol=load_protocol(); device=torch.device('cuda:1'); torch.set_num_threads(4)
    batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    rows=[]
    original_shared=None
    for condition,config in CONFIGS.items():
        torch.manual_seed(2021); settings=load_e23_settings(config,protocol=protocol)
        model=build_forecast_model(protocol,settings,device=device)
        assert parameter_hash(model.predictor)==PREDICTOR_SHA
        # All unchanged model tensors must initialize identically across A and B.
        shared={k:v.detach().clone() for k,v in model.repair.state_dict().items()
                if not k.startswith('historical.key_encoder.')}
        if original_shared is None: original_shared=shared
        else: assert all(torch.equal(v,original_shared[k]) for k,v in shared.items())
        memory=_new_memory(protocol); repair=model.repair
        starts=_starts(batch,stride=8,patches=12,device=device)
        repair.historical.retrieval.begin_epoch(1)
        staged=_stage_observed(memory,repair,batch,starts,set()); assert staged>0
        changed_target=batch.normalized_target.clone(); changed_target[~batch.observed_mask]=12345
        spoofed=_new_memory(protocol)
        _stage_observed(spoofed,repair,replace(batch,normalized_target=changed_target),starts,set())
        assert _memory_state_sha256(spoofed.state_dict())==_memory_state_sha256(memory.state_dict())
        _publish_epoch(memory,repair,1); _verify_bank_key_pair(memory,repair)
        restored=_new_memory(protocol); restored.load_state_dict(memory.state_dict())
        assert restored.snapshot_hash()==memory.snapshot_hash()
        before=(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
        model.eval()
        forecast,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        retrieval=_retrieval_loss(repair,batch)
        loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,
            retrieval,retrieval_weight=.1,loss_mode='global_mse_v1',query_std=batch.query_std)
        loss=loss+(forecast-batch.forecast_target).square().mean(); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert repair.historical.key_encoder.network[0].weight.grad.norm()>0
        assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
        assert torch.equal(out.explicit.sequence_observed,batch.observed_mask)
        assert before==(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
        with torch.no_grad():
            altered=batch.normalized_observed.clone(); altered[~batch.observed_mask]=12345
            other,_=model(altered,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            assert torch.equal(forecast,other)
        members=[m for groups in memory.snapshot().groups for group in groups for m in group.members]
        if condition=='B':
            for m in members:
                assert (m.context_values[~m.context_mask]==0).all()
                assert m.evidence_end<=protocol.config.data.datasets['ETTh1'].train_end
        rows.append(dict(condition=condition,query_mode=settings.historical_query_mode,
            staged=staged,bank_members=memory.snapshot().member_count,loss=float(loss),
            repair_parameters=sum(p.numel() for p in repair.parameters()),
            predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
            predictor_initial_sha256=PREDICTOR_SHA,hidden_input_invariant=True,
            bank_and_key_unchanged=True,formal_reuses_preflight_weights=False))
        del model,repair,memory,restored,spoofed,out,forecast,loss,other; torch.cuda.empty_cache()
    record=dict(conditions=rows,unchanged_head_sources=head_files,stage2_ast_unchanged=True)
    (root/'cuda-preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__': main()
