"""Discarded optimizer checks; cached checkpoints and prediction head must remain exact."""
import ast
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.aligned_stage2 import reset_training_rng
from integrations.e23.train import build_forecast_model,load_stage1_selection,_new_memory,_starts
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed,iter_p4_batches
from integrations.forecast_compare.author_profiles import load_author_profiles,build_author_model
from integrations.forecast_compare.author_provenance import author_source_identity
from integrations.forecast_compare.t1_p4 import T1ForecastModel
from malr.data.patching import patchify_preserving_values
from malr.evaluation.report import file_sha256
from scripts.run_author_aligned_p4 import _forecast_head

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v42-mse/E23')
ORIGINAL=Path('/home/xyw/paper/.e23-t1-standard/E23')
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
TAIL=ROOT.parent/'e23_v40_local_point_refinement_seed502_20261004/tail32'
HEAD='26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c'


def main():
    torch.set_num_threads(4);device=torch.device('cuda:1')
    base=load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')
    cached=[]
    for seed in [2021,102,202,302,402,502]:
        config=WORK/'configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml' if seed==2021 else FORMAL/f'configs/e23_seed_{seed}.yaml'
        stage1=ROOT.parent/'e23_v31_forecast_sensitive/common_stage1/runs/e23-stage1.json' if seed==2021 else FORMAL/f'seed_{seed}/E23-v31-B/stage1/runs/e23-stage1.json'
        protocol=with_training_seed(base,seed);settings=load_e23_settings(config,protocol=protocol)
        payload=load_stage1_selection(stage1,protocol,settings,device=torch.device('cpu'))
        cached.append(dict(seed=seed,config_sha256=file_sha256(config),stage1_sha256=file_sha256(stage1),checkpoint_sha256=json.loads(stage1.read_text())['checkpoint_sha256']))
        del payload
    protocol=with_training_seed(base,502)
    batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    records=[]
    for label,config,stage1 in [('v31',FORMAL/'configs/e23_seed_502.yaml',FORMAL/'seed_502/E23-v31-B/stage1/runs/e23-stage1.json'),
                              ('tail32',TAIL/'e23_seed502.yaml',TAIL/'stage1/runs/e23-stage1.json')]:
        settings=load_e23_settings(config,protocol=protocol)
        payload=load_stage1_selection(stage1,protocol,settings,device=device)
        reset_training_rng(502);model=build_forecast_model(protocol,settings,device=device)
        assert parameter_hash(model.predictor)==HEAD and sum(p.numel() for p in model.predictor.parameters())==35427
        model.repair.load_state_dict(payload['model_state'],strict=True)
        h=model.repair.historical;h.retrieval.load_state_dict(payload['retrieval_state']);h.key_encoder.requires_grad_(False)
        memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);h.refresh_snapshot(memory.snapshot())
        model.forecast_padding='replicate_last_value_v1';starts=_starts(batch,stride=8,patches=12,device=device)
        captured=[];hook=model.predictor.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        model.eval()
        with torch.no_grad():
            prediction,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        hook.remove()
        expected=patchify_preserving_values(out.explicit.normalized_sequence,batch.observed_mask,16,8).values
        assert torch.equal(captured[0],expected)
        visible=out.explicit.normalized_sequence[batch.observed_mask]
        assert torch.equal(visible,batch.normalized_observed[batch.observed_mask])
        before=parameter_hash(model.repair);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0)
        reset_training_rng(502)
        for _ in range(2):
            model.train();optimizer.zero_grad(set_to_none=True)
            prediction,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            loss=(prediction-batch.forecast_target).square().mean();loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
        assert before!=parameter_hash(model.repair) and HEAD!=parameter_hash(model.predictor)
        records.append(dict(model=label,steps=2,head_sha256=HEAD,head_parameters=35427,finite_gradients=True,padding_matches_T1=True,visible_clamp_exact=True))
        del model,payload,optimizer;torch.cuda.empty_cache()
    profiles_path=FORMAL/'configs/author_seed_502.json';profiles=load_author_profiles(profiles_path);cfg=profiles['models']['T1']
    first_path=FORMAL/'seed_502/T1/training/repair-selection.json';first=json.loads(first_path.read_text())
    identity=author_source_identity(profiles_path,cfg,protocol)
    assert identity['author_profiles_sha256']==first['author_profiles_sha256'] and identity['upstream_source_sha256']==first['upstream_source_sha256']
    reset_training_rng(502);imputer=build_author_model('T1',cfg,WORK/cfg['root'],device)
    imputer.load_state_dict(torch.load(first['checkpoint'],map_location=device,weights_only=False)['model_state'],strict=True)
    reset_training_rng(502);model=T1ForecastModel(imputer,_forecast_head(protocol)).to(device)
    assert parameter_hash(model.predictor)==HEAD and sum(p.numel() for p in model.predictor.parameters())==35427
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0);reset_training_rng(502)
    for _ in range(2):
        model.train();optimizer.zero_grad(set_to_none=True);loss=(model(batch)-batch.forecast_target).square().mean();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    records.append(dict(model='T1',steps=2,head_sha256=HEAD,head_parameters=35427,finite_gradients=True,Stage1_reused=True,author_configuration_exact=True))
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    assert functions(WORK/'integrations/e23/train.py')['build_forecast_model']==functions(ORIGINAL/'integrations/e23/train.py')['build_forecast_model']
    assert (WORK/'src/malr/models/patchtst.py').read_bytes()==(ORIGINAL/'src/malr/models/patchtst.py').read_bytes()
    result=dict(status='passed',discarded_optimizer_steps=6,cached_seeds=cached,records=records,original_prediction_head_unchanged=True)
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
