from dataclasses import replace
import ast,json
from pathlib import Path
import pytest,torch
from integrations.e23 import train as tr
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_recent_mean import batch
from integrations.e23.forecast_sensitive import parameter_hash


def test_padding_alignment_matches_t1_input_keeps_head_parameters_masks_and_real_values(tmp_path):
 p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates='robust_local_v1');m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));seen={}
 def capture(_m,args,kwargs):seen['values']=args[0].clone();seen['mask']=args[1].clone();seen['validity']=kwargs['patch_validity'].clone()
 h=m.predictor.register_forward_pre_hook(capture,with_kwargs=True);original_hash=parameter_hash(m.predictor)
 with torch.no_grad():_,o=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
 old=seen.copy();m.forecast_padding='replicate_last_value_v1'
 with torch.no_grad():_,z=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
 h.remove();assert parameter_hash(m.predictor)==original_hash;valid=seen['validity'];assert torch.equal(old['mask'],seen['mask']) and torch.equal(old['validity'],valid);assert torch.equal(old['values'][valid],seen['values'][valid]);assert old['values'][~valid].eq(0).all()
 last=z.explicit.normalized_sequence[:,-1,:][:,:,None,None].expand_as(seen['values']);assert torch.equal(seen['values'][~valid],last[~valid]);assert torch.equal(o.explicit.normalized_sequence,z.explicit.normalized_sequence)


def test_padding_checkpoint_identity_replay_and_tamper_rejection(tmp_path,monkeypatch):
 p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates='robust_local_v1');monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
 s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
 s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1',forecast_padding='replicate_last_value_v1')
 payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'));assert payload['forecast_padding_alignment_only'] is True and payload['forecast_padding']=='replicate_last_value_v1'
 record=json.loads(s2.read_text());m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));m.forecast_padding=payload['forecast_padding'];m.load_state_dict(payload['model_state']);m.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=tr._new_memory(p);memory.load_state_dict(payload['memory_state']);m.repair.historical.refresh_snapshot(memory.snapshot());actual=tr._validation_forecast_metrics(m,p,cfg,device=torch.device('cpu'));assert abs(actual['mse']-record['best_val_mse'])<1e-6
 for k,v in [('forecast_padding','wrong'),('forecast_padding_alignment_only',False)]:
  d=record.copy();d[k]=v;wrong=tmp_path/f'{k}.json';wrong.write_text(json.dumps(d))
  with pytest.raises(ValueError,match='padding'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


def test_loss_epoch_loop_and_head_sources_unchanged():
 old=Path('/home/xyw/paper/.e23-v59-temporal-evidence/E23');new=Path(__file__).resolve().parents[2]
 def function(root,name):return next(n for n in ast.parse((root/'integrations/e23/train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(function(old,'stage2_objective'))==ast.dump(function(new,'stage2_objective'))
 a=next(n for n in function(old,'train_forecast').body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch');b=next(n for n in function(new,'train_forecast').body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch');assert ast.dump(a)==ast.dump(b)
 for rel in ['src/malr/models/repaired_forecast.py','src/malr/models/patchtst.py','src/malr/models/e23/model.py']:assert (old/rel).read_bytes()==(new/rel).read_bytes()


@pytest.mark.parametrize('padding',['wrong',True,1])
def test_invalid_padding_rejected_early(tmp_path,padding):
 p,cfg=setup(tmp_path)
 with pytest.raises(ValueError,match='padding'):tr.train_forecast(p,cfg,stage1_manifest=tmp_path/'absent',output=tmp_path/'output',device=torch.device('cpu'),forecast_padding=padding)
