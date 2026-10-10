from dataclasses import replace
import importlib.util,json
from pathlib import Path
import pytest,torch,yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
from malr.data.patching import patchify_with_state
from malr.models.e23.window_context import WindowContextEncoder
from malr.models.e23.window_signature import WindowSignatureContext
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_old_conditioned_signature_constructor_and_state_unchanged():
 path=Path('/home/xyw/paper/.e23-v62-repair-updates/E23/src/malr/models/e23/window_signature.py')
 spec=importlib.util.spec_from_file_location('previous_window_signature',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 torch.manual_seed(502);old=module.WindowSignatureContext(64,7);rng=torch.get_rng_state()
 torch.manual_seed(502);new=WindowSignatureContext(64,7)
 assert torch.equal(rng,torch.get_rng_state())
 assert old.state_dict().keys()==new.state_dict().keys()
 for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k


def test_plain_signature_zero_start_preserves_body_rng_and_head(tmp_path):
 p,cfg=setup(tmp_path);torch.manual_seed(502);old=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();rng=torch.get_rng_state()
 torch.manual_seed(502);new=tr.build_forecast_model(p,replace(cfg,front_variable_context='signature_plain_v2'),device=torch.device('cpu')).eval()
 assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(new.predictor)
 for k,v in old.repair.state_dict().items():assert torch.equal(v,new.repair.state_dict()[k]),k
 # Synthetic integration protocol has three variables; signature params don't depend on C without statistics MLP.
 sig=new.repair.front_context.variable_context
 assert sum(v.numel() for v in sig.parameters())==105088
 assert all(block.statistics is None for block in sig.blocks)
 b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
 with torch.no_grad():
  a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
  z,oz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
 assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_plain_signature_statistics_unused_and_visible_source_only():
 torch.manual_seed(9);x=torch.randn(2,96,3);mask=torch.rand_like(x)>.73;valid=torch.ones_like(mask);valid[:,-1]=False
 pv=patchify_with_state(x,mask,16,8,validity=valid).validity;mean=torch.zeros(2,3);std=torch.ones(2,3)
 front=WindowContextEncoder(64,96,16,8,variable_context='signature_plain_v2',variable_count=3,coordinates='robust_local_v1').eval()
 with torch.no_grad():
  front.output.weight.normal_(std=.05)
  for block in front.variable_context.blocks:block.project.weight.normal_(std=.03)
 a=front(x,mask,mean,std,pv,valid_mask=valid)
 polluted=torch.where(mask&valid,x,torch.full_like(x,float('inf')))
 z=front(polluted,mask,mean,std,pv,valid_mask=valid);assert torch.equal(a,z)
 changed=x.clone();changed[:,:,1]+=5;q=front(changed,mask,mean,std,pv,valid_mask=valid);assert not torch.equal(a[:,0],q[:,0])
 mask[:,:,1:]=False;pv=patchify_with_state(x,mask,16,8,validity=valid).validity
 assert torch.equal(front(x,mask,mean,std,pv,valid_mask=valid),front(changed,mask,mean,std,pv,valid_mask=valid))
 points=torch.randn(6,64,96);flatvalid=valid.transpose(1,2).reshape(6,1,96)
 sig=front.variable_context
 assert torch.equal(sig(points,mask,flatvalid,mean,std),sig(points,mask,flatvalid,torch.full_like(mean,float('nan')),torch.full_like(std,float('nan'))))
 assert torch.isfinite(front(polluted,torch.zeros_like(mask),mean,std,pv,valid_mask=valid)).all()


def test_plain_signature_learns_and_stage2_freezes_it(tmp_path):
 p,cfg=setup(tmp_path);cfg=replace(cfg,front_variable_context='signature_plain_v2');m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=_batch('train');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
 for step in range(4):
  opt.zero_grad();o=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);loss,_=tr.stage1_objective(o.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std);loss.backward()
  block=m.repair.front_context.variable_context.blocks[0]
  if step>=1:assert block.project.weight.grad.abs().sum()>0
  if step>=2:assert block.query.large.weight.grad.abs().sum()>0
  assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None);opt.step()
 opt=build_stage2_optimizer(m,mode='freeze_front_v1',learning_rate=1e-4,weight_decay=0);stage2_training_mode(m,'freeze_front_v1');ids={id(v) for group in opt.param_groups for v in group['params']}
 assert not m.repair.front_context.variable_context.training
 assert not any(v.requires_grad or id(v) in ids for v in m.repair.front_context.variable_context.parameters())


def test_plain_signature_dual_checkpoint_identity_roundtrip(tmp_path,monkeypatch):
 p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text());doc['model'].update(variable_calibration='none',front_variable_context='signature_plain_v2');path.write_text(yaml.safe_dump(doc));cfg=load_e23_settings(path,protocol=p)
 monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
 s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1,training_batch_size=32)
 s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1',extra_mse_checkpoint=True)
 for selected in [s2,s2.parent/'e23-stage2-mse.json']:
  payload=tr.load_stage2_selection(selected,p,cfg,device=torch.device('cpu'));assert payload['front_signature_statistics']=='none' and payload['front_signature_stat_width']==0 and payload['front_signature_extra_parameters']==105088
  assert 'front_signature_scale_bounds' not in payload and 'front_signature_offset_bound' not in payload
 rec=json.loads(s2.read_text());rec['front_signature_statistics']='visible_only_mean_logstd';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
 with pytest.raises(ValueError,match='front_signature_statistics.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))
