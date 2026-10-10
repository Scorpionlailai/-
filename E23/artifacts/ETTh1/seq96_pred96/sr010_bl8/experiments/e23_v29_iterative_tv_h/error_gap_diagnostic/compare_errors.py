"""Read-only, paired E23-r1/T1 error attribution on the existing three test draws."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import load_stage2_selection,build_forecast_model,_new_memory,_key_sha256,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank import load_test_mask_bank
from integrations.forecast_compare.author_profiles import build_author_model
from integrations.forecast_compare.t1_p4 import T1ForecastModel,t1_inputs
from scripts.run_author_aligned_p4 import _forecast_head
from malr.migration import MigrationCatalog

torch.set_num_threads(4)
device=torch.device('cuda:1');out=Path(__file__).resolve().parent
base=Path('artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments');eroot=base/'e23_v29_iterative_tv_h_r1';troot=Path('artifacts/author_p4_manual_20261002/T1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load('../E22/migration/paper_catalog.jsonl',project='E23'),protocol_config='configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts='../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')
settings=load_e23_settings('configs/e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml',protocol=protocol)
eman=eroot/'stage2/runs/e23-stage2.json';epayload=load_stage2_selection(eman,protocol,settings,device=device);erecord=json.loads(eman.read_text())
e=build_forecast_model(protocol,settings,device=device);e.load_state_dict(epayload['model_state'],strict=True);e.repair.historical.retrieval.load_state_dict(epayload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(epayload['memory_state']);e.repair.historical.refresh_snapshot(memory.snapshot());e.eval()
key_before=_key_sha256(e.repair.historical.retrieval.state_dict());bank_before=memory.snapshot_hash()
trecord=json.loads((troot/'forecast-selection.json').read_text());assert sha(trecord['checkpoint'])==trecord['checkpoint_sha256'];assert trecord['source_csv_sha256']==protocol.source_sha256 and trecord['validation_mask_sha256']==protocol.mask_sha256['validation']
t=T1ForecastModel(build_author_model('T1',trecord['profile'],Path(trecord['profile']['root']),device),_forecast_head(protocol)).to(device)
tpayload=torch.load(trecord['checkpoint'],map_location=device,weights_only=False);t.load_state_dict(tpayload['model_state'],strict=True);t.eval()
bank=Path('artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json');views=load_test_mask_bank(protocol,bank)
old_e=json.loads((eroot/'diagnostic_three_test_draws.json').read_text());old_t=json.loads((troot/'three-test-draws.json').read_text())
variables=next(csv.reader(Path('../data/raw/ETTh1.csv').open()))[1:]
all_e=[];all_t=[];all_miss=[];all_std=[];all_route=[];group_records=[]
repair_sse=np.zeros((2,7));repair_count=np.zeros(7)
with torch.inference_mode():
 for group_id,view in enumerate(views):
  ee=[];tt=[];miss=[];std=[];routes=[];starts=[]
  for cpu in iter_p4_batches(view,'test',batch_size=128):
   batch=cpu.to(device);patch_starts=_starts(batch,stride=e.repair.stride,patches=e.repair.patch_count,device=device)
   ef,repair=e(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,patch_starts)
   tf=t(batch);error_e=ef-batch.forecast_target;error_t=tf-batch.forecast_target
   ee.append(error_e.cpu().numpy());tt.append(error_t.cpu().numpy());miss.append((~batch.observed_mask).float().mean(1).cpu().numpy());std.append(batch.query_std.cpu().numpy())
   starts.extend(ref.target_start for ref in cpu.refs)
   damaged=(~repair.explicit.observation_patches & repair.explicit.validity_patches).any(-1)
   route=(repair.router.weights*damaged[...,None]).sum(2)/damaged.sum(2).clamp_min(1)[...,None]
   routes.append(route.cpu().numpy())
   ev=(repair.explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:,None,:]
   values,mask=t1_inputs(batch);tv=t.imputer(values,mask=mask)-(batch.normalized_target*batch.query_std[:,None,:]+batch.query_mean[:,None,:]);hidden=~batch.observed_mask
   repair_sse[0]+=ev.square().masked_fill(~hidden,0).sum((0,1)).cpu().numpy()
   repair_sse[1]+=tv.square().masked_fill(~hidden,0).sum((0,1)).cpu().numpy();repair_count+=hidden.sum((0,1)).cpu().numpy()
  a,b=np.concatenate(ee),np.concatenate(tt);m,s,r=np.concatenate(miss),np.concatenate(std),np.concatenate(routes)
  assert a.shape==b.shape==(2688,7,96)
  stats={}
  for name,errors,old in [('E23_v29_r1',a,old_e),('T1',b,old_t)]:
   mse=float(np.square(errors.astype(np.float64)).mean());mae=float(np.abs(errors.astype(np.float64)).mean());expected=old['test_groups'][group_id]
   assert abs(mse-expected['mse'])<1e-6 and abs(mae-expected['mae'])<1e-6,(name,group_id,mse,mae,expected)
   stats[name]=dict(mse=mse,mae=mae)
  np.savez_compressed(out/f'group_{group_id}_errors.npz',e23_error=a,t1_error=b,missing_fraction=m,query_std=s,route_weights=r,target_start=np.array(starts))
  all_e.append(a);all_t.append(b);all_miss.append(m);all_std.append(s);all_route.append(r);group_records.append(dict(group_id=group_id,seed=old_e['test_groups'][group_id]['seed'],mask_sha256=view.mask_sha256['test'],metrics=stats))
  print('reproduced group',group_id,stats,flush=True)
a=np.concatenate(all_e).astype(np.float64);b=np.concatenate(all_t).astype(np.float64);m=np.concatenate(all_miss);s=np.concatenate(all_std);r=np.concatenate(all_route)

def measure(x,y):
 return dict(coordinates=int(x.size),e23_mse=float((x*x).mean()),t1_mse=float((y*y).mean()),mse_gap=float((x*x-y*y).mean()),e23_mae=float(np.abs(x).mean()),t1_mae=float(np.abs(y).mean()),mae_gap=float((np.abs(x)-np.abs(y)).mean()),e23_bias=float(x.mean()),t1_bias=float(y.mean()),e23_abs_error_win_fraction=float((np.abs(x)<np.abs(y)).mean()))
summary=measure(a,b);per_var=[];per_h=[];per_vh=[];total=a.size
for c,name in enumerate(variables):
 row=dict(variable=name,index=c,**measure(a[:,c],b[:,c]));row['contribution_to_total_mse_gap']=row['mse_gap']/7;row['e23_hidden_repair_mse']=float(repair_sse[0,c]/repair_count[c]);row['t1_hidden_repair_mse']=float(repair_sse[1,c]/repair_count[c]);row['route_none_H_V_T']=r[:,c].mean(0).tolist();per_var.append(row)
 for h in range(4):per_vh.append(dict(variable=name,horizon=f'{h*24+1}-{(h+1)*24}',**measure(a[:,c,h*24:(h+1)*24],b[:,c,h*24:(h+1)*24])))
for h in range(4):per_h.append(dict(horizon=f'{h*24+1}-{(h+1)*24}',**measure(a[:,:,h*24:(h+1)*24],b[:,:,h*24:(h+1)*24])))
quantiles=[.5,.75,.9,.95,.99,.999];q={name:{str(p):float(np.quantile(np.abs(errors),p)) for p in quantiles} for name,errors in [('E23',a),('T1',b)]}
flat_a=a.reshape(-1);flat_b=b.reshape(-1);difficulty=np.maximum(np.abs(flat_a),np.abs(flat_b));order=np.argsort(difficulty);paired_tail=[]
for fraction in (.01,.05,.10):
 n=int(np.ceil(total*fraction));ix=order[-n:];low=order[:-n];rr=measure(flat_a[ix],flat_b[ix]);rr.update(top_fraction=fraction,definition='same coordinates ranked by max(abs(E23_error),abs(T1_error))',contribution_to_total_mse_gap=float((flat_a[ix]**2-flat_b[ix]**2).sum()/total),remaining_mse_gap=float((flat_a[low]**2-flat_b[low]**2).mean()));paired_tail.append(rr)
missing_bins=[]
for lo,hi in [(0,.25),(.25,.50),(.50,.75),(.75,1.000001)]:
 mask=(m>=lo)&(m<hi);missing_bins.append(dict(missing_fraction_range=[lo,min(hi,1.)],**measure(a[mask],b[mask])))
std_bins=[]
for lo,hi in [(0,.01),(.01,.1),(.1,.5),(.5,1.),(1.,float('inf'))]:
 mask=(s>=lo)&(s<hi)
 if mask.any():std_bins.append(dict(query_std_range=[lo,None if np.isinf(hi) else hi],**measure(a[mask],b[mask])))
# Equal quarters of target-window starts; same windows in each draw.
time_regions=[]
for i in range(4):
 ix=np.concatenate([np.arange(g*2688+i*672,g*2688+(i+1)*672) for g in range(3)]);time_regions.append(dict(test_window_quarter=i+1,**measure(a[ix],b[ix])))
positive=flat_a**2-flat_b**2;pos_sum=float(positive[positive>0].sum()/total);neg_sum=float(positive[positive<0].sum()/total)
key_after=_key_sha256(e.repair.historical.retrieval.state_dict());bank_after=memory.snapshot_hash();assert key_before==key_after and bank_before==bank_after
result=dict(purpose='read-only test error attribution; no retraining, checkpoint selection or calibration',e23_checkpoint_sha256=erecord['checkpoint_sha256'],t1_checkpoint_sha256=trecord['checkpoint_sha256'],test_mask_bank_sha256=sha(bank),variables=variables,group_records=group_records,overall=summary,per_variable=per_var,per_horizon=per_h,per_variable_horizon=per_vh,absolute_error_quantiles=q,paired_large_error_subsets=paired_tail,missing_fraction_bins=missing_bins,query_std_bins=std_bins,test_window_quarters=time_regions,positive_squared_error_gap_contribution=pos_sum,negative_squared_error_gap_contribution=neg_sum,h_state_unchanged=True,diagnostic_code_sha256=sha(__file__))
with (out/'error_gap.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
for name,rows in [('per_variable',per_var),('per_horizon',per_h),('per_variable_horizon',per_vh),('paired_large_error_subsets',paired_tail),('missing_fraction_bins',missing_bins),('query_std_bins',std_bins),('test_window_quarters',time_regions)]:
 with (out/(name+'.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(overall=summary,per_variable=per_var,per_horizon=per_h,quantiles=q,paired_tails=paired_tail,missing_bins=missing_bins,std_bins=std_bins,time_quarters=time_regions),indent=2))
