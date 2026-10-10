"""CPU-only fixed-validation diagnostic by variable and gap length."""
from pathlib import Path
import json,torch,hashlib
from evaluate_repair import ROOT,protocol_for,sha
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,build_repair_model,_new_memory,_starts
from integrations.forecast_compare.author_profiles import build_author_model
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history
from integrations.e23.forecast_sensitive import parameter_hash

def gap_lengths(hidden):
 left=torch.zeros_like(hidden,dtype=torch.int64);right=left.clone()
 for i in range(hidden.shape[1]):left[:,i]=(left[:,i-1]+1)*hidden[:,i] if i else hidden[:,i].long()
 for i in range(hidden.shape[1]-1,-1,-1):right[:,i]=(right[:,i+1]+1)*hidden[:,i] if i+1<hidden.shape[1] else hidden[:,i].long()
 return (left+right-1)*hidden

@torch.inference_mode()
def main():
 torch.set_num_threads(4);p=json.loads((ROOT/'plan.json').read_text());scene=next(v for v in p['scenarios'] if v['name']=='sr010_bl16');protocol=protocol_for(p,scene);device=torch.device('cpu')
 settings=load_e23_settings(Path(p['models'][0]['config']),protocol=protocol);selection=ROOT/'E23-v92'/scene['name']/'stage1/runs/e23-stage1.json'
 payload=load_stage1_selection(selection,protocol,settings,device=device);e23=build_repair_model(protocol,settings,device=device);e23.load_state_dict(payload['model_state'],strict=True);e23.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);e23.historical.refresh_snapshot(memory.snapshot());e23.eval().requires_grad_(False)
 t=json.loads(Path(scene['t1_repair_selection']).read_text());cp=Path(t['checkpoint']);assert sha(cp)==t['checkpoint_sha256'];official=Path('/home/xyw/paper/E23')/t['profile']['root'];t1=build_author_model('T1',t['profile'],official,device);t1.load_state_dict(torch.load(cp,map_location=device,weights_only=False)['model_state'],strict=True);t1.eval().requires_grad_(False)
 before=(parameter_hash(e23),parameter_hash(t1),memory.snapshot_hash());totals={};windows=0
 for index,b in enumerate(iter_p4_batches(protocol,'validation',batch_size=128),1):
  starts=_starts(b,stride=e23.stride,patches=e23.patch_count,device=device);out=e23(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts).explicit
  values,mask=t1_inputs(b);t1out=t1(values,mask=mask).float();errors={'E23-v92':(out.normalized_sequence-b.normalized_target)*b.query_std[:,None],'T1':t1out-t1_clean_history(b)}
  hidden=~b.observed_mask & out.sequence_validity
  visible=torch.where(b.observed_mask,b.normalized_observed*b.query_std[:,None]+b.query_mean[:,None],0)
  pred=out.normalized_sequence*b.query_std[:,None]+b.query_mean[:,None]
  index=torch.arange(96)[None,:,None];left=torch.cummax(torch.where(b.observed_mask,index,-1),1).values;right=torch.cummin(torch.where(b.observed_mask,index,96).flip(1),1).values.flip(1)
  lv=visible.gather(1,left.clamp_min(0));rv=visible.gather(1,right.clamp_max(95));span=(right-left).clamp_min(1);both=(left>=0)&(right<96)&(right>left)
  linear=lv+(rv-lv)*(index-left)/span;bs=both.float()*torch.exp(-(span-1).float()/16)
  one=(left>=0)^(right<96);distance=torch.where(left>=0,index-left,right-index).clamp_min(0);osupport=one.float()*torch.exp(-distance.float()/16)
  guide=torch.where(both,linear,torch.where(left>=0,lv,rv));support=bs+osupport
  errors['v92_plus_both_guide_halfstep']=errors['E23-v92']+.5*bs*(linear-pred)
  errors['v92_plus_one_or_both_guide_halfstep']=errors['E23-v92']+.5*support*(guide-pred)
  length=gap_lengths(hidden);regions={'all_hidden':hidden,'last32_hidden':hidden & (torch.arange(96)[None,:,None]>=64)}
  groups={f'variable_{v}':torch.arange(7)[None,None,:]==v for v in range(7)};groups.update({'gap_1_8':length.le(8),'gap_9_16':length.gt(8)&length.le(16),'gap_17_32':length.gt(16)&length.le(32),'gap_33_plus':length.gt(32),'all':torch.ones_like(hidden)})
  for model,error in errors.items():
   for region,scope in regions.items():
    for group,gm in groups.items():
     r=error[scope & gm].double();key=(model,region,group);t=totals.setdefault(key,[0.,0.,0]);t[0]+=r.square().sum().item();t[1]+=r.abs().sum().item();t[2]+=r.numel()
  windows+=b.observed_mask.shape[0]
  print(f'fixed_validation_diagnostic batch={index}/21',flush=True)
 assert before==(parameter_hash(e23),parameter_hash(t1),memory.snapshot_hash())
 rows=[{'model':model,'region':region,'group':group,'hidden_coordinates':n,'mse':s/n if n else None,'mae':a/n if n else None} for (model,region,group),(s,a,n) in totals.items()]
 result={'scene':scene['name'],'split':'fixed_validation','device':'cpu','windows':windows,'metric_space':'train_standardized','models_unchanged':True,'rows':rows}
 (ROOT/'v92-one-sided-guide-validation-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'rows':len(rows),'completed':True}),flush=True)
if __name__=='__main__':main()
