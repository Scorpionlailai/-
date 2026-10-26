"""Small correctness checks, no formal performance claim or checkpoint choice."""
import argparse,os,json
from pathlib import Path
import torch
from bootstrap import ROOT,verify_sources
from data import load_data
from variants import make_repair,make_calibrator,library_hash
from single import FrozenRepair,SingleBranchModel
from support import write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--dataset',default='ETTh1');p.add_argument('--scene',default='sr015_bl8');p.add_argument('--assets',type=Path,default=ROOT/'outputs/selfcheck_assets');a=p.parse_args()
 os.environ['R2_DATA_ROOT']=str(a.data_root.resolve());torch.set_num_threads(4);verify_sources();d=load_data(a.dataset,a.scene,a.assets);rows={}
 truth,seen,starts=next(d.batches('train',2))
 for v in ['FULL','SUM','CTX']:
  torch.manual_seed(2023);m=make_repair(d,v,2023);loss=m.loss(truth,seen,starts);assert torch.isfinite(loss);loss.backward()
  if v=='CTX':assert not hasattr(m,'memory') and all(p.grad is not None for p in m.parameters() if p.requires_grad)
  m.after_step();m.publish_epoch(1);head=make_calibrator(d.n_vars,v,2023);full=SingleBranchModel(FrozenRepair(m),head).eval()
  for case in ['normal','all_missing','long_gap','one_variable']:
   mask=seen.clone()
   if case=='all_missing':mask[:]=False
   if case=='long_gap':mask[:,8:88]=False
   if case=='one_variable':mask[:]=False;mask[:,:,0]=True
   values=torch.where(mask,truth,0)
   with torch.no_grad():out=full(values,mask,starts)
   assert torch.isfinite(out).all() and torch.equal(out[mask],values[mask])
  total=sum(p.numel() for p in full.parameters());excluded=sum(p.numel() for n,p in full.named_parameters() if n.startswith(('frozen.repair.ema_encoder.','frozen.repair.core.historical.key_encoder.')))
  if a.dataset=='ETTh1':assert total-excluded==(353256 if v=='CTX' else 353248)
  rows[v]={'finite_loss':True,'observation_exact':True,'all_missing_finite':True,'effective_parameters':total-excluded}
 result={'dataset':a.dataset,'scene':a.scene,'source_sha':d.manifest.get('source_sha256'),'checks':rows,'formal_performance':False};write_json(a.assets/'selfcheck.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
