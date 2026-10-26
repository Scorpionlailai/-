"""Evaluate selected checkpoints, 20 masks, pre/post calibration, exact export."""
import argparse,json,os
from pathlib import Path
import torch
from bootstrap import ROOT,verify_sources
from data import load_data
from reuse import load_model_checkpoint
from support import write_json,artifact_root,sha,digest
from train import export_candidate,load_export
from variants import library_hash
from integrations.pamap2_baselines.run import evaluate,aggregate

def aggregate_checked(groups):
 # One-mask smoke has no estimable sample SD; do not serialize NaN as a result.
 if len(groups)==1:
  return {name:{'mse_mean':groups[0][name]['mse'],'mae_mean':groups[0][name]['mae'],'mse_mask_std_ddof1':None,'mae_mask_std_ddof1':None,'coordinate_weighted':groups[0][name]} for name in ['full96','recent32']}
 return aggregate(groups)

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='ETTh1');p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,default=ROOT/'outputs');p.add_argument('--variant',choices=['R2','FULL','CTX','SUM'],default='R2');p.add_argument('--scene',default='sr015_bl8');p.add_argument('--seed',type=int,default=2023);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-batches',type=int,default=None)
 args=p.parse_args();os.environ.update(A64_DATASET=args.dataset,R2_DATA_ROOT=str(args.data_root.resolve()),R2_OUTPUT_ROOT=str(args.output_root.resolve()));torch.set_num_threads(4);verify_sources()
 v='FULL' if args.variant=='R2' else args.variant;device=torch.device(args.device)
 if device.type=='cuda':torch.cuda.set_device(device)
 d=load_data(args.dataset,args.scene,artifact_root()/'assets');m,info=load_model_checkpoint(d,args.scene,args.seed,v,device,smoke=args.smoke_batches)
 before=digest(m.frozen.repair);bank=library_hash(m.frozen.repair)
 val=evaluate(m,d,'validation',device,batch_size=128,max_batches=1 if args.smoke_batches else None)
 for metric in ['mse','mae']:
  if abs(val['full96'][metric]-info['validation']['full96'][metric])>1e-6:raise RuntimeError('selected checkpoint validation replay differs')
 groups=[];pre=[]
 seeds=d.test_seeds[:1] if args.smoke_batches else d.test_seeds
 for seed in seeds:
  kw=dict(batch_size=128,mask_seed=seed,max_batches=1 if args.smoke_batches else None)
  groups.append(evaluate(m,d,'test',device,**kw));pre.append(evaluate(m.frozen.repair,d,'test',device,**kw))
 out=artifact_root()/('smoke_evaluation' if args.smoke_batches else 'evaluation')/args.dataset/args.scene/f'seed_{args.seed}'/v
 export=out/'model.pt';export_candidate(export,m,{'variant':v,'seed':args.seed,'coarse_epoch':info['coarse']['epoch'],'head_epoch':info['epoch'],'dataset':args.dataset})
 restored=load_export(export,d,device);truth,seen,starts=next(d.batches('validation',16));truth=truth.to(device);seen=seen.to(device);values=torch.where(seen,truth,0)
 a=m(values,seen,starts);b=restored(values,seen,starts)
 if not torch.equal(a,b) or not torch.equal(a[seen],values[seen]):raise RuntimeError('export/observed preservation differs')
 if digest(m.frozen.repair)!=before or library_hash(restored.frozen.repair)!=bank:raise RuntimeError('frozen model/memory changed')
 total=sum(p.numel() for p in m.parameters());excluded=sum(p.numel() for n,p in m.named_parameters() if n.startswith(('frozen.repair.ema_encoder.','frozen.repair.core.historical.key_encoder.')))
 result={'dataset':args.dataset,'scene':args.scene,'seed':args.seed,'variant':v,'validation':val,'test':aggregate_checked(groups),'precal_test':aggregate_checked(pre),'info':info,'test_masks':len(seeds),'effective_parameters':total-excluded,'registered_parameters':total,'export_verified':True,'export_sha256':sha(export),'smoke_only':bool(args.smoke_batches)}
 write_json(out/'results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
