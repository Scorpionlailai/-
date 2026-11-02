from pathlib import Path
import hashlib
import json
import os
import torch
import bootstrap
from aligned_h_model import make_repair,ORIGINAL_FACTORY
from data import load_data
from variants import make_calibrator
from single import FrozenRepair,SingleBranchModel
from support import write_json,digest
from malr.data.patching import patchify_with_state


def run(args):
 os.environ['R2_DATA_ROOT']=str(args.data_root.resolve());torch.set_num_threads(4)
 out=args.output.resolve()
 bootstrap.verify_sources()
 data=load_data('ETTh1','sr015_bl8',out/'assets')
 torch.manual_seed(2023);model=make_repair(data,'H_GLOBAL',2023)
 torch.manual_seed(2023);old=ORIGINAL_FACTORY(data,'FULL',2023)
 assert digest(model)==digest(old)
 assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in old.parameters())
 truth,seen,starts=next(data.batches('train',4));valid=torch.ones_like(seen)
 expected_q=patchify_with_state(torch.where(seen,truth,0),seen,16,8)
 expected_t=patchify_with_state(truth,valid,16,8)
 eligible=expected_q.rho.lt(1)&expected_t.effective_mask.any(-1)
 captures=[]
 hook=model.core.historical.key_encoder.register_forward_pre_hook(lambda m,args:captures.append((args[0].detach().clone(),args[1].detach().clone())))
 model.epoch=1;model.batch_index=1
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
 loss=model.loss(truth,seen,starts)
 assert len(captures)==2
 assert torch.equal(captures[0][0],expected_q.values[eligible]) and torch.equal(captures[0][1],expected_q.effective_mask[eligible])
 assert torch.equal(captures[1][0],expected_t.values[eligible])
 q=model.core.historical.key_encoder(captures[0][0],captures[0][1]);t=model.core.historical.key_encoder(captures[1][0],captures[1][1])
 from malr.prototypes.retrieval import masked_pair_loss
 retrieval=masked_pair_loss(q,t)
 key_grad=torch.autograd.grad(retrieval,model.core.historical.key_encoder.network[0].weight,retain_graph=True)[0]
 assert torch.isfinite(key_grad).all() and key_grad.norm()>0
 loss.backward();opt.step();opt.zero_grad(set_to_none=True);hook.remove()
 model.after_step();model.publish_epoch(1)
 # Actual query path must encode global visible values, but returned candidates
 # must still be normalized to the original target window.
 query_calls=[]
 hook=model.core.historical.retrieval.frozen_encoder.register_forward_pre_hook(lambda m,args:query_calls.append((args[0].detach().clone(),args[1].detach().clone())))
 result,norm,_=model.output(truth,seen,starts)
 hook.remove();assert query_calls
 p=patchify_with_state(norm.student,seen,16,8)
 expected_global=torch.where(p.effective_mask,torch.where(p.effective_mask,p.values,0)*norm.std[:,:,None,None]+norm.mean[:,:,None,None],0)
 assert torch.equal(query_calls[0][0],expected_global.reshape(-1,16))
 assert torch.equal(query_calls[0][1],p.effective_mask.reshape(-1,16))
 # Verify the actual published raw members by serialization-independent traversal.
 members=[member for groups in model.memory.snapshot().groups for group in groups for member in group.members]
 import numpy as np
 raw=torch.as_tensor(np.stack([x.raw_patch for x in members]),dtype=torch.float32)
 with torch.no_grad():keys=model.core.historical.retrieval.frozen_encoder(raw,torch.ones_like(raw,dtype=torch.bool))
 stored=torch.as_tensor(np.stack([x.clean_key for x in members]),dtype=torch.float32)
 assert torch.allclose(keys,stored,rtol=2e-6,atol=2e-6)
 # Construct a pair for which an exact fully observed member is encoded exactly
 # like the restored-global query. No promise for masked ranking is inferred.
 torch.manual_seed(71)
 local=torch.randn(1,16);mean=torch.tensor(2.);std=torch.tensor(3.)
 global_values=local*std+mean;mask=torch.ones_like(local,dtype=torch.bool)
 enc=model.core.historical.retrieval.frozen_encoder
 with torch.no_grad():
  true_key=enc(global_values,mask);other=enc(local,mask)
  restored=enc(local*std+mean,mask)
  aligned_true=float((restored*true_key).sum());aligned_other=float((restored*other).sum())
 assert torch.equal(restored,true_key) and aligned_true>=aligned_other-1e-6
 model.eval()
 main=(result.explicit.standardized_sequence[~seen]-truth[~seen]).square().mean()
 hparam=model.core.historical.message_projector[0].weight
 hgrad=torch.autograd.grad(main,hparam,allow_unused=True)[0]
 # Null availability at this tiny sample is reported, not disguised as evidence
 # of trained source usefulness. Formal epoch2 gets an additional audit below.
 hnorm=None if hgrad is None else float(hgrad.norm())
 assert hgrad is None or torch.isfinite(hgrad).all()
 model.zero_grad(set_to_none=True)
 full=SingleBranchModel(FrozenRepair(model),make_calibrator(7,'FULL',2023)).eval()
 before=digest(model)
 for case in ['normal','all_missing','long_gap']:
  mask=seen.clone()
  if case=='all_missing':mask[:]=False
  if case=='long_gap':mask[:,8:88]=False
  with torch.no_grad():
   a=full(torch.where(mask,truth,0),mask,starts)
   b=full(torch.where(mask,truth,torch.full_like(truth,float('nan'))),mask,starts)
  assert torch.isfinite(a).all() and torch.equal(a,b) and torch.equal(a[mask],truth[mask])
 full.head.train();o=torch.optim.Adam(full.head.parameters(),lr=.001)
 y=full(torch.where(seen,truth,0),seen,starts);l=(y[~seen]-truth[~seen]).abs().mean();l.backward();o.step()
 assert digest(model)==before and all(p.grad is None for p in model.parameters())
 # Ensure derived source has exactly the registered query-only code delta.
 original=(bootstrap.ROOT/'vendor/E24/integrations/v125_ablation/model/historical_candidates.py').read_text()
 target=original.replace('from .contracts import','from integrations.v125_ablation.model.contracts import').replace('from .nucleus import','from integrations.v125_ablation.model.nucleus import')
 target=target.replace('if getattr(self,"key_policy","legacy_v1") == "global_diverse_v1":','if getattr(self,"key_policy","legacy_v1") == "global_diverse_v1" or getattr(self,"query_coordinates","window_local") == "train_global":')
 assert (bootstrap.ROOT/'aligned_h_candidates.py').read_text()=='# Derived from the pinned provider; only query encoding coordinates differ.\n'+target
 write_json(out/'preflight.json',dict(passed=True,initial_weights_identical=True,parameters_identical=True,
    train_query_clean_global_exact=True,inference_query_global_exact=True,published_keys_recomputed_from_raw=True,
    key_auxiliary_gradient_norm=float(key_grad.norm()),H_main_gradient_norm_tiny_probe=hnorm,
    synthetic_exact_similarity=aligned_true,synthetic_other_similarity=aligned_other,
    vendor_unchanged=True,hidden_nan_invariant=True,observations_exact=True,prefix_frozen=True,
    H_policy=model.core.historical.key_policy,topk=model.core.historical.topk,threshold=model.core.historical.similarity_threshold,
    real_performance_claim=False))
 print('H coordinate checks passed; tiny-probe H main gradient norm:',hnorm)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 run(p.parse_args())
