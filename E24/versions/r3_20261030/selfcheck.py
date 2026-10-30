"""Portable R3 preflight; observed/hidden invariants and frozen calibration."""
import argparse,os,json
from pathlib import Path
import torch
import bootstrap
from aligned_h_model import install
install()
from variants import make_repair,make_calibrator,library_hash
from data import load_data
from single import FrozenRepair,SingleBranchModel
from support import digest,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',default='ETTh1');p.add_argument('--scene',default='sr015_bl8')
    p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
    os.environ['R2_DATA_ROOT']=str(a.data_root.resolve());os.environ['A64_DATASET']=a.dataset
    torch.set_num_threads(4);bootstrap.verify_sources()
    d=load_data(a.dataset,a.scene,a.assets);m=make_repair(d,'H_GLOBAL',2023)
    truth,seen,starts=next(d.batches('train',2));loss=m.loss(truth,seen,starts)
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
    m.zero_grad(set_to_none=True);m.after_step();m.publish_epoch(1)
    full=SingleBranchModel(FrozenRepair(m),make_calibrator(d.n_vars,'H_GLOBAL',2023)).eval()
    before=digest(m);bank=library_hash(m)
    with torch.no_grad():
        for case in ['normal','all_missing','long_gap']:
            mask=seen.clone()
            if case=='all_missing':mask[:]=False
            if case=='long_gap':mask[:,8:88]=False
            y=full(torch.where(mask,truth,0),mask,starts)
            y_nan=full(torch.where(mask,truth,torch.full_like(truth,float('nan'))),mask,starts)
            assert torch.isfinite(y).all() and torch.equal(y,y_nan) and torch.equal(y[mask],truth[mask])
    full.head.train();full(torch.where(seen,truth,0),seen,starts).square().mean().backward()
    assert digest(m)==before and library_hash(m)==bank and all(x.grad is None and not x.requires_grad for x in m.parameters())
    r=dict(dataset=a.dataset,scene=a.scene,finite_forward_backward=True,hidden_nan_invariant=True,observations_exact=True,frozen_coarse=True,H_query_coordinates=m.core.historical.query_coordinates,H_policy=m.core.historical.key_policy,registered_parameters=sum(x.numel() for x in full.parameters()),formal_performance=False)
    write_json(a.assets/'selfcheck.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
