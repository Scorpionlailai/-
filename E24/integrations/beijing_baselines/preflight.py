"""Real GPU gradients, edge inputs, checkpoint round trip and cached head step."""
import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch

from .adapters import build_adapter
from .data import SharedData, SCENES, write_json
from .run import seed_all, state_sha, repair, build_head, forecast
from .cache import RepairCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda:0');records=[]
    for scene in SCENES:
        seed_all(2023)
        data=SharedData('BeijingAirQuality',scene,args.root/'shared_assets')
        model=build_adapter(args.model,data,device)
        truth,observed,starts=next(data.batches('train',2))
        truth,observed=truth.to(device),observed.to(device)
        if args.model=='v125':
            model.inference_path='coarse';model.epoch=1;model.core.historical.retrieval.begin_epoch(1)
        amp=args.model=='ImputeFormer'
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        with torch.cuda.amp.autocast(enabled=amp):
            loss=model.loss(truth,observed,starts)
        assert torch.isfinite(loss)
        loss.backward();assert any(p.grad is not None for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();model.zero_grad(set_to_none=True)
        if args.model=='v125':
            model.after_step();model.publish_epoch(1);model.inference_path='ensemble'
        model.requires_grad_(False).eval()
        if args.model=='CSDI':model.n_samples=2
        before=state_sha(model)
        with torch.no_grad():
            visible=torch.where(observed,truth,0.)
            result=repair(model,visible,observed,starts,2023)
            assert result.shape==truth.shape and torch.isfinite(result).all()
            assert torch.equal(result[observed],truth[observed])
            empty=observed.clone();empty[0,:,0]=False;empty[1]=False
            empty_result=repair(model,torch.where(empty,truth,0.),empty,starts,2023)
            assert torch.isfinite(empty_result).all()
        with tempfile.TemporaryDirectory(prefix='beijing-preflight-') as directory:
            out=Path(directory);(out/'stage1').mkdir()
            torch.save(model.state_dict(),out/'stage1/selected.pt')
            restored=build_adapter(args.model,data,device).eval()
            restored.load_state_dict(torch.load(out/'stage1/selected.pt',map_location=device,weights_only=True))
            assert state_sha(restored)==before
            if args.model=='CSDI':restored.n_samples=2
            tiny=copy.copy(data);tiny.starts={s:x[:2] for s,x in data.starts.items()}
            cache=RepairCache(model,tiny,out,device,2)
            cache.load('train');cache.load('validation');cache.load('test',data.test_seeds[0])
            reused=RepairCache(restored,tiny,out,device,2,allow_generation=False)
            # If the cached forecast path re-enters the repair network it fails.
            def forbidden(*unused):raise AssertionError('cache reuse called repair model')
            restored.forward=forbidden
            completed,seen,cached_starts=next(reused.batches('train',2))
            head=build_head(data,device);head_sha=state_sha(head)
            assert state_sha(build_head(data,device))==head_sha
            prediction=forecast(head,completed.to(device),seen.to(device),data,cached_starts)
            assert prediction.shape==(2,12,96)
            future=torch.from_numpy(np.stack([data.values[s+96:s+192] for s in cached_starts])).to(device).permute(0,2,1)
            valid=torch.from_numpy(data.valid_for(cached_starts,96)).to(device).permute(0,2,1)
            head_optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
            (prediction-future)[valid].square().mean().backward();head_optimizer.step()
            assert all(p.grad is None for p in model.parameters()) and state_sha(model)==before
        records.append(dict(scene=scene,loss=float(loss),checkpoint_round_trip=True,cache_only_forecast=True,
            common_head_initial_sha256=head_sha,shape=[2,96,12],empty_input_finite=True,
            validation_windows=3411,test_windows=6919,cuda_device=torch.cuda.get_device_name(0),
            csdi_samples='2_preflight_only_production5' if args.model=='CSDI' else None))
        print(records[-1],flush=True)
        del model,restored,head,optimizer,cache,reused
        torch.cuda.empty_cache()
    write_json(args.root/f'preflight_{args.model}.json',dict(passed=True,model=args.model,cases=records))


if __name__=='__main__':main()
