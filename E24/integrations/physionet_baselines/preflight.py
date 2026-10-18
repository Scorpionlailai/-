"""Real GPU gradients, edge inputs, checkpoint round trip and cached head step."""
import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch

from .adapters import build_adapter, training_profile
from .data import SharedData, SCENES, write_json
from .run import seed_all, state_sha, repair, build_head, forecast
from .cache import RepairCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda:0');records=[]
    full_batch_probe=None
    for scene in SCENES:
        seed_all(2023)
        data=SharedData('PhysioNet2012',scene,args.root/'shared_assets')
        model=build_adapter(args.model,data,device)
        if full_batch_probe is None:
            profile=training_profile(args.model)
            actual_batch=profile['batch_size']
            large_truth,large_observed,large_starts=next(data.batches('train',actual_batch))
            large_truth,large_observed=large_truth.to(device),large_observed.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            large_optimizer=torch.optim.Adam(model.parameters(),lr=profile['learning_rate'])
            for probe_step in range(2 if args.model=='v125' else 1):
                if args.model=='v125':
                    model.inference_path='coarse';model.epoch=probe_step+1
                    model.core.historical.retrieval.begin_epoch(probe_step+1)
                large_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.model=='ImputeFormer'):
                    large_loss=model.loss(large_truth,large_observed,large_starts)
                assert torch.isfinite(large_loss)
                large_loss.backward();large_optimizer.step()
                if args.model=='v125':model.after_step();model.publish_epoch(probe_step+1)
            if args.model=='CSDI':
                model.eval()
                with torch.no_grad():
                    production=repair(model,torch.where(large_observed,large_truth,0.),large_observed,large_starts,2023)
                    assert torch.isfinite(production).all()
                del production
            torch.cuda.synchronize()
            full_batch_probe=dict(batch_size=actual_batch,n_vars=data.n_vars,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                optimizer_steps=probe_step+1,csdi_production_samples=5 if args.model=='CSDI' else None)
            print(dict(full_batch_probe=full_batch_probe),flush=True)
            del model,large_optimizer,large_truth,large_observed,large_loss
            torch.cuda.empty_cache();seed_all(2023)
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
            assert prediction.shape==(2,data.n_vars,96)
            future=torch.from_numpy(np.stack([data.values[s+96:s+192] for s in cached_starts])).to(device).permute(0,2,1)
            valid=torch.from_numpy(data.valid_for(cached_starts,96)).to(device).permute(0,2,1)
            head_optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
            (prediction-future)[valid].square().mean().backward();head_optimizer.step()
            assert all(p.grad is None for p in model.parameters()) and state_sha(model)==before
        records.append(dict(scene=scene,loss=float(loss),checkpoint_round_trip=True,cache_only_forecast=True,
            common_head_initial_sha256=head_sha,shape=[2,96,data.n_vars],empty_input_finite=True,
            validation_windows=len(data.starts['validation']),test_windows=len(data.starts['test']),cuda_device=torch.cuda.get_device_name(0),
            csdi_samples='2_preflight_only_production5' if args.model=='CSDI' else None))
        print(records[-1],flush=True)
        del model,restored,head,optimizer,cache,reused
        torch.cuda.empty_cache()
    write_json(args.root/f'preflight_{args.model}.json',dict(passed=True,model=args.model,cases=records,
               full_training_batch_probe=full_batch_probe))


if __name__=='__main__':main()
