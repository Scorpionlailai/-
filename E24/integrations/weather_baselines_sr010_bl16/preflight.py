"""Risk checks for author adapters, isolated from all formal model training."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import _atomic_json
from integrations.forecast_compare.protocol import iter_p4_batches
from .models import build_author,build_head,frozen_forecaster,S4MWrapper
from .run import seed,autocast,objective,forward,freeze_hook


def check(args,cfg,profiles,protocol,device,ident):
    seed(cfg['seed']);model=build_author(args.model,cfg,protocol,device)
    batch=next(iter_p4_batches(protocol,'train',epoch=0,batch_size=cfg['training']['batch_size'])).to(device)
    stage='repair' if cfg['kind']=='imputer' else 'forecast'
    stamps=None
    if args.model=='CoIFNet':
        from integrations.forecast_compare.coifnet_p4 import load_timestamp_features
        stamps=load_timestamp_features(Path(protocol.config.data.datasets['Weather'].path))
    model.train()
    if isinstance(model,S4MWrapper):model.warm(batch)
    with autocast(cfg['training'],device):loss=objective(args.model,model,batch,stamps,cfg,stage)
    if not torch.isfinite(loss):raise FloatingPointError('preflight loss nonfinite')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    model.zero_grad(set_to_none=True)
    frozen=False;head_sha=None
    if cfg['kind']=='imputer':
        batch=next(iter_p4_batches(protocol,'train',epoch=0,batch_size=profiles['shared_forecast_stage']['batch_size'])).to(device)
        seed(2023);head=build_head(protocol).to(device);head_sha=parameter_hash(head)
        assert head_sha=='fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561'
        model=frozen_forecaster(model,head);freeze_hook(model);model.train()
        assert all(not m.training for m in model.imputer.modules())
        assert not any(p.requires_grad or p.grad is not None for p in model.imputer.parameters())
        before=parameter_hash(model.imputer)
        optimizer=torch.optim.Adam([p for p in model.predictor.parameters() if p.requires_grad],lr=.0001)
        assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.predictor.parameters() if p.requires_grad}
        prediction,_=forward(args.model,model,batch,stamps)
        torch.nn.functional.mse_loss(prediction,batch.forecast_target).backward();optimizer.step()
        assert before==parameter_hash(model.imputer)
        frozen=True
    model.eval()
    with torch.no_grad():
        torch.manual_seed(2023)
        a,repaired=forward(args.model,model,batch,stamps)
        changed=replace(batch,normalized_target=batch.normalized_target+111.,forecast_target=batch.forecast_target+999.)
        torch.manual_seed(2023)
        b,_=forward(args.model,model,changed,stamps)
        assert a.shape==batch.forecast_target.shape and torch.isfinite(a).all()
        assert torch.equal(a,b), 'hidden/future labels leaked into predictions'
    result=dict(status='passed',model=args.model,loss=float(loss.detach()),batch_size=len(batch.refs),
                output_shape=list(a.shape),finite_backward=True,labels_excluded=True,
                shared_head_initial_sha256=head_sha,stage2_freeze_checked=frozen,
                smoke_weights_discarded=True,identity=ident)
    args.output.mkdir(parents=True,exist_ok=True)
    _atomic_json(result,args.output/'acceptance.json');print(json.dumps(result),flush=True)
