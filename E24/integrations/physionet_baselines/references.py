"""T1 and the immutable v125 coarse-plus-two-calibrators reference."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
import yaml

from .data import ROOT, write_json, file_sha256
from .losses import hidden_error, mean_or_zero

V125_ROOT = ROOT / 'E23/versions/e23-v125-shared-calibration-20261009/E23'


def snapshot_module(kind, name):
    package = '_physionet_v125_' + kind
    path = V125_ROOT / ('src/malr/models/e23' if kind == 'models' else 'integrations/e23')
    if package not in sys.modules:
        module = ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    return importlib.import_module(package + '.' + name)


class T1Adapter(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data
        from integrations.forecast_compare.t1_p4 import T1_SETTINGS
        self.config = dict(T1_SETTINGS, enc_in=data.n_vars)
        source = ROOT / 'T1/models/T1.py'
        spec = importlib.util.spec_from_file_location('_physionet_official_t1', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.core = module.Model(SimpleNamespace(**self.config))

    def forward(self, values, observed, starts):
        candidate = self.core(values, mask=observed.to(values.dtype))
        return torch.where(observed, values, candidate)

    def loss(self, truth, observed, starts):
        valid = torch.from_numpy(self.data.valid_for(starts)).to(truth.device)
        candidate = self(torch.where(observed, truth, 0.), observed, starts)
        return mean_or_zero(hidden_error(candidate, truth, observed, valid).square())


class V125Adapter(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data
        cfg = yaml.safe_load((ROOT / 'E24/configs/e24_v79.yaml').read_text())
        options = dict(cfg['model'])
        options.pop('residual_scale')
        for key in ('temporal_boundaries', 'variable_offsets'):
            options[key] = tuple(options[key])
        model_module = snapshot_module('models', 'model')
        self.core = model_module.MultiExpertRepairEncoder(variable_count=data.n_vars,
            window_length=96, patch_length=16, stride=8, retrieval_dim=64, **options)
        self.coarse_projector = self.core.recent_mean_projector
        ensemble = snapshot_module('models', 'calibration_ensemble').RecentMeanEnsembleProjector
        self.core.recent_mean_projector = ensemble(data.n_vars, widths=[64,256], weights=[.5,.5])
        self.config = dict(coarse=cfg, source=str(V125_ROOT), calibration_widths=[64,256],
            calibration_epochs=[100,100], weights=[.5,.5], coarse_epochs=100,
            source_checkpoint_transfer='same-data coarse selected by validation MSE; inherited mean weights',
            natural_missing='masked repair, structure and key supervision; observed-only memory',
            contrastive_execution='checkpointed_query_row_blocks512_all_original_negatives',
            memory_deserialization='preserve_original_float64_member_keys_exactly')
        from malr.prototypes.online_bank import OnlinePrototypeMemory
        self.memory = OnlinePrototypeMemory(patch_len=16, patch_stride=8, train_end=data.bounds.train.end, variable_count=data.n_vars)
        self.known = set()
        self.register_buffer('memory_blob', torch.empty(0, dtype=torch.uint8))
        # These encoders otherwise live outside the PyTorch module tree.
        self.ema_encoder = self.core.historical.retrieval.ema_encoder
        self.frozen_encoder = self.core.historical.retrieval.frozen_encoder
        self.inference_path = 'ensemble'
        self.epoch, self.batch_index = 0, 0

    def publish_epoch(self, epoch):
        snapshot_module('training', 'train')._publish_epoch(self.memory, self.core, epoch)
        raw = json.dumps(self.memory.state_dict(), sort_keys=True, separators=(',',':'),
                         default=lambda value: value.tolist()).encode()
        self.memory_blob = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(next(self.parameters()).device)

    def _load_from_state_dict(self, state, prefix, *args):
        blob = state[prefix+'memory_blob']
        self.memory_blob = torch.empty_like(blob, device=self.memory_blob.device)
        super()._load_from_state_dict(state, prefix, *args)
        if self.memory_blob.numel():
            payload = json.loads(self.memory_blob.cpu().numpy().tobytes())
            self.memory.load_state_dict(payload)
            # The legacy loader normalizes already-unit float64 keys again,
            # changing their last bits. Restore the validated serialized keys
            # exactly, so the actual memory snapshot agrees with the checkpoint.
            if payload['pending']:raise ValueError('only published memory can be a selected checkpoint')
            for groups,serialized_groups in zip(self.memory.snapshot().groups,payload['snapshot']['groups']):
                for group,serialized_group in zip(groups,serialized_groups):
                    for member,serialized in zip(group.members,serialized_group['members']):
                        key=np.asarray(serialized['clean_key'],dtype=np.float64)
                        if not np.isfinite(key).all() or abs(np.linalg.norm(key)-1.)>1e-12:
                            raise ValueError('checkpoint contains a non-unit prototype key')
                        key.setflags(write=False)
                        object.__setattr__(member,'clean_key',key)
            self.known = set(self.memory.snapshot().seen)
            self.core.historical.refresh_snapshot(self.memory.snapshot())

    def context(self, truth, observed, starts):
        from malr.data.normalization import masked_window_normalize, fit_train_variable_stats
        if not hasattr(self.data, '_reference_fallback'):
            self.data._reference_fallback = fit_train_variable_stats(torch.from_numpy(self.data.values[:self.data.bounds.train.end]), eps=0.)
        norm = masked_window_normalize(truth, observed, self.data._reference_fallback, eps=1e-5)
        positions = torch.as_tensor(starts,device=truth.device)[:,None] + torch.arange(self.core.patch_count,device=truth.device)[None]*8
        return norm, positions

    def output(self, truth, observed, starts, *, training_coarse=False):
        norm, positions = self.context(truth, observed, starts)
        ensemble = self.core.recent_mean_projector
        if self.inference_path == 'coarse':
            projector = self.coarse_projector
        elif self.inference_path.startswith('calibration'):
            projector = ensemble.members[0 if self.inference_path=='calibration64' else 1]
        else:
            projector = ensemble
        self.core.recent_mean_projector = projector
        try:
            result = self.core(norm.student, observed, norm.mean, norm.std, positions,
                epoch=self.epoch, explore_variable=training_coarse,
                specialist_exposure=.2 if training_coarse else 0.,
                specialist_seed=2023+self.epoch*100000+self.batch_index)
        finally:
            self.core.recent_mean_projector = ensemble
        return result, norm, positions

    def forward(self, values, observed, starts):
        result, _, _ = self.output(values, observed, starts)
        return torch.where(observed, values, result.explicit.standardized_sequence)

    def loss(self, truth, observed, starts):
        result, norm, positions = self.output(truth, observed, starts, training_coarse=True)
        valid = torch.from_numpy(self.data.valid_for(starts)).to(truth.device)
        candidate = result.explicit.standardized_sequence
        error = hidden_error(candidate, truth, observed, valid)
        hidden = ~observed & valid
        tail = torch.where(hidden, candidate-truth, 0.)[:,-32:]
        structure = tail.mean(1).square().mean()
        from malr.data.patching import patchify_with_state, patchify_preserving_values
        from malr.prototypes.retrieval import masked_pair_loss
        from .losses import chunked_contrastive_loss
        query = patchify_with_state(norm.student, observed,16,8)
        clean = patchify_with_state(torch.where(valid,norm.target,0.),valid,16,8)
        eligible = query.rho.lt(1.) & clean.effective_mask.any(-1)
        if eligible.any():
            encoder = self.core.historical.key_encoder
            q = encoder(query.values[eligible],query.effective_mask[eligible])
            t = encoder(clean.values[eligible],clean.effective_mask[eligible])
            retrieval = masked_pair_loss(q,t) + .5*chunked_contrastive_loss(q,t)/np.log(max(len(q),2))
        else:
            retrieval = error.sum()*0.
        from integrations.forecast_compare.protocol import P4Batch
        self._pending_batch = (P4Batch(norm.student,norm.target,observed,norm.mean,norm.std,
            truth.new_empty(len(truth),self.data.n_vars,0)), positions)
        return mean_or_zero(error.square())+.1*retrieval+structure

    def after_step(self):
        batch, positions = self._pending_batch
        snapshot_module('training', 'train')._stage_observed(self.memory,self.core,batch,positions,self.known)
        del self._pending_batch


def build_reference(name, data, device):
    return (V125Adapter(data) if name=='v125' else T1Adapter(data)).to(device)


def train_v125(args, model, data, output, device, profile):
    from .run import evaluate, resources, synchronize, seed_all
    stage = output/'stage1'
    stage.mkdir()
    started = time.perf_counter()
    steps, selected = 0, {}
    model.inference_path = 'coarse'
    # Calibration modules are excluded from the coarse optimizer and remain frozen.
    for p in model.core.recent_mean_projector.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    best = float('inf')
    for epoch in range(1,101):
        model.epoch = epoch
        model.core.historical.retrieval.begin_epoch(epoch)
        model.train()
        losses=[]
        for index,(truth,observed,starts) in enumerate(data.batches('train',128,epoch=epoch-1),1):
            model.batch_index = index
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(truth.to(device),observed.to(device),starts)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite v125 coarse loss')
            loss.backward();optimizer.step();model.after_step()
            losses.append(loss.item());steps+=1
        model.publish_epoch(epoch)
        validation=evaluate(model,data,'validation',device,batch_size=128)
        if validation['full96']['mse']<best:
            best=validation['full96']['mse'];selected['coarse']=epoch
            torch.save(model.state_dict(),stage/'coarse_selected.pt')
        record=dict(subphase='coarse',epoch=epoch,total_epoch=epoch,train_loss=float(np.mean(losses)),validation=validation,optimizer_steps=steps,elapsed_seconds=time.perf_counter()-started)
        with (stage/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write_json(output/'status.json',dict(phase='stage1_training',**record));print(json.dumps(record),flush=True)
    model.load_state_dict(torch.load(stage/'coarse_selected.pt',map_location=device,weights_only=True))
    coarse_state=model.coarse_projector.state_dict()
    for index,width in enumerate((64,256)):
        member=model.core.recent_mean_projector.members[index]
        member.load_state_dict(coarse_state,strict=False)
        model.requires_grad_(False)
        member.requires_grad_(True)
        model.inference_path=f'calibration{width}'
        seed_all(2023)
        optimizer=torch.optim.Adam(member.parameters(),lr=.001)
        best=float('inf')
        for epoch in range(1,101):
            model.eval();member.train();losses=[]
            for truth,observed,starts in data.batches('train',128,epoch=epoch-1):
                truth,observed=truth.to(device),observed.to(device)
                valid=torch.from_numpy(data.valid_for(starts)).to(device)
                optimizer.zero_grad(set_to_none=True)
                candidate=model(torch.where(observed,truth,0.),observed,starts)
                error=hidden_error(candidate,truth,observed,valid)
                tail=hidden_error(candidate[:,-32:],truth[:,-32:],observed[:,-32:],valid[:,-32:])
                loss=mean_or_zero(error.abs())+.25*mean_or_zero(tail.abs())
                if not torch.isfinite(loss): raise FloatingPointError('non-finite v125 calibration loss')
                loss.backward();optimizer.step();losses.append(loss.item());steps+=1
            validation=evaluate(model,data,'validation',device,batch_size=128)
            if validation['full96']['mae']<best:
                best=validation['full96']['mae'];selected[f'calibration{width}']=epoch
                torch.save(member.state_dict(),stage/f'calibration{width}_selected.pt')
            record=dict(subphase=f'calibration{width}',epoch=epoch,total_epoch=100+index*100+epoch,
                train_loss=float(np.mean(losses)),validation=validation,optimizer_steps=steps,elapsed_seconds=time.perf_counter()-started)
            with (stage/'curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(output/'status.json',dict(phase='stage1_training',**record));print(json.dumps(record),flush=True)
        member.load_state_dict(torch.load(stage/f'calibration{width}_selected.pt',map_location=device,weights_only=True))
    model.inference_path='ensemble';model.requires_grad_(False).eval()
    torch.save(model.state_dict(),stage/'selected.pt')
    synchronize(device)
    summary=dict(profile=profile,actual_epochs=300,optimizer_steps=steps,selected_epochs=selected,
        checkpoint_sha256=file_sha256(stage/'selected.pt'),training_seconds=time.perf_counter()-started,
        combination_weights=[.5,.5],resources=resources(device),history_snapshot_hash=model.memory.snapshot_hash())
    write_json(stage/'selection.json',summary)
    return summary
