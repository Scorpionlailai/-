import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from integrations.repair_baselines.data import ROOT, file_sha256, write_json
from integrations.pamap2_baselines.run import evaluate, metric, aggregate, forecast, resources, state_sha, seed_all, synchronize
from integrations.physionet_baselines.losses import hidden_error, mean_or_zero

STUDY = ROOT / 'E24/artifacts/v125_ablation_20261018'


def data_for(dataset, scene, root=STUDY):
    if dataset == 'PAMAP2':
        from .pamap2_data import SharedData
    elif dataset == 'PhysioNet2012':
        from .physionet_data import SharedData
    else:
        raise ValueError(dataset)
    return SharedData(dataset, scene, Path(root)/'shared_assets')


def implementation():
    paths = list((ROOT/'E24/integrations/v125_ablation').rglob('*.py'))
    paths += list((ROOT/'E24/src/malr').rglob('*.py'))
    paths += list((ROOT/'E23/versions/e23-v125-shared-calibration-20261009/E23').rglob('*.py'))
    for package in ('pamap2_baselines','physionet_baselines'):
        paths += list((ROOT/'E24/integrations'/package).glob('*.py'))
    paths += [ROOT/'E24/configs/e24_v79.yaml']
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(paths)}


def verify_lock(root):
    lock = json.loads((Path(root)/'design_lock.json').read_text())
    if lock['implementation'] != implementation():
        raise RuntimeError('implementation differs from pre-test design lock; refusing formal run')
    return lock


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def parameter_hashes(model):
    return {key:hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for key,value in model.named_parameters()}


def output_path(root, dataset, scene, variant, seed):
    return Path(root)/dataset/scene/variant/f'seed_{seed}'


def append(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
