"""A5 comparison sign and an all-checkpoints-before-test barrier."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

OUT=Path(__file__).resolve().parents[1]
REPO=Path('/home/xyw/paper-e23-v125-ablation')
COARSE=Path('/home/xyw/paper/E24/artifacts/v125_pamap2_coarse_seed_stability_20261019')
CAL=Path('/home/xyw/paper/E24/artifacts/v125_calibration_confirmation_20261020')
DIAG=Path('/home/xyw/paper/E24/artifacts/v125_contribution_diagnostics_20261019')
BASE='50a2eb84dbeeec3e559a1f551091bdc034fb06ac'

def external(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def training_modules():
    sys.path.insert(0,str(CAL/'tools'))
    coarse=external('round1_locked_worker',COARSE/'tools/worker.py')
    calibration=external('round2_locked_calibration',CAL/'tools/calibration_worker.py')
    return coarse,calibration

def require_barrier(checkpoints):
    expected={f'A5_{seed}_{phase}' for seed in (2023,2024,2025) for phase in ('coarse','calibration64','calibration256')}
    if not expected.issubset(checkpoints) or any(not checkpoints[k] for k in expected):
        raise ValueError('all A5 coarse and calibration checkpoints must be locked before test replay')
    return True

def paired_training_summary(rows):
    out={};maps={}
    for arm in ('A0','A5'):
        take=sorted([x for x in rows if x['variant']==arm],key=lambda x:x['training_seed'])
        means=[float(np.mean(x['values'])) for x in take];seeds=[x['training_seed'] for x in take]
        out[arm]=dict(training_seeds=seeds,training_means=means,n_training_seeds=len(take),
            mean=float(np.mean(means)) if means else None,training_std_ddof1=float(np.std(means,ddof=1)) if len(means)>1 else None)
        maps[arm]=dict(zip(seeds,means))
    common=sorted(set(maps['A0']) & set(maps['A5']))
    out['paired_A5_minus_A0']=[maps['A5'][s]-maps['A0'][s] for s in common]
    return out
