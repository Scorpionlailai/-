"""External data root plus packaged legacy protocol references; no local paths."""
from pathlib import Path
import importlib,os
import numpy as np
import bootstrap

def load_data(dataset,scene,assets):
    if dataset == 'Exchange-Rate':
        from exchange_data import SharedData
        data=SharedData(dataset,scene,assets)
    elif dataset in ['ETTh1','Weather']:
        module=importlib.import_module('protocol_data')
        module.DATA_ROOT=Path(os.environ.get('R2_DATA_ROOT',str(bootstrap.ROOT))).resolve()
        data=module.SharedData(dataset,scene,assets)
    else:
        packages={'BeijingAirQuality':'beijing_baselines.data','PAMAP2':'v125_ablation.pamap2_data','PhysioNet2012':'v125_ablation.physionet_data'}
        if dataset not in packages:raise ValueError('unsupported dataset '+dataset)
        module=importlib.import_module('baseline_physionet_data' if dataset=='PhysioNet2012' else 'integrations.'+packages[dataset]);oldroot=module.ROOT
        try:
            module.ROOT=Path(os.environ.get('R2_DATA_ROOT',str(bootstrap.ROOT))).resolve()
            data=module.SharedData(dataset,scene,assets)
        finally:module.ROOT=oldroot
    if not hasattr(data,'valid_for'):data.valid_for=lambda starts,offset=0:np.ones((len(starts),96,data.n_vars),dtype=bool)
    return data
