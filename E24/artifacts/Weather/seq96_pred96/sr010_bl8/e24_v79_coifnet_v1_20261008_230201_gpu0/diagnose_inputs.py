from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import torch,json
from integrations.e24_coifnet.run import *
torch.set_num_threads(4)
a=SimpleNamespace(parent=DEFAULT_PARENT,e24_config=ROOT/'configs/e24_v79.yaml')
m=read_manifest(DEFAULT_MANIFEST)
p=load_dataset_protocol(DEFAULT_MANIFEST,config_path=Path(m['config_path']),artifacts=DEFAULT_MANIFEST.parent)
c=load_author_profiles(DEFAULT_PROFILES)['models']['CoIFNet'];p=replace(p,training_seed=2023)
model=construct(a,p,c,torch.device('cpu')).eval();b=next(iter_p4_batches(p,'train',batch_size=512,epoch=0));t=load_timestamp_features(Path(p.config.data.datasets['Weather'].path))
with torch.no_grad():
 r=model.repaired_history(b);v,mask,ts=coifnet_inputs(b,t)
 original=model.coifnet.impute(v,mask,feat=ts)
 print('NATIVE',author_coifnet_loss(original,v,mask,b.forecast_target.permute(0,2,1),weight=.2).item(),flush=True)
 hybrid=repaired_coifnet_output(model.coifnet,r,v,mask,ts)
 print('HYBRID',author_coifnet_loss(hybrid,v,mask,b.forecast_target.permute(0,2,1),weight=.2).item(),flush=True)
 z=model.coifnet.revin_layer(r,'norm-fore');clean=b.normalized_target*b.query_std[:,None]+b.query_mean[:,None]
 print(json.dumps({'repaired_max':r.abs().max().item(),'clean_max':clean.abs().max().item(),'hidden_repair_mse':((r-clean)[~b.observed_mask]).square().mean().item(),'normalized_repaired_max':z.abs().max().item(),'normalized_visible_max':model.coifnet.revin_layer(v,'norm',mask).abs().max().item(),'pred_max':hybrid.abs().max().item(),'per_variable_repaired_max':r.abs().amax((0,1)).tolist(),'per_variable_std_min':model.coifnet.revin_layer.stdev.amin((0,1)).tolist()}),flush=True)
