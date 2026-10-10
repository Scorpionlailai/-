"""Read-only final T1 repair diagnostics on fixed validation, no training/test."""
from pathlib import Path
import hashlib,json,os
import torch
from integrations.e23.config import load_e23_p4_protocol
from integrations.forecast_compare.author_profiles import build_author_model,load_author_profiles
from integrations.forecast_compare.author_provenance import author_source_identity
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parent;WORK=Path('/home/xyw/paper/.e23-v74-pre-tail-attention/E23')
BASE=ROOT.parent/'t1_v41_higher_missing_seed502_20261004'
scene=BASE/'sr010_bl16';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
profiles_path=Path(read(BASE/'plan.json')['author_profiles']);cfg=load_author_profiles(profiles_path)['models']['T1']
os.environ['MALR_MISSING_SCENARIO']=str(scene/'scenario.json');torch.set_num_threads(4);device=torch.device('cuda:1')
protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
selection=scene/'T1/training/forecast-selection.json';rec=read(selection)
assert protocol.mask_sha256['validation']==rec['validation_mask_sha256']
assert sha(rec['checkpoint'])==rec['checkpoint_sha256']; before=(sha(selection),sha(rec['checkpoint']))
identity=author_source_identity(profiles_path,cfg,protocol);assert all(rec.get(k)==v for k,v in identity.items())
payload=torch.load(rec['checkpoint'],map_location=device,weights_only=False); assert all(payload.get(k)==v for k,v in identity.items())
model=build_author_model('T1',cfg,WORK/cfg['root'],device).eval()
state={k[len('imputer.'):]:v for k,v in payload['model_state'].items() if k.startswith('imputer.')};model.load_state_dict(state,strict=True)
totals={k:[0.,0.,0] for k in ['all96','before64','last32']}
with torch.inference_mode():
    for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
        batch=cpu.to(device);values,mask=t1_inputs(batch);candidate=model(values,mask=mask)
        error=candidate-t1_clean_history(batch);hidden=~batch.observed_mask
        position=torch.arange(96,device=device)[None,:,None]
        for region,m in [('all96',hidden),('before64',hidden&(position<64)),('last32',hidden&(position>=64))]:
            e=error[m].double(); totals[region][0]+=e.square().sum().item();totals[region][1]+=e.abs().sum().item();totals[region][2]+=e.numel()
assert before==(sha(selection),sha(rec['checkpoint']))
result=dict(model='T1',stage='selected_stage2',split='fixed_validation',metric_space='global_train_standardized',precision='float32',selection=str(selection),checkpoint_sha256=rec['checkpoint_sha256'],validation_mask_sha256=protocol.mask_sha256['validation'],source_sha256=protocol.source_sha256,metrics={k:dict(mse=s/n,mae=a/n,hidden_coordinates=n) for k,(s,a,n) in totals.items()},no_training=True,no_test_scoring=True)
with (ROOT/'t1-final-repair-diagnostics.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
