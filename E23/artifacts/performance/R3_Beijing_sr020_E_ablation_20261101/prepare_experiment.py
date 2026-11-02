from pathlib import Path
import shutil,json,hashlib
R=Path(__file__).resolve().parent
P=R.parent
B=P/'R3_final_three_datasets_3seed_20261030'
groups={'sources':'R3_ETTh1_candidate_ablation_20261031','context':'R3_ETTh1_context_summary_ablation_20261031','point':'R3_ETTh1_patch_shared_ablation_20261031'}
for group,source in groups.items():
 dst=R/group;dst.mkdir(exist_ok=True)
 shutil.copytree(P/source/'implementation',dst/'implementation',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 detail=(P/source/'evaluate_details.py').read_text()
 detail=detail.replace("A64_DATASET='ETTh1'","A64_DATASET='BeijingAirQuality'")
 detail=detail.replace("R2_DATA_ROOT='/home/xyw/paper'",f"R2_DATA_ROOT='{B/'data_root'}'")
 detail=detail.replace("load_data('ETTh1'","load_data('BeijingAirQuality'")
 detail=detail.replace("outputs/evaluation/ETTh1","outputs/evaluation/BeijingAirQuality")
 (dst/'evaluate_details.py').write_text(detail)
 protocol=json.loads((dst/'implementation/protocol.json').read_text())
 protocol.update(dataset='BeijingAirQuality',scenes=['sr020_bl8'],updates_each_stage=19000,epochs=100,seeds=[2023,2024,2025])
 if group=='context':protocol['variants']=['E6']
 (dst/'implementation/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 with (dst/'implementation/README.md').open('a') as f:f.write('\nBeijing sr020_bl8 explanatory replication, 2026-11-01. Same model operators and hyperparameters as corresponding ETTh1 ablation. Three seeds; identical original masks; frozen calibration. No forecast, no E7.\n')
 for stage in ['outputs','smoke_outputs']:
  dest=dst/stage/'assets/BeijingAirQuality/sr020_bl8'
  dest.parent.mkdir(parents=True,exist_ok=True)
  shutil.copytree(B/'outputs/assets/BeijingAirQuality/sr020_bl8',dest,dirs_exist_ok=True)
plan=dict(dataset='BeijingAirQuality',scene='sr020_bl8',seeds=[2023,2024,2025],variants=['E1','E2','E3','E4','E5','E6','E8'],reuse_E0=str(B),data_root=str(B/'data_root'),epochs_each_stage=100,updates_each_stage=19000,batch_size=128,test_masks=20,validation_mask_seed=2021,mask_order_seed=2023,main_selection='validation full96 MSE',calibration_selection='validation full96 MAE',calibration_frozen_entire_coarse=True,forecast=False,T1_reference_mse=.22552305693429445,T1_not_retrained=True,ignored=['E7'],interpretation='Ablation deltas are conditional R3 component effects, not additive attribution of the R3-T1 gap.',allocation={'0':['E6','E1','E3','E5'],'1':['E2','E4','E8']})
(R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for group in groups for p in sorted((R/group).rglob('*')) if p.is_file() and ('implementation' in p.parts or p.name=='evaluate_details.py')}
(R/'source_lock.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(R,len(files))
