import json
from pathlib import Path
import yaml
import numpy as np
from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.missing_scenarios import SCENARIOS,TEST_SEEDS,apply_scenario,load_scenario_test_bank
from integrations.forecast_compare.test_mask_bank import _missing_fractions
from malr.data.masks import generate_variable_missing_mask,save_fixed_mask
from malr.evaluation.report import file_sha256

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/t1_v41_higher_missing_seed502_20261004')
MASKS=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def seal(p,d):write(p,d);p.with_suffix('.sha256').write_text(file_sha256(p)+'\n')
def main():
    base=load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',MASKS)
    rows=[]
    for name,rate,length in SCENARIOS:
        cell=ROOT/name;cell.mkdir();(cell/'masks').mkdir();(cell/'test_masks').mkdir()
        config=yaml.safe_load(base.config_path.read_text());config['missing'].update(start_rate=rate,block_len=length)
        cp=cell/'protocol.yaml';cp.write_text(yaml.safe_dump(config,sort_keys=False))
        entries={}
        for split in ('validation','test'):
            b=base.bounds.bounds_for(split)
            mask,meta=generate_variable_missing_mask(b.end-b.start+96,7,rate,length,2021,dataset='ETTh1',split=split,support_start=b.start-96)
            p=cell/'masks'/f'ETTh1_{split}.npz';save_fixed_mask(p,mask,meta)
            entries[split]=dict(path=str(p.relative_to(cell)),sha256=file_sha256(p),seed=2021,missing_fraction=meta.final_rate)
        manifest=cell/'scenario.json'
        seal(manifest,dict(schema_version=1,scenario=name,start_rate=rate,block_len=length,allow_block_overlap=True,
            source_csv_sha256=base.source_sha256,base_config_sha256=file_sha256(base.config_path),
            config_path=cp.name,config_sha256=file_sha256(cp),masks=entries))
        protocol=apply_scenario(base,manifest);groups=[]
        b=protocol.bounds.test
        for i,seed in enumerate(TEST_SEEDS):
            mask,meta=generate_variable_missing_mask(b.end-b.start+96,7,rate,length,seed,dataset='ETTh1',split='test',support_start=b.start-96)
            p=cell/'test_masks'/f'test_group_{i:02d}.npz';save_fixed_mask(p,mask,meta)
            timeline,window,count=_missing_fractions(protocol,mask)
            groups.append(dict(group_id=f'group_{i:02d}',path=p.name,sha256=file_sha256(p),seed=seed,
                missing_timeline_fraction=timeline,missing_evaluation_fraction=window,evaluation_windows=count))
        bank=cell/'test_masks/manifest.json'
        seal(bank,dict(schema_version=3,scenario=name,source_csv_sha256=base.source_sha256,
            validation_mask_sha256=protocol.mask_sha256['validation'],start_rate=rate,block_len=length,
            seq_len=96,pred_len=96,n_vars=7,seeds=list(TEST_SEEDS),groups=groups,
            purpose='fixed before training; same 20 test masks for both models; never used for selection'))
        assert len(load_scenario_test_bank(protocol,bank))==20
        rows.append(dict(name=name,start_rate=rate,block_len=length,scenario_manifest=str(manifest),test_bank=str(bank),
            actual_test_missing_mean=float(np.mean([g['missing_evaluation_fraction'] for g in groups]))))
    write(ROOT/'plan.json',dict(scenarios=rows,training_seed=502,models=['E23-v41-slow','T1'],device='cuda:1',
        selection='fixed validation MAE',head_parameters=35427,predictor_initial_sha256='26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c',
        e23_config=str(ROOT/'configs/e23_v41_slow_seed502.yaml'),author_profiles=str(ROOT/'configs/author_seed502.json'),
        stage2_difference='E23 original v41 mixed loss/retention/slow update/legacy padding; T1 original author pure-MSE shared forecast training; fixed head source/config/initialization',
        test_used_for_selection=False,new_training_per_scenario=True))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
