import json
import os
from pathlib import Path
import numpy as np
import pytest
from integrations.forecast_compare.protocol import load_p4_protocol,iter_p4_batches,with_training_seed
from integrations.forecast_compare.missing_scenarios import SCENARIOS,load_scenario_test_bank
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from malr.data.masks import EpochMaskManager
from malr.evaluation.report import file_sha256

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/t1_v41_higher_missing_seed502_20261004')
CONFIG=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'
MASKS=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'

@pytest.mark.parametrize('name,rate,length',SCENARIOS)
def test_scenario_changes_train_masks_and_shares_exact_support(monkeypatch,name,rate,length):
    monkeypatch.delenv('MALR_MISSING_SCENARIO',raising=False)
    base=load_p4_protocol(CONFIG,MASKS)
    monkeypatch.setenv('MALR_MISSING_SCENARIO',str(ROOT/name/'scenario.json'))
    p=with_training_seed(load_p4_protocol(CONFIG,MASKS),502)
    assert np.array_equal(p.standardized,base.standardized) and p.refs==base.refs
    assert p.config.missing.start_rate==rate and p.config.missing.block_len==length
    assert p.mask_sha256['validation'] != base.mask_sha256['validation']
    a=next(iter_p4_batches(p,'train',epoch=7,batch_size=16))
    b=next(iter_p4_batches(p,'train',epoch=7,batch_size=128))
    assert a.refs==b.refs[:16]
    assert np.array_equal(a.observed_mask.numpy(),b.observed_mask[:16].numpy())
    expected=EpochMaskManager(8640,7,rate,length,502).mask_for_sample(7,a.refs[0].target_start,length=96)
    assert np.array_equal(a.observed_mask[0].numpy(),expected)
    bank=ROOT/name/'test_masks/manifest.json'
    views=load_test_bank(p,bank)
    assert len(views)==20 and all(v.mask_sha256['validation']==p.mask_sha256['validation'] for v in views)
    other=next(n for n,_,_ in SCENARIOS if n!=name)
    with pytest.raises(ValueError):load_scenario_test_bank(p,ROOT/other/'test_masks/manifest.json')


def test_seal_tampering_is_rejected(monkeypatch,tmp_path):
    p=tmp_path/'bad.json';p.write_text('{}');p.with_suffix('.sha256').write_text('bad')
    monkeypatch.setenv('MALR_MISSING_SCENARIO',str(p))
    with pytest.raises(ValueError,match='seal'):load_p4_protocol(CONFIG,MASKS)
