import sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import generate_tables as g

def test_all_six_formal_datasets_have_four_unique_scenes():
    rows=g.formal_results()
    assert len(rows)==24
    assert {r['dataset'] for r in rows}==set(g.DATASETS)
    for dataset in g.DATASETS:
        assert {r['scene'] for r in rows if r['dataset']==dataset}==set(g.SCENES)
        assert all(r['training_seeds']==3 and r['test_masks']==20 for r in rows if r['dataset']==dataset)

def test_missing_and_duplicate_results_are_rejected():
    rows=g.formal_results()
    with pytest.raises(ValueError):g.validate_formal(rows[:-1])
    with pytest.raises(ValueError):g.validate_formal(rows+[rows[0]])

def test_analysis_does_not_replace_formal_pamap_numbers_or_seed_scope():
    formal=next(r for r in g.formal_results() if r['dataset']=='PAMAP2' and r['scene']=='sr015_bl8')
    core=next(r for r in g.core_results() if r['dataset']=='PAMAP2' and r['stage']=='final')
    assert formal['mse']==pytest.approx(.32666130377809155)
    assert core['e0_mse_mean']==pytest.approx(.3245715274876882)
    assert abs(formal['mse']-core['e0_mse_mean'])>.001
    component=g.component_results()
    assert all(int(r['n_training_seeds'])==1 for r in component if r['dataset']=='PAMAP2')
    assert all(int(r['n_training_seeds'])==3 for r in component if r['dataset']=='ETTh1')

def test_gain_uses_context_control_as_denominator():
    for r in g.core_results():
        expected=100*(r['e6_mse_mean']-r['e0_mse_mean'])/r['e6_mse_mean']
        assert r['mse_reduction_pct']==pytest.approx(expected)

def test_frozen_source_hashes_match():
    g.verify_snapshots()
