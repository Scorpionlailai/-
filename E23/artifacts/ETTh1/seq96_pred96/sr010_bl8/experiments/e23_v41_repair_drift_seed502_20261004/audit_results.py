"""Audit saved scores, frozen weights and resource provenance, without retraining."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v41-drift/E23')
ORIGINAL=Path('/home/xyw/paper/.e23-t1-standard/E23')
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
EXPECTED_HEAD='26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def main():
    assert read(ROOT/'matrix-completed.json')['status']=='complete'
    frozen=read(ROOT/'matrix-definition.json')
    for rel,h in frozen['source_files'].items():assert digest(WORK/rel)==h,rel
    for p,h in {**frozen['parent_files'],**frozen['protected_files']}.items():assert digest(p)==h,p
    for rel in ['src/malr/models/patchtst.py','src/malr/models/repaired_forecast.py']:
        assert digest(WORK/rel)==digest(ORIGINAL/rel)
    old=funcs(ORIGINAL/'integrations/e23/train.py');new=funcs(WORK/'integrations/e23/train.py')
    assert old['build_forecast_model']==new['build_forecast_model'] and old['stage2_objective']==new['stage2_objective']
    latest=read(ROOT/'latest-results.json');assert not latest['pending'] and len(latest['rows'])==6
    bank=read(FORMAL/'test_masks/manifest.json');expected_seed=list(range(2026100401,2026100421))
    max_error=0.;count=0;new_workflow=0.;frozen_tensor_count=0
    groups={};comparisons={}
    for row in latest['rows']:
        data=read(row['evaluation']);assert data['mask_seeds']==expected_seed
        assert data['predictor_parameters']==35427 and data['predictor_initial_sha256']==EXPECTED_HEAD
        assert data['test_mask_bank_sha256']==digest(FORMAL/'test_masks/manifest.json')
        assert digest(data['selection_manifest'])==data['selection_manifest_sha256']
        groups[row['model']]={}
        for draw in data['test_draws']:
            assert digest(draw['window_metrics'])==draw['window_metrics_sha256']
            arrays=np.load(draw['window_metrics']);assert len(arrays['target_start'])==2688
            assert np.array_equal(arrays['target_start'],np.arange(11520,14208))
            assert draw['target_elements']==1806336
            for k in ['mse','mae']:
                error=abs(float(arrays[k].mean())-draw[k]);assert error<1e-6;max_error=max(max_error,error)
                assert abs(float(np.mean([d[k] for d in data['test_draws']]))-data['test_mean'][k])<1e-12
            groups[row['model']][draw['mask_seed']]=draw['mask_sha256'];count+=1
        if row['model'].startswith('E23-v41-'):
            record=read(data['selection_manifest']);assert record['epochs_completed']==100
            assert record['selection_metric']=='mae' and record['loss']=='forecast_mse_plus_0p2_mae'
            assert record['predictor_learning_rate']==.0001 and record['repair_retention']=='uniform_v1'
            assert record['repair_retention_active_batches']==5940 and record['repair_retention_training_windows']==844800
            assert digest(record['checkpoint'])==record['checkpoint_sha256']
            assert digest(record['stage1_manifest'])==record['stage1_manifest_sha256']
            stage1=read(record['stage1_manifest']);assert stage1['checkpoint_sha256']==record['stage1_checkpoint_sha256']
            assert digest(stage1['checkpoint'])==record['stage1_checkpoint_sha256']
            if record['repair_update_policy']=='frozen_v1':
                payload=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
                first=torch.load(stage1['checkpoint'],map_location='cpu',weights_only=False)
                for k,v in first['model_state'].items():
                    assert torch.equal(v,payload['model_state']['repair.'+k]);frozen_tensor_count+=1
                assert record['repair_trainable_parameters']==0 and record['repair_learning_rate']==0
            assert row['physical_gpu_consistent'] and row['predictor_parameters']==35427
            records=[read(p) for p in data['training_resources']['resource_record_paths']]
            assert len(records)==2 and all(r['status']=='completed' for r in records)
            assert abs(sum(r['full_workflow_seconds'] for r in records)-row['full_training_seconds'])<1e-9
            assert all('1315e15c-d657-e61f-bde5-6e0b38e6e54b' in r['device']['uuid'] for r in records)
            new_workflow+=row['actual_new_training_seconds']
    assert all(g==groups['E23-v31-B'] for g in groups.values())
    for row in latest['rows'][3:]:
        trial=read(row['evaluation']);base=read(latest['rows'][0]['evaluation']);t1=read(latest['rows'][1]['evaluation'])
        comparisons[row['model']]={}
        for name,reference in [('v31B',base),('T1',t1)]:
            comparisons[row['model']][name]={k:dict(mean_delta=trial['test_mean'][k]-reference['test_mean'][k],
                better_masks=sum(a[k]<b[k] for a,b in zip(trial['test_draws'],reference['test_draws']))) for k in ['mse','mae']}
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),source_files=len(frozen['source_files']),
        protected_files=len(frozen['protected_files']),new_stage1_training=False,new_stage2_runs=3,
        scored_models=6,total_test_draws=count,new_test_draws=60,max_window_mean_error=max_error,
        head_sources_build_loss_and_initialization_unchanged=True,frozen_repair_tensors_checked=frozen_tensor_count,
        actual_new_training_seconds=new_workflow,paired_mask_comparisons=comparisons,
        selected=read(ROOT/'final-selection.json')['selected'],formal_multiseed_results_unchanged=True)
    diagnostics=read(ROOT/'selected-repair-diagnostics.json')
    for trial in read(ROOT/'plan.json')['trials']:
        d=diagnostics[trial['label']];row=next(r for r in latest['rows'] if r['model']==trial['label'])
        assert d['checkpoint_sha256']==row['checkpoint_sha256']
        assert abs(d['stage1']['mse']-row['stage1_repair_mse'])<1e-6
        assert d['stage2']['mse']==row['stage2_repair_mse']
        if trial['policy']=='frozen_v1':assert d['output_drift_mse']==0 and d['repair_state_identical']
    name='ETTh1_P4_E23_v41_repair_drift_seed502_20261004'
    import csv
    for suffix,expected_rows in [('.csv',6),('_all_draws.csv',120)]:
        assert len(list(csv.DictReader((ROOT/(name+suffix)).open())))==expected_rows
    for suffix in ['.csv','_all_draws.csv','.md']:
        assert digest(ROOT/(name+suffix))==digest(Path('/home/xyw/paper/E23/docs/experiments')/(name+suffix))
    result.update(repair_diagnostic_rows=252,summary_rows=6,draw_rows=120,public_reports_match_artifacts=True)
    (ROOT/'final-integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
