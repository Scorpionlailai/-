"""Rebuild reviewed repair tables from completed evaluations; never train/evaluate."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import json
import sys

BASE=Path(__file__).resolve().parent
WORK=Path('/home/xyw/.cache/codex-worktrees/e23-v115-dense-width256-20261009')
sys.path[:0]=[str(WORK),str(WORK/'src')]
from integrations.forecast_compare.resources import selected_training_resources
from repair_resource_scope import stage1_manifests

SCENES=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8')
RUNS=[('E23-v107','e23_v107_weather_context_gate_20261009'),('E23-v108','e23_v108_weather_mae_repair_20261009'),('E23-v115','e23_v115_weather_dense256_20261009'),('E23-v125','e23_v125_shared_calibration_20261009')]
METRICS=('repair96_mse','repair96_mae','repair32_mse','repair32_mae')


def main():
    definitions=[]
    protocol_plan=json.loads((SCENES/'e23_v115_weather_dense256_20261009'/'plan.json').read_text())
    for item in protocol_plan['repair_baselines']:
        if item['label'].startswith('E23-v107'):continue
        definitions.append((item['label'],'baseline',Path(item['evaluation'])))
    for label,run in RUNS:
        for criterion in ('mae','mse'):
            path=SCENES/run/label/'sr010_bl8'/f'repair_evaluation_{criterion}/twenty-test-draws.json'
            if path.exists():definitions.append((label,criterion,path))
    rows=[];expected=None
    for label,criterion,path in definitions:
        result=json.loads(path.read_text())
        fixtures=[(r['seed'],r['mask_sha256'],r['windows'],r['hidden96_coordinates'],r['hidden32_coordinates']) for r in result['draws']]
        if expected is None:expected=fixtures
        assert fixtures==expected and len(fixtures)==20
        selected,manifests=stage1_manifests(result);record=json.loads(selected.read_text())
        resources=selected_training_resources(manifests)
        assert resources['resource_status']=='measured_all_stages'
        assert all(json.loads(p.read_text())['stage']=='repair' for p in manifests)
        selection=record.get('selection_metric',record.get('training',{}).get('selection_metric',record.get('selection','see_source')))
        row=dict(model=label,selection=selection,primary=criterion in ('mae','baseline'),test_masks=20)
        for key in METRICS:
            row[key]=result['summary'][key]['mean'];row[key+'_mask_std_ddof1']=result['summary'][key]['mask_std_ddof1']
        row.update(stage1_full_seconds=resources['full_workflow_seconds'],stage1_full_minutes=resources['full_workflow_seconds']/60,
                   stage1_peak_allocated_gib=resources['peak_allocated_gib'],stage1_peak_reserved_gib=resources['peak_reserved_gib'],
                   source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stage1_selection_paths=json.dumps([str(p) for p in manifests]),stage1_resource_paths=json.dumps(resources['resource_record_paths']))
        rows.append(row)
    output=BASE/'weather_repair_reviewed_comparison.csv'
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    primary=[r for r in rows if r['primary']]
    minima={key:min(row[key] for row in primary) for key in METRICS}
    winners=[r['model'] for r in primary if all(r[key]<=minima[key] for key in METRICS)]
    all_completed=all(json.loads((SCENES/run/'status.json').read_text())['status']=='completed' for _,run in RUNS)
    progress=dict(updated_at=datetime.now(timezone.utc).isoformat(),scope='matched Weather sr010/bl8 seq96 repair development20 masks; primary validation MAE candidate selection; original baselines own Stage1 selection',metric_minima=minima,single_model_all_four_best=winners,baseline_report_scope='Stage1 only; includes coarse/refinement parents; forecast resources excluded',comparison=str(output),rows=len(rows),all_formal_jobs_completed=all_completed,goal_complete=bool(winners) and all_completed)
    (BASE/'weather_repair_goal_progress.json').write_text(json.dumps(progress,indent=2))
    print(json.dumps(progress));print(json.dumps([{k:r[k] for k in ['model','selection',*METRICS,'stage1_full_minutes']} for r in rows]))


if __name__=='__main__':main()
