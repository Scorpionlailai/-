"""Finish resource reports only; preserve all sealed sources and trained weights."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/xyw/paper/E23/artifacts/performance/weather_iteration_20261008')
from resource_summary import combine_stage_costs


def main():
    plan=json.loads((ROOT/'plan.json').read_text())
    for path,expected in json.loads((ROOT/'source_freeze.json').read_text())['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    archive=ROOT/'failed_post_training_report_20261009'
    archive.mkdir()
    shutil.copy2(ROOT/'status.json',archive/'status.json')
    assert json.loads((ROOT/'status.json').read_text())['status']=='failed'
    directories=[Path(plan['coarse_parent_selection']).parents[1],
                 Path(plan['parent_selection']).parents[1],
                 ROOT/'E23-v105/sr010_bl8/stage1',ROOT/'E23-v105/sr010_bl8/stage2']
    stages=[];records=[]
    for directory in directories:
        entries=[(p,json.loads(p.read_text())) for p in (directory/'resources').glob('*.json')]
        entries=[(p,r) for p,r in entries if r['status']=='completed']
        assert len(entries)==1,directory
        path,record=entries[0];records.append(record)
        stages.append(dict(directory=str(directory),resource=str(path),seconds=record['full_workflow_seconds'],
                     allocated_gib=None if record['peak_allocated_bytes'] is None else record['peak_allocated_bytes']/2**30,
                     reserved_gib=None if record['peak_reserved_bytes'] is None else record['peak_reserved_bytes']/2**30,
                     device=record['device']))
    result=dict(stages=stages,**combine_stage_costs(records),training_seed=2023,
                scope='reused coarse/refinement plus new CPU projection verification plus frozen Stage2; excludes test and search; not one uninterrupted cold run',
                cpu_gpu_memory_handling='CPU stages retain null memory; measured GPU stages define GPU peak',
                baseline_control_stage2_is_search_cost_not_production_cost=True)
    (ROOT/'complete_training_cost.json').write_text(json.dumps(result,indent=2))
    rows=[]
    for model,base in [('E23-v97',Path(plan['parent_experiment'])),('E23-v105',ROOT)]:
        cost=json.loads((base/'complete_training_cost.json').read_text())
        for criterion in ('mae','mse'):
            path=base/f'{model}/sr010_bl8/evaluation_{criterion}/twenty-test-draws.json'
            evaluation=json.loads(path.read_text())
            assert len(evaluation['draws'])==20 and evaluation['repair_frozen_verified']
            selected=json.loads(Path(evaluation['selection_manifest']).read_text())
            assert hashlib.sha256(Path(selected['checkpoint']).read_bytes()).hexdigest()==selected['checkpoint_sha256']
            row=dict(model=model,stage2_selection=criterion,test_groups=20,
                     full_training_seconds=cost['full_training_seconds'],peak_allocated_gib=cost['peak_allocated_gib'],
                     peak_reserved_gib=cost['peak_reserved_gib'],repair_frozen_verified=True)
            for key in ('forecast_mse','forecast_mae','repair96_mse','repair96_mae','repair32_mse','repair32_mae'):
                row[key]=evaluation['summary'][key]['mean']
                row[key+'_mask_std_ddof1']=evaluation['summary'][key]['mask_std_ddof1']
            rows.append(row)
    with (ROOT/'v97_v105_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    record=dict(status='completed',updated_at=datetime.now(timezone.utc).isoformat(),exit_code=0,
                comparison=str(ROOT/'v97_v105_comparison.csv'),report_recovery=str(Path(__file__)),
                training_or_evaluation_rerun=False,original_failure_preserved=str(archive))
    temporary=ROOT/'status.tmp';temporary.write_text(json.dumps(record,indent=2));temporary.replace(ROOT/'status.json')
    print(json.dumps(record))


if __name__=='__main__':main()
