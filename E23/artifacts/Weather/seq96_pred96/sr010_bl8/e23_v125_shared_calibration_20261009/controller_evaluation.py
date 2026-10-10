"""Evaluate the sealed Stage1 calibration once, with no forecast training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent


def status(state,**extra):
    record=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
    temporary=ROOT/'status.tmp';temporary.write_text(json.dumps(record,indent=2));temporary.replace(ROOT/'status.json')
    print(json.dumps(record),flush=True)


def main():
    for path,digest in json.loads((ROOT/'source_freeze.json').read_text())['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    plan=json.loads((ROOT/'plan.json').read_text());assert json.loads((ROOT/'preflight.json').read_text())['validation_alpha_locked']==.5
    with (ROOT/'evaluate_repair_mae.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'evaluate_repair.py'),'--selection','mae'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status('running',phase='evaluate_repair.py',pid=process.pid,selection='mae')
        checked=time.monotonic()
        while process.poll() is None:
            time.sleep(5)
            if time.monotonic()-checked>=300:
                status('running',phase='evaluate_repair.py',pid=process.pid,selection='mae');checked=time.monotonic()
        assert process.returncode==0,'inspect evaluate_repair_mae.log'
    entry,scene=plan['models'][0],plan['scenarios'][0]
    result_path=ROOT/entry['label']/scene['name']/'repair_evaluation_mae/twenty-test-draws.json'
    result=json.loads(result_path.read_text());resources=result['training_resources']
    values={k:result['summary'][k]['mean'] for k in plan['repair_targets']}
    checks={k:values[k]<threshold for k,threshold in plan['repair_targets'].items()}
    (ROOT/'repair_goal_audit.json').write_text(json.dumps(dict(primary_selection='mae',alpha_fixed_before_test=.5,values=values,targets_at_launch=plan['repair_targets'],passed_by_metric=checks,all_four_passed=all(checks.values()),claim_scope='matched development baselines; one training seed'),indent=2))
    (ROOT/'complete_stage1_training_cost.json').write_text(json.dumps(dict(full_stage1_seconds=resources['full_workflow_seconds'],full_stage1_minutes=resources['full_workflow_seconds']/60,peak_allocated_gib=resources['peak_allocated_gib'],peak_reserved_gib=resources['peak_reserved_gib'],resource_record_paths=resources['resource_record_paths'],scope='shared coarse training once plus both independent100-epoch calibration trainings plus measured validation calibration; no Stage2'),indent=2))
    status('completed',exit_code=0,results=str(result_path),forecast_deferred=True,metrics=values)


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',exit_code=1,error=repr(error));raise
