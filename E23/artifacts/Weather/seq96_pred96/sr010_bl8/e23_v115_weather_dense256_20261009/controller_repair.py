"""Repair-first sealed queue on the configured GPU; forecast is deliberately deferred."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write_status(state, **extra):
    record = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = ROOT/'status.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(ROOT/'status.json')
    print(json.dumps(record), flush=True)


def freeze_check():
    for path, expected in json.loads((ROOT/'source_freeze.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Sealed input changed: '+path)


def run(script, *args):
    freeze_check()
    suffix = '_'+args[-1] if script == 'evaluate_repair.py' else ''
    log = ROOT/(Path(script).stem+suffix+'.log')
    with log.open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/script), *args], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT)
        write_status('running', phase=script, selection=args[-1] if suffix else None,
                     pid=process.pid, log=str(log))
        checked = time.monotonic()
        while process.poll() is None:
            time.sleep(30)
            if time.monotonic()-checked >= 300:
                lines = log.read_text(errors='replace').splitlines()
                write_status('running', phase=script, pid=process.pid, log=str(log),
                             last_progress=lines[-1] if lines else None)
                checked = time.monotonic()
        if process.returncode:
            raise RuntimeError(f'{script}: exit {process.returncode}; inspect {log}')


def repair_report():
    import weather_common  # installs this candidate's sealed integration paths
    from integrations.forecast_compare.resources import selected_training_resources
    plan = json.loads((ROOT/'plan.json').read_text())
    label, scene = plan['models'][0]['label'], plan['scenarios'][0]['name']
    evaluations = [ROOT/label/scene/f'repair_evaluation_{criterion}/twenty-test-draws.json'
                   for criterion in ('mae','mse')]
    definitions = [(label, path, json.loads(path.read_text())) for path in evaluations]
    for baseline in plan['repair_baselines']:
        path = Path(baseline['evaluation'])
        definitions.append((baseline['label'], path, json.loads(path.read_text())))
    expected = [(r['seed'],r['mask_sha256'],r['windows'],r['hidden96_coordinates'],r['hidden32_coordinates'])
                for r in definitions[0][2]['draws']]
    rows = []
    for model, path, record in definitions:
        actual = [(r['seed'],r['mask_sha256'],r['windows'],r['hidden96_coordinates'],r['hidden32_coordinates'])
                  for r in record['draws']]
        assert actual == expected, ('Different test fixtures', model)
        if model == label:
            stage1_selection = record['selection_metric']
            resources = record['training_resources']
        else:
            manifests = [Path(p) for p in record['training_resources']['selection_manifest_paths']]
            manifest = manifests[-1]
            stage1_record = json.loads(manifest.read_text())
            stage1_selection = stage1_record.get('selection_metric',
                stage1_record.get('training', {}).get('selection_metric', stage1_record.get('selection', 'see_source')))
            if stage1_record.get('parent_selection'):
                manifests.insert(0, Path(stage1_record['parent_selection']))
            resources = selected_training_resources(manifests)
        assert resources['resource_status'] == 'measured_all_stages', model
        row = dict(model=model, stage1_selection=stage1_selection,
                   test_masks=len(actual), source=str(path))
        for key in ('repair96_mse','repair96_mae','repair32_mse','repair32_mae'):
            row[key] = record['summary'][key]['mean']
            row[key+'_mask_std_ddof1'] = record['summary'][key]['mask_std_ddof1']
        row.update(stage1_full_training_seconds=resources['full_workflow_seconds'],
                   stage1_peak_allocated_gib=resources['peak_allocated_gib'],
                   stage1_peak_reserved_gib=resources['peak_reserved_gib'],
                   stage1_resource_record_paths=json.dumps(resources['resource_record_paths']))
        rows.append(row)
    fields = list(rows[0])
    with (ROOT/'repair_comparison.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    evaluation = definitions[0][2]
    resources = evaluation['training_resources']
    cost = dict(full_stage1_training_seconds=resources['full_workflow_seconds'],
                peak_allocated_gib=resources['peak_allocated_gib'],peak_reserved_gib=resources['peak_reserved_gib'],
                resource_record_paths=resources['resource_record_paths'],
                scope='reused Weather-v79 coarse Stage1 plus new full refinement; includes cache/validation/save; excludes test; not one uninterrupted cold run',
                stage2_trained=False)
    (ROOT/'complete_stage1_training_cost.json').write_text(json.dumps(cost, indent=2))
    target = plan['repair_targets']
    checks = {key:evaluation['summary'][key]['mean'] < value for key,value in target.items()}
    (ROOT/'repair_goal_audit.json').write_text(json.dumps(dict(primary_selection='mae',
        passed_by_metric=checks,all_four_passed=all(checks.values()),targets=target,
        evidence=str(evaluations[0]),claim_scope='existing matched development baselines, not universal literature SOTA'),indent=2))


def main():
    freeze_check()
    plan=json.loads((ROOT/'plan.json').read_text())
    dependency=Path(plan['queue_after'])
    checked=-float('inf')
    while True:
        parent=json.loads((dependency/'status.json').read_text())
        if parent['status']=='failed':
            raise RuntimeError('Dependency failed: '+parent.get('error','unknown'))
        gpu=subprocess.check_output(['nvidia-smi','-i',plan['physical_gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if parent['status']=='completed' and not gpu:
            break
        if time.monotonic()-checked >= 300:
            write_status('queued',waiting_for=str(dependency),parent_status=parent,
                         next_stage='100 full Stage1 epochs, both selections, repair20; no automatic Stage2')
            checked=time.monotonic()
        time.sleep(30)
    assert json.loads((ROOT/'preflight.json').read_text())['zero_residual_reproduces_parent_exactly']
    run('train_refinement.py',plan['scenarios'][0]['name'])
    for criterion in ('mae','mse'):
        run('evaluate_repair.py','--selection',criterion)
    repair_report()
    write_status('completed',exit_code=0,results=str(ROOT/'repair_comparison.csv'),forecast_deferred=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write_status('failed',exit_code=1,error=repr(error))
        raise
