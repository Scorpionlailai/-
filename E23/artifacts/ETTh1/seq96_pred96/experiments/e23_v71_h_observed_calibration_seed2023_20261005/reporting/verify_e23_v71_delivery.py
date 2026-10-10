from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v71_h_observed_calibration_seed2023_20261005')
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
PREFIX = DOCS / 'ETTh1_E23_v71_H_observed_calibration_seed2023_20261005'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    status = read(ROOT / 'status.json')
    assert status['status'] == 'complete' and status['exit_code'] == 0
    complete = read(ROOT / 'completed-audit.json')
    assert complete['status'] == 'complete'
    assert complete['new_stage1_runs'] == complete['new_stage2_runs'] == 1
    assert complete['new_T1_training_runs'] == 0
    assert complete['test_used_for_selection'] is False
    plan = read(ROOT / 'plan.json')
    frozen = read(ROOT / 'frozen-definition.json')
    assert sha(ROOT / 'plan.json') == frozen['plan_sha256']
    locks = {str(Path(plan['work']) / p): h for p, h in frozen['sources'].items()}
    locks.update(frozen['protected'])
    locks.update(read(ROOT / 'reference-locks.json'))
    locks.update({str(ROOT / p): h for p, h in frozen['controller'].items()})
    locks[str(ROOT / 'plan.json')] = frozen['plan_sha256']
    for path, digest in locks.items():
        assert sha(path) == digest, path
    counts = {'.csv': 3, '_all_draws.csv': 60, '_paired.csv': 40, '_selections.csv': 2,
              '_repair_diagnostics.csv': 2}
    tables = {}
    for suffix, count in counts.items():
        path = Path(str(PREFIX) + suffix)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == count, (path, len(rows))
        tables[suffix] = rows
    audit = read(ROOT / 'report-audit.json')
    assert audit['all_scenes_complete'] is True
    assert audit['csv_sha256'] == sha(PREFIX.with_suffix('.csv'))
    assert len(audit['repair_tensors_bitwise_frozen']) == 2
    assert set(audit['repair_tensors_bitwise_frozen'].values()) == {214}
    diagnostic = read(ROOT / 'diagnostic-report-audit.json')
    assert diagnostic['coverage_verified_against_actual_masks'] is True
    assert diagnostic['csv_sha256'] == sha(str(PREFIX) + '_repair_diagnostics.csv')
    assert diagnostic['report_sha256'] == sha(PREFIX.with_suffix('.md'))
    index_path = DOCS / 'E23_v71_h_observed_calibration_candidates_20261005.json'
    index = read(index_path)
    assert len(index['candidates']) == 1 and index['main_source_integrated'] is False
    actual_new_seconds = sum(float(r['full_training_seconds']) for r in tables['.csv'] if r['model'] == 'E23-v71')
    assert abs(actual_new_seconds - index['measured_new_training_seconds']) < 1e-6
    for candidate in index['candidates']:
        assert sha(candidate['checkpoint']) == candidate['checkpoint_sha256']
        selection = read(candidate['selection'])
        assert selection['selection_metric'] == 'mae'
        assert selection['epochs_completed'] == 100 and selection['repair_trainable_parameters'] == 0
        assert selection['predictor_initial_sha256'] == plan['predictor_initial_sha256']
        row = next(r for r in tables['.csv'] if r['scenario'] == candidate['scenario'] and r['model'] == 'E23-v71')
        assert row['checkpoint_sha256'] == candidate['checkpoint_sha256']
        assert float(row['mse']) == candidate['scores']['mse']
        assert float(row['mae']) == candidate['scores']['mae']
    files = [Path(str(PREFIX) + suffix) for suffix in counts] + [PREFIX.with_suffix('.md'), index_path]
    for row in tables['_selections.csv']:
        selection_path = Path(row['selection'])
        selection = read(selection_path)
        checkpoint = Path(selection['checkpoint'])
        lock_path = selection_path.parent.parent.parent / 'validation-lock-before-tests.json'
        assert sha(checkpoint) == selection['checkpoint_sha256'] == row['checkpoint_sha256']
        assert read(lock_path)['checkpoint_sha256'] == row['checkpoint_sha256']
        files += [selection_path, checkpoint, lock_path]
    for candidate in index['candidates']:
        stage1_selection = Path(candidate['stage1_selection'])
        record = read(stage1_selection)
        assert sha(record['checkpoint']) == record['checkpoint_sha256']
        files += [stage1_selection, Path(record['checkpoint'])]
    files += [ROOT / p for p in ['status.json', 'completed-audit.json', 'report-audit.json',
                                  'diagnostic-report-audit.json', 'test-audit.json', 'preflight.json',
                                  'implementation-review.md', 'final-review.md', 'final-focused-tests.log',
                                  'validation-coverage-counts.json', 'frozen-definition.json',
                                  'reference-locks.json', 'frozen-source.tar.gz']]
    files += [Path('/tmp') / p for p in ['publish_e23_v71.py', 'publish_e23_v71_diagnostics.py',
                                        'e23_v71_coverage_counts.py', 'verify_e23_v71_delivery.py']]
    for name in ['publish_e23_v71.py', 'publish_e23_v71_diagnostics.py',
                 'e23_v71_coverage_counts.py', 'verify_e23_v71_delivery.py']:
        archived = ROOT / 'reporting' / name
        assert sha(archived) == sha(Path('/tmp') / name)
        files.append(archived)
    result = dict(status='verified', verified_utc=datetime.now(timezone.utc).isoformat(),
                  unique_pretraining_locks_checked=len(locks), rows=counts,
                  stage2_bitwise_frozen_repair_tensors=audit['repair_tensors_bitwise_frozen'],
                  new_stage1_runs=1, new_stage2_runs=1, new_T1_training_runs=0,
                  all_reports_present=True, main_source_integrated=False,
                  artifacts={str(p): sha(p) for p in files})
    (ROOT / 'delivery-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}))


if __name__ == '__main__':
    main()
