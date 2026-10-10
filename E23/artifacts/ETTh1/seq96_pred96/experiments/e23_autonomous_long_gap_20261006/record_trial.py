"""Append completed validation-controlled trials to a single research ledger."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SERIES = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    audit = json.loads((root/'report-audit.json').read_text())
    assert audit['status'] == 'published'
    primary_csv = next(Path(p) for p in audit['reports'] if Path(p).suffix == '.csv' and not any(
        p.endswith(s) for s in ['_all_draws.csv','_paired.csv','_selections.csv']))
    assert hashlib.sha256(primary_csv.read_bytes()).hexdigest() == audit['reports'][str(primary_csv)]
    with primary_csv.open() as f:
        own = next(r for r in csv.DictReader(f) if r['model'] == args.version)
    entry = dict(version=args.version, root=str(root), results=own,
                 validation_upgrade_vs_parent=audit['validation_upgrade_vs_parent'],
                 retained_candidate=audit['retained_candidate'],
                 validation_goal_vs_T1=audit['validation_goal_vs_T1'],
                 development_goal_vs_T1=audit['development_goal_vs_T1'],
                 report=str(primary_csv.with_suffix('.md')), recorded_utc=datetime.now(timezone.utc).isoformat())
    state_path = SERIES/'series.json'
    state = json.loads(state_path.read_text())
    assert not any(t['version'] == args.version for t in state['trials']), 'duplicate trial'
    state['trials'].append(entry)
    if entry['validation_upgrade_vs_parent']:
        state['parent'] = args.version
    state['current_trial'] = dict(version=args.version, root=str(root), status='complete')
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    rows = [{**t['results'], 'validation_upgrade_vs_parent':t['validation_upgrade_vs_parent'],
             'validation_goal_vs_T1':t['validation_goal_vs_T1'], 'development_goal_vs_T1':t['development_goal_vs_T1'],
             'retained_candidate':t['retained_candidate'], 'experiment_root':t['root']} for t in state['trials']]
    path = DOCS/'ETTh1_E23_autonomous_long_gap_iterations_20261006.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in entry.items() if k != 'results'}))


if __name__ == '__main__':
    main()
