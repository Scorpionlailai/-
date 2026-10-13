"""Verify every completed forecast retained the selected repair checkpoint."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010')

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

cases = []
for folder in sorted(ROOT.glob('*/*/*/seed_2023')):
    stage1 = json.loads((folder / 'stage1/selection.json').read_text())
    stage2 = json.loads((folder / 'stage2/selection.json').read_text())
    audit = json.loads((folder / 'stage2/cache_audit.json').read_text())
    records = audit['records']
    tests = {key: value for key, value in records.items() if key.startswith('test_')}
    cases.append(dict(run=str(folder.relative_to(ROOT)),
        checkpoint_parent_match=sha(folder / 'stage1/selected.pt') == stage1['checkpoint_sha256'] == stage2['parent_stage1_sha256'],
        head_checkpoints_match=all(sha(folder / f'stage2/selected_{key}.pt') == digest for key, digest in stage2['checkpoints'].items()),
        cache_bindings_match=all(value['binding']['checkpoint_sha256'] == stage1['checkpoint_sha256'] and
                                value['binding']['state_sha256'] == stage2['repair_state_sha256'] for value in records.values()),
        test_cache_groups=len(tests),
        stage2_complete=json.loads((folder / 'status.json').read_text()).get('phase') == 'stage2_complete'))
failed = [case for case in cases if case['test_cache_groups'] != 20 or
          not all(case[key] for key in ('checkpoint_parent_match', 'head_checkpoints_match', 'cache_bindings_match', 'stage2_complete'))]
result = dict(passed=len(cases) == 32 and not failed, cases=cases)
(ROOT / 'diagnostics/checkpoint_binding_final_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(passed=result['passed'], cases=len(cases), failed=failed)))
assert result['passed']
