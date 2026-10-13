"""Integration regression: GPU query failure must not erase experiment progress."""
import json
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parent.parent
result = subprocess.run([sys.executable, str(root / 'session_snapshot.py')], capture_output=True, text=True)
assert result.returncode == 0, 'progress snapshot aborted on GPU monitoring error: ' + result.stderr
snapshot = json.loads(result.stdout)
assert snapshot['completed_repairs'] >= 24
assert len(snapshot['active']) == 2
assert all(item['completed_masks'] is not None for item in snapshot['active'])
if snapshot.get('gpu_error'):
    assert snapshot['gpu'] == []
    assert 'Driver/library version mismatch' in snapshot['gpu_error']
print(json.dumps(dict(passed=True, snapshot=snapshot)))
