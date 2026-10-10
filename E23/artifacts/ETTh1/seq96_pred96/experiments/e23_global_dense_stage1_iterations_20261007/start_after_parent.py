"""Continue Stage1 architecture trials when the current queue cannot meet the goal."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    plan = json.loads((ROOT / 'queue-plan.json').read_text())
    parent = Path(plan['parent_queue'])
    print(json.dumps(dict(status='waiting_for_current_queue', parent=str(parent))), flush=True)
    while True:
        state = json.loads((parent / 'queue-state.json').read_text())
        if state['status'] != 'running':
            break
        time.sleep(300)
    if state['status'] == 'candidate_passed_development':
        (ROOT / 'deferred-status.json').write_text(json.dumps(dict(
            status='skipped_parent_already_passed_development', candidate=state['candidate']), indent=2) + '\n')
        sys.exit(0)
    if state['status'] != 'preset_trials_completed_need_next_design':
        (ROOT / 'deferred-status.json').write_text(json.dumps(dict(
            status='parent_error_requires_fix', parent_state=state), indent=2) + '\n')
        sys.exit(1)
    assert json.loads((ROOT / 'prelaunch-verification.json').read_text())['status'] == 'passed'
    with (ROOT / 'queue.log').open('x') as handle:
        result = subprocess.run([sys.executable, str(ROOT / 'queue.py')],
                                stdout=handle, stderr=subprocess.STDOUT)
    sys.exit(result.returncode)
