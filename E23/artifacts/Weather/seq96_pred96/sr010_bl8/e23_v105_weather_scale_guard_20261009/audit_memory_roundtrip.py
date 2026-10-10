"""Compare the serialized parent bank with the existing loader round trip."""
import json
from pathlib import Path
import numpy as np
import torch
from weather_common import ROOT, protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection, _new_memory, _memory_state_sha256

plan = json.loads((ROOT / 'plan.json').read_text())
protocol = protocol_for(plan, plan['scenarios'][0])
settings = load_e23_settings(Path(plan['parent_settings']), protocol=protocol,
                             allow_variable_scaling=True)
parent = load_stage1_selection(Path(plan['parent_selection']), protocol, settings,
                              device=torch.device('cpu'))
memory = _new_memory(protocol)
memory.load_state_dict(parent['memory_state'])
loaded = memory.state_dict()
differences = []

def compare(a, b, path='memory_state'):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype, path
        if not np.array_equal(a, b):
            differences.append(dict(path=path, kind='array', dtype=str(a.dtype),
                                    changed=int(np.count_nonzero(a != b)),
                                    max_absolute_change=float(np.max(np.abs(a - b)))))
    elif type(a) != type(b) or a != b:
        differences.append(dict(path=path, kind='scalar', before=repr(a), after=repr(b)))

compare(parent['memory_state'], loaded)
record = dict(parent_hash=parent['memory_sha256'], loaded_hash=_memory_state_sha256(loaded),
              difference_count=len(differences), differences=differences)
(ROOT / 'memory_roundtrip_audit.json').write_text(json.dumps(record, indent=2))
print(json.dumps(dict(parent_hash=record['parent_hash'], loaded_hash=record['loaded_hash'],
                     difference_count=len(differences), first_differences=differences[:12]), indent=2))
