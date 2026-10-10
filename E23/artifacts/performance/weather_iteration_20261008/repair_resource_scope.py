"""Select only Stage1 resource chains from repair or forecast evaluations."""
from pathlib import Path
import json


def stage1_manifests(evaluation):
    candidates=[Path(p) for p in evaluation['training_resources']['selection_manifest_paths']]
    repair=[p for p in candidates if json.loads(p.read_text()).get('stage')=='repair']
    if not repair:
        raise ValueError('No Stage1 repair selection manifest in evaluation')
    selected=repair[-1]
    paths=[];active=set()
    def visit(path):
        if path in active:
            raise ValueError('Cyclic Stage1 repair parent chain')
        if path in paths:return
        active.add(path);record=json.loads(path.read_text())
        if record.get('stage')!='repair':
            raise ValueError('Stage1 repair chain references another stage')
        for parent in record.get('additional_parent_selections',[]):visit(Path(parent))
        if record.get('parent_selection'):visit(Path(record['parent_selection']))
        paths.append(path);active.remove(path)
    visit(selected)
    return selected,paths
