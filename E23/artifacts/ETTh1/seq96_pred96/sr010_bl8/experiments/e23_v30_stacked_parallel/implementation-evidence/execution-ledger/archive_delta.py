import difflib, hashlib, json, sys
from pathlib import Path
root=Path('artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v30_stacked_parallel/implementation-evidence')
baseline=root/'baseline'
# Baseline copies live under files/.
copyroot=baseline/'files'
if not copyroot.exists(): copyroot=baseline
out=root/f'task-{sys.argv[1]}'
out.mkdir(exist_ok=True)
records={}
for name in sys.argv[2:]:
 p=Path(name); old=copyroot/p
 destination=out/p; destination.parent.mkdir(parents=True,exist_ok=True)
 destination.write_bytes(p.read_bytes())
 previous=old.read_text() if old.exists() else ''
 patch=''.join(difflib.unified_diff(previous.splitlines(True),p.read_text().splitlines(True),fromfile='baseline/'+name,tofile=name))
 destination.with_suffix(destination.suffix+'.patch').write_text(patch)
 records[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'hashes.json').write_text(json.dumps(records,indent=2)+'\n')
