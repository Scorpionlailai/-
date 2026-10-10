"""One-time reversible directory migration; saved experiment contents stay intact."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, os, shutil, subprocess, tarfile

PAPER = Path('/home/xyw/paper')
OLD = PAPER / 'E23'
NEW = PAPER / 'E24'
ARCHIVE = OLD / 'archive'
RUN = OLD / 'artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007'
GIT = '/home/xyw/miniforge3/envs/malr/bin/git'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def tree(path):
    records = {}
    for folder, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__', '.pytest_cache') and not (Path(folder)/d).is_symlink())
        for name in sorted(files):
            p = Path(folder)/name
            if not p.is_symlink() and p.suffix not in ('.pyc', '.pyo'):
                records[str(p.relative_to(path))] = sha(p)
    return records

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')

assert not NEW.exists(), 'E24 already exists; refusing to overwrite'
assert json.loads((RUN/'verification.json').read_text())['status'] == 'verified'
assert json.loads((OLD/'artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_continuous_iterations_20261007/queue-state.json').read_text())['status'] == 'paused_by_user'
frozen = json.loads((RUN/'frozen-definition.json').read_text())
work = Path(json.loads((RUN/'plan.json').read_text())['work'])
assert all(sha(work/rel) == digest for rel,digest in frozen['sources'].items())
versions = sorted(p for p in PAPER.glob('.e23*') if p.is_dir() and not p.is_symlink())
aux = PAPER/'.t1-frozen-stage2-20261007'
if aux.is_dir(): versions.append(aux)
base_names = ['BRITS','BiTGraph','CRIB','ChannelTokenFormer','CoIFNet','T1']
assert all(not (PAPER/name).exists() for name in base_names)
version_plan = [dict(old=str(p), new=str(OLD/'versions'/p.name.lstrip('.')), content_hashes=tree(p)) for p in versions]
baseline_plan = []
for name in base_names:
    p = OLD/'artifacts/upstream'/name
    baseline_plan.append(dict(name=name,old=str(p),new=str(PAPER/name),revision=subprocess.check_output([GIT,'-C',str(p),'rev-parse','HEAD'],text=True).strip(),content_hashes=tree(p)))
write(ARCHIVE/'directory_migration_plan_20261007.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),versions=version_plan,baselines=baseline_plan))
# A recovery bundle is made before moving any source directories.
bundle = ARCHIVE/'E23-v79_sealed_20261007.tar.gz'
with tarfile.open(bundle, 'x:gz', dereference=False) as t:
    t.add(RUN, arcname='experiment/e23_v79', filter=lambda i: None if '__pycache__' in Path(i.name).parts else i)
    for p in sorted((OLD/'docs/experiments').glob('*20261007*')):
        if p.is_file() and ('v79' in p.name or 'v69_T1_pureMSE' in p.name): t.add(p,arcname='reports/'+p.name)
    for name in ['t1-v79-repair-comparison.json','t1-v79-repair-low-scene-readonly.json','clean-reference-verification.json','v79-clean-gap.csv']:
        p=RUN.parent/'e23_v79_toward_clean_20261007'/name
        if p.exists(): t.add(p,arcname='diagnostics/'+name)
    p=RUN.parent/'e23_four_scene_confirmation_reserved_20261007'
    if p.exists(): t.add(p,arcname='confirmation20',filter=lambda i: None if '__pycache__' in Path(i.name).parts else i)
# Make E24 from the exact source archive used for v79, excluding historical drivers/configurations.
NEW.mkdir()
keep_scripts={'run_author_aligned_p4.py','run_author_s4m_p4.py','run_saits_p4.py','run_t1_p4.py','run_crib_p4.py','run_coifnet_p4.py','run_bitgraph_p4.py','run_ctf_p4.py','evaluate_e23_t1_replicate.py','report_author_p4_comparison.py'}
with tarfile.open(RUN/'frozen-source.tar.gz') as t:
    for member in t.getmembers():
        rel=Path(member.name)
        assert not rel.is_absolute() and '..' not in rel.parts
        keep=rel.parts[0] in ('src','integrations') or (rel.parts[0]=='scripts' and rel.name in keep_scripts)
        if member.isfile() and keep:
            target=NEW/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as source,target.open('xb') as out: shutil.copyfileobj(source,out)
for directory in ['configs/baselines','docs/experiments','artifacts','migration','tests']:(NEW/directory).mkdir(parents=True,exist_ok=True)
for name in ['pyproject.toml','THIRD_PARTY_NOTICES.md']:shutil.copy2(OLD/name,NEW/name)
shutil.copy2(work/'configs/e23_v79_v69_global_level_seed2023.yaml',NEW/'configs/e24_v79.yaml')
shutil.copy2(OLD/'configs/stage1_seq96_dynamic_temporal_repair.yaml',NEW/'configs/etth1_seq96_pred96.yaml')
profile=json.loads((work/'configs/baselines/etth1_p4_author_20261002.json').read_text())
for name,cfg in profile['models'].items(): cfg['root']='../'+name
write(NEW/'configs/baselines/etth1_p4_author_20261002.json',profile)
(NEW/'data').symlink_to('../data')
# Retain the trusted data/mask catalog and add the byte-identical E24 protocol alias.
catalog=OLD.parent/'E22/migration/paper_catalog.jsonl'
text=catalog.read_text()
proto=NEW/'configs/etth1_seq96_pred96.yaml'
entry=dict(source=str(OLD/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),destination_relative=str(proto.relative_to(PAPER)),size=proto.stat().st_size,sha256=sha(proto),owner='E24')
(NEW/'migration/paper_catalog.jsonl').write_text(text.rstrip()+'\n'+json.dumps(entry)+'\n')
# This is a path adapter only: the actual model and training functions remain byte-identical.
p=NEW/'integrations/e23/config.py';s=p.read_text()
a='if not config.is_relative_to(root / "E23"):'
b='if not any(config.is_relative_to(root / name) for name in ("E23", "E24")):'
assert s.count(a)==1;s=s.replace(a,b).replace('E23 P4 protocol config must come from E23','inherited P4 protocol config must come from E23 or E24');p.write_text(s)
# Move baselines physically; old archive lookups retain relative compatibility links.
for row in baseline_plan:
    old,new=Path(row['old']),Path(row['new']);old.rename(new)
    old.symlink_to(os.path.relpath(new,old.parent),target_is_directory=True)
    assert tree(new)==row['content_hashes']
# Consolidate the old source directories. No hidden source aliases are left at paper root.
(OLD/'versions').mkdir()
for row in version_plan:
    old,new=Path(row['old']),Path(row['new']);old.rename(new)
    if (new/'E23/.git').is_file(): subprocess.run([GIT,'-C',str(OLD),'worktree','repair',str(new/'E23')],check=True,capture_output=True)
    assert tree(new)==row['content_hashes']
# Repair the two links between historical versions using the longest relocated prefix.
path_map={r['old']:r['new'] for r in version_plan}
for row in version_plan:
    for folder,dirs,files in os.walk(row['new'],followlinks=False):
        dirs[:]=[d for d in dirs if d!='.git']
        for name in dirs+files:
            p=Path(folder)/name
            if p.is_symlink():
                target=os.readlink(p)
                for old,new in sorted(path_map.items(),key=lambda x:-len(x[0])):
                    if target==old or target.startswith(old+'/'):
                        p.unlink();p.symlink_to(new+target[len(old):],target_is_directory=True);break
for row in baseline_plan:path_map[row['old']]=row['new']
write(ARCHIVE/'legacy_path_map_20261007.json',path_map)
with (OLD/'versions/index.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['version_directory','old_directory','new_directory','source_directory']);w.writeheader()
    for row in version_plan:w.writerow(dict(version_directory=Path(row['new']).name,old_directory=row['old'],new_directory=row['new'],source_directory=str(Path(row['new'])/'E23')))
manifest=dict(status='sealed',created_utc=datetime.now(timezone.utc).isoformat(),sealed_model='E23-v79',e24_parent='E23-v79',
 source_archive=str(RUN/'frozen-source.tar.gz'),source_archive_sha256=sha(RUN/'frozen-source.tar.gz'),
 recovery_bundle=str(bundle),recovery_bundle_sha256=sha(bundle),
 archived_source=str(OLD/'versions/e23-v79-global-level-four/E23'),source_hashes=frozen['sources'],
 e24_root=str(NEW),e24_adapter_changes=['integrations/e23/config.py: allow E24 protocol path'],
 version_directories_moved=len(version_plan),baseline_repositories_moved=len(baseline_plan),training_started=False,
 baseline_revisions={r['name']:r['revision'] for r in baseline_plan})
manifest['checkpoints']={}
for p in RUN.glob('*/E23-v79/*/runs/*.json'):
    record=json.loads(p.read_text())
    if 'checkpoint' not in record: continue
    checkpoint=Path(record['checkpoint'])
    assert sha(checkpoint)==record['checkpoint_sha256']
    manifest['checkpoints'][str(p)]=dict(manifest_sha256=sha(p),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
write(ARCHIVE/'E23-v79_seal_manifest_20261007.json',manifest)
shutil.copy2(ARCHIVE/'E23-v79_seal_manifest_20261007.json',NEW/'migration/e23_v79_parent.json')
(NEW/'artifacts/parent_e23_v79').symlink_to(os.path.relpath(RUN,NEW/'artifacts'),target_is_directory=True)
for p in (OLD/'docs/experiments').glob('ETTh1_E23_v79_T1_forecast_full_repair_last32_pause_20261007.*'):shutil.copy2(p,NEW/'docs/experiments'/p.name)
print(json.dumps(dict(status='organized',e24=str(NEW),versions_moved=len(version_plan),baselines_moved=len(baseline_plan),sealed_bundle=str(bundle),checkpoints_verified=len(manifest['checkpoints'])),ensure_ascii=False))
