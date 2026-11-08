"""Read-only collection of original selected repair assets; never trains or selects."""
import hashlib, json, os, subprocess, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = Path('/home/xyw/paper')
ART = PAPER / 'E24/artifacts'
PACKS = Path('/home/xyw/ogrr_transfer_packages_20261108')
PACKS.mkdir(exist_ok=True)
SCENES = ['sr010_bl8', 'sr015_bl8', 'sr020_bl8', 'sr010_bl16']
MODELS = ['BRITS', 'CSDI', 'ImputeFormer', 'TimesNet', 'T1']
GROUPS = dict(ETTh1='repair_reproduction_20261010', Weather='repair_reproduction_20261010', BeijingAirQuality='beijing_reproduction_20261013', PAMAP2='pamap2_reproduction_20261014', PhysioNet2012='physionet_reproduction_20261015')
TAG = 'ogrr-baseline-transfer-20261108'
URL = 'https://github.com/Scorpionlailai/-/releases/download/' + TAG + '/'
cache = {}
def sha(p):
    p = Path(p)
    if p not in cache:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
        cache[p] = h.hexdigest()
    return cache[p]
def read(p):
    return json.loads(Path(p).read_text()) if Path(p).is_file() else {}
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

rows, bundles, all_files = [], [], []
for dataset, group in GROUPS.items():
    files = {}
    def add(p):
        p = Path(p)
        if not p.is_file(): return None
        # Preserve original repo-relative layout inside each archive.
        rel = str(p.relative_to(PAPER))
        files[rel] = p
        return rel
    def metadata_tree(p):
        if Path(p).is_dir():
            for f in Path(p).rglob('*'):
                if f.is_file() and f.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.csv', '.md'):
                    add(f)
    root = ART/group
    for f in root.iterdir():
        if f.is_file() and f.suffix in ('.json', '.csv', '.md', '.yaml'): add(f)
    metadata_tree(root/'comparisons')
    for scene in SCENES:
        bank = root/'shared_assets'/dataset/scene
        for f in bank.glob('*'):
            if f.is_file() and f.name != 'asset.lock': add(f)
        for model in MODELS:
            run = root/dataset/scene/model/'seed_2023'
            selection_path = run/'stage1/selection.json'
            config = run/'plan.json'
            result = run/'evaluation/repair_summary.json'
            checkpoint = run/'stage1/selected.pt'
            legacy = model == 'T1' and dataset in ('ETTh1', 'Weather')
            extra = []
            if legacy and dataset == 'ETTh1':
                workflow = PAPER/'E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v125_fixed_four_scene_transfer_20261009'/scene/'workflow.json'
                scenario = read(workflow)['scenario']
                selection_path = Path(scenario['t1_repair_selection'])
                run = selection_path.parent.parent
                config = workflow
                checkpoint = Path(read(selection_path)['checkpoint'])
                bank = Path(scenario['test_bank']).parent
                result = PAPER/'E23/artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_mean_and_dense_20261008/T1'/scene/'repair-audit.json'
                extra = [Path(scenario['scenario_manifest'])] if scenario.get('scenario_manifest') else []
            elif legacy:
                suffix = {'sr010_bl8':'','sr015_bl8':'_sr015','sr020_bl8':'_sr020','sr010_bl16':'_sr010_bl16'}[scene]
                parent = ART/'Weather/seq96_pred96'/scene/('weather_author_comparison'+suffix+'_20261008')
                run = parent/'T1'
                config = parent/'plan.json'
                selection_path = run/'stage1/selection.json'
                checkpoint = Path(read(selection_path)['checkpoint'])
                bank = parent.parent/'shared_protocol_v1'
                result = run/'evaluation/mae/twenty-test-draws.json'
                extra = [parent/'source_seal.json', Path(read(config)['profiles'])]
            selection = read(selection_path)
            plan = read(config)
            missing = []
            for key, p in [('checkpoint',checkpoint),('configuration',config),('selection',selection_path),('original_result',result)]:
                if not p.is_file(): missing.append({'kind':key,'original_path':str(p)})
                else: add(p)
            metadata_tree(run/'stage1')
            metadata_tree(run/'training')
            metadata_tree(run/'evaluation')
            for p in extra: add(p)
            metadata_tree(bank)
            for p in bank.rglob('*.npz'): add(p)
            tests = sorted(p for p in bank.rglob('*.npz') if 'test' in p.name)
            vals = sorted(p for p in bank.rglob('*.npz') if 'val' in p.name)
            if len(tests) != 20: missing.append({'kind':'20_test_masks','found':len(tests),'original_directory':str(bank)})
            if len(vals) != 1: missing.append({'kind':'validation_mask','found':len(vals),'original_directory':str(bank)})
            expected = selection.get('checkpoint_sha256')
            actual = sha(checkpoint) if checkpoint.is_file() else None
            if actual and expected and actual != expected:
                raise RuntimeError('Selected checkpoint hash mismatch: '+str(checkpoint))
            row = dict(dataset=dataset, scene=scene, model=model,
                training_seed=selection.get('seed', selection.get('training_seed', selection.get('profile',{}).get('seed'))),
                selected_epoch=selection.get('selected_epoch',selection.get('best_epoch')),
                original_checkpoint=str(checkpoint), checkpoint_archive_path=add(checkpoint),
                original_checkpoint_sha256=expected, actual_checkpoint_sha256=actual,
                configuration=add(config), selection_record=add(selection_path), original_result_source=add(result),
                source_seal=plan.get('source_seal'), original_profile=selection.get('profile'),
                code_version=selection.get('upstream_revision', selection.get('profile',{}).get('revision')),
                protocol='historical_T1_original_bank' if legacy else 'original_reproduction_shared_bank',
                mask_directory=str(bank), test_masks=[add(p) for p in tests],validation_masks=[add(p) for p in vals],
                missing=missing, primary_assets_complete=not missing,
                archive=dataset+'.tar.gz', download_url=URL+dataset+'.tar.gz')
            rows.append(row)
    records = [dict(path=rel,original_path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for rel,p in sorted(files.items())]
    manifest = dict(dataset=dataset,files=records)
    save(HERE/'manifests'/ (dataset+'.json'),manifest)
    checks = ''.join(f"{r['sha256']}  {r['path']}\n" for r in records)
    checkpath=HERE/'manifests'/(dataset+'.SHA256SUMS');checkpath.write_text(checks)
    archive=PACKS/(dataset+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for rel,p in sorted(files.items()): tar.add(p,arcname=rel,recursive=False)
        tar.add(checkpath,arcname='SHA256SUMS')
        tar.add(HERE/'manifests'/(dataset+'.json'),arcname='FILE_MANIFEST.json')
    bundles.append(dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),download_url=URL+archive.name))
    all_files.extend(records)
    print(dataset,'checkpoint',sum(r['actual_checkpoint_sha256'] is not None for r in rows if r['dataset']==dataset),'primary complete',sum(r['primary_assets_complete'] for r in rows if r['dataset']==dataset),'archive bytes',archive.stat().st_size,flush=True)
save(HERE/'asset_index.json',dict(branch='assets/ogrr-baseline-transfer-20261108',release_tag=TAG,
    source_repository_commit=subprocess.check_output(['git','-C',str(PAPER),'rev-parse','HEAD'],text=True).strip(),
    delivery_base_commit=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip(),
    scope='100 dataset/scene/model combinations; original selections only; Exchange-Rate deferred',
    completeness_definition='checkpoint hash matches original selection; configuration, selection, original result, one validation and 20 test masks exist. Historical source-code availability audited separately.',
    rows=rows,bundles=bundles))
