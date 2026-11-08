"""Collect code by original source-seal SHA, explicitly retain unresolved versions."""
exec((__import__('pathlib').Path(__file__).parent/'build_transfer.py').read_text().split('rows, bundles, all_files = [], [], []')[0])
index = read(HERE/'asset_index.json')
required = {}
for row in index['rows']:
    selection=read(PAPER/row['selection_record'])
    for key in ('adapter_sha256','author_profiles_sha256'):
        if selection.get(key): required.setdefault(selection[key],set()).add('historical_T1/'+key)
    seal = row.get('source_seal') or {}
    for name, digest in seal.get('files',{}).items():
        required.setdefault(digest, set()).add(name)
    if not row['code_version']:
        registry = seal.get('registry',{}).get('baselines',[])
        row['code_version'] = next((r['commit'] for r in registry if r['model']==row['model']),None)
    if row['model']=='T1' and row['dataset']=='Weather':
        parent=Path('/home/xyw/paper')/row['configuration']
        seal=read(parent.parent/'source_seal.json')
        for f in seal.get('files',[]):
            name=f['path']
            if '/T1/' in name or '/E24/' in name or '/E23/' in name:
                required.setdefault(f['sha256'],set()).add(name)

pool = {}
search_roots = [PAPER/m for m in MODELS] + [PAPER/'E24/integrations',PAPER/'E24/src',PAPER/'E24/configs',PAPER/'E23/integrations',PAPER/'E23/src',PAPER/'E23/configs',PAPER/'E23/scripts',PAPER/'E24/scripts']
search_roots += list(ART.glob('*reproduction*/source_snapshots'))
search_roots += [Path('/home/xyw/r3_results_github_20261101')/m for m in MODELS]
search_roots += [Path('/home/xyw/r3_final_runs_20261030/snapshot/E24')]
for root in search_roots:
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('.git','__pycache__','node_modules')]
        for name in names:
            p=Path(directory)/name
            if p.suffix in ('.py','.yaml','.yml','.json','.toml','.md','.txt') and p.is_file() and p.stat().st_size<10_000_000:
                pool.setdefault(sha(p),p)
for names in required.values():
    for name in names:
        p=Path(name) if name.startswith('/') else PAPER/name
        if p.is_file(): pool.setdefault(sha(p),p)
records=[]
archive=PACKS/'source_code.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
    for digest,names in sorted(required.items()):
        p=pool.get(digest)
        arc='source_by_sha256/'+digest+'/'+Path(sorted(names)[0]).name
        records.append(dict(expected_sha256=digest,original_paths=sorted(names),found=p is not None,
                            archive_path=arc if p else None,recovered_from=str(p) if p else None))
        if p: tar.add(p,arcname=arc,recursive=False)
    # Include current adapters and model source as supplementary code, explicitly versioned.
    supplementary=[]
    for root in search_roots[:len(MODELS)+6]:
        for p in root.rglob('*.py'):
            if p.is_file() and '__pycache__' not in p.parts:
                arc='supplementary_current/'+str(p.relative_to(PAPER))
                tar.add(p,arcname=arc,recursive=False)
                supplementary.append(dict(path=arc,sha256=sha(p),original_path=str(p)))
    save(HERE/'manifests/source_code.json',dict(original_sealed_versions=records,supplementary_current_files=supplementary,
        note='Only source_by_sha256 entries are proven original sealed versions. Supplementary current code is not a replacement for missing historical versions.'))
    tar.add(HERE/'manifests/source_code.json',arcname='FILE_MANIFEST.json')
    checkpath=HERE/'manifests/source_code.SHA256SUMS'
    checkpath.write_text(''.join(f"{r['expected_sha256']}  {r['archive_path']}\n" for r in records if r['found'])+''.join(f"{r['sha256']}  {r['path']}\n" for r in supplementary))
    tar.add(checkpath,arcname='SHA256SUMS')
for row in index['rows']:
    seal=(row.get('source_seal') or {}).get('files',{})
    row['missing_original_code']=[dict(path=n,sha256=h) for n,h in seal.items() if h not in pool]
    selection=read(PAPER/row['selection_record'])
    row['historical_adapter_checks']=[dict(kind=k,sha256=selection[k],found=selection[k] in pool) for k in ('adapter_sha256','author_profiles_sha256') if selection.get(k)]
    row['source_code_archive']='source_code.tar.gz'
    row['all_assets_complete']=row['primary_assets_complete'] and not row['missing_original_code'] and all(v['found'] for v in row['historical_adapter_checks'])
    row['necessary_assets_complete']=row['primary_assets_complete'] and not any(not c['path'].endswith('/parallel_queue.py') for c in row['missing_original_code']) and all(v['found'] for v in row['historical_adapter_checks'])
index['bundles']=[b for b in index['bundles'] if b['name']!='source_code.tar.gz']
index['bundles'].append(dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),download_url=URL+archive.name))
save(HERE/'asset_index.json',index)
print('Original sealed code versions:',sum(r['found'] for r in records),'/',len(records))
print('Missing:',[(r['original_paths']) for r in records if not r['found']])
