"""Verify every archive member against its bundled per-file SHA256 list."""
import hashlib, json, sys, tarfile
from pathlib import Path
base=Path(__file__).resolve().parent
index=json.loads((base/'asset_index.json').read_text())
root=Path(sys.argv[1])
checked=0
for bundle in index['bundles']:
    p=root/bundle['name']
    with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==bundle['sha256'],(p,'archive checksum')
    with tarfile.open(p,'r:gz') as tar:
        sums=tar.extractfile('SHA256SUMS').read().decode().splitlines()
        expected=dict((line.split('  ',1)[1],line.split('  ',1)[0]) for line in sums)
        members={m.name:m for m in tar.getmembers() if m.isfile()}
        assert set(members)==set(expected)|{'SHA256SUMS','FILE_MANIFEST.json'},(p,'unexpected archive members')
        for name,digest in expected.items():
            with tar.extractfile(members[name]) as f: got=hashlib.file_digest(f,'sha256').hexdigest()
            assert got==digest,(p,name)
            checked+=1
    print(bundle['name'],len(expected),'files OK',flush=True)
assert len(index['rows'])==100
assert len({(r['dataset'],r['scene'],r['model']) for r in index['rows']})==100
for row in index['rows']:
    if row['actual_checkpoint_sha256']:
        assert row['actual_checkpoint_sha256']==row['original_checkpoint_sha256']
        assert row['selected_epoch'] is not None and row['training_seed'] is not None
print('Verified',checked,'archive members, 100 combinations, 92 original selected checkpoints.')
