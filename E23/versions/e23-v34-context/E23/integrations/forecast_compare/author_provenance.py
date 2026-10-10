"""Bind author-profile experiments to exact source, configuration and P4 data."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

from malr.evaluation.report import file_sha256
from .author_profiles import ROOT


def write_new_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def author_source_identity(profile_path, cfg, protocol):
    root = (ROOT / cfg['root']).resolve()
    git = '/home/xyw/miniforge3/envs/malr/bin/git'
    revision = subprocess.check_output([git, '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['revision']:
        raise ValueError(f'upstream revision mismatch: {root}')
    tracked = subprocess.check_output([git, '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    digest = sha256()
    for name in sorted(tracked):
        path = root / name
        if path.suffix in ('.py', '.yaml', '.yml', '.sh', '.ini') and path.is_file():
            digest.update(name.encode()); digest.update(path.read_bytes())
    adapter = sha256()
    paths = [ROOT / 'scripts/run_author_aligned_p4.py', ROOT / 'scripts/run_author_s4m_p4.py',
             *sorted((ROOT / 'integrations/forecast_compare').glob('*.py'))]
    for path in paths:
        adapter.update(path.name.encode()); adapter.update(path.read_bytes())
    return {'author_profiles_sha256': file_sha256(profile_path), 'upstream_revision': revision,
            'upstream_source_sha256': digest.hexdigest(), 'adapter_sha256': adapter.hexdigest(),
            'protocol_config_sha256': file_sha256(protocol.config_path),
            'source_csv_sha256': protocol.source_sha256,
            'validation_mask_sha256': protocol.mask_sha256['validation'],
            'original_test_mask_sha256': protocol.mask_sha256['test']}
