"""Read-only final audit of selected weights, cached outputs, and 20-mask metrics."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from integrations.repair_baselines.data import file_sha256

ROOT = Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010')
torch.set_num_threads(4)
cases = []
for folder in sorted(ROOT.glob('*/*/*/seed_2023')):
    selection = json.loads((folder / 'stage2/selection.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    state = torch.load(folder / 'stage1/selected.pt', map_location='cpu', weights_only=True)
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    state_digest = digest.hexdigest()
    assert state_digest == selection['repair_state_sha256']
    checkpoint_digest = file_sha256(folder / 'stage1/selected.pt')
    manifest_count = test_count = file_count = 0
    for manifest in sorted((folder / 'repair_cache').glob('*/manifest.json')):
        record = json.loads(manifest.read_text())
        binding = record['binding']
        assert binding['state_sha256'] == state_digest
        assert binding['checkpoint_sha256'] == checkpoint_digest
        assert binding['data_manifest_sha256'] == plan['data_manifest_sha256']
        for name, expected in record['files'].items():
            assert file_sha256(manifest.parent / name) == expected, str(manifest.parent / name)
            file_count += 1
        for name in ('completed.npy', 'observed.npy'):
            array = np.load(manifest.parent / name, mmap_mode='r')
            assert list(array.shape) == binding['shape']
        manifest_count += 1
        test_count += binding['split'] == 'test'
    assert manifest_count == 22 and test_count == 20
    for label in ('repair', 'forecast_mae', 'forecast_mse'):
        result = json.loads((folder / f'evaluation/{label}_summary.json').read_text())
        groups = result['groups']
        assert len(groups) == len({group['mask_seed'] for group in groups}) == 20
        for section in ('full96', 'recent32') + (() if label == 'repair' else ('forecast',)):
            for metric, total in (('mse', 'squared_error_sum'), ('mae', 'absolute_error_sum')):
                values = np.array([group[section][metric] for group in groups], dtype=np.float64)
                assert np.isfinite(values).all()
                assert np.isclose(values.mean(), result['summary'][section][metric + '_mean'], rtol=0., atol=1e-12)
                assert np.isclose(values.std(ddof=1), result['summary'][section][metric + '_mask_std_ddof1'], rtol=0., atol=1e-12)
                for group in groups:
                    item = group[section]
                    assert np.isclose(item[metric], item[total] / item['coordinates'], rtol=0., atol=1e-12)
    cases.append(dict(run=str(folder.relative_to(ROOT)), cache_manifests=manifest_count,
                      test_mask_groups=test_count, checked_files=file_count, selected_state_matches=True,
                      cache_file_hashes_match=True, metric_aggregation_passed=True))
    print(json.dumps(cases[-1]), flush=True)
assert len(cases) == 32
record = dict(passed=True, checked_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              cases=cases, total_test_groups=sum(case['test_mask_groups'] for case in cases),
              total_cache_files=sum(case['checked_files'] for case in cases), repair_calls=0)
(ROOT / 'final_saved_output_audit.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key != 'cases'}), flush=True)
