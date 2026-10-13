"""Check pending CSDI test banks using real checkpoints; forbid repair calls."""
import json
from pathlib import Path
import torch
from integrations.repair_baselines.adapters import build_adapter, training_profile
from integrations.repair_baselines.cache import RepairCache
from integrations.repair_baselines.data import SharedData, write_json, file_sha256

ROOT = Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010')
torch.set_num_threads(4)
device = torch.device('cpu')
cases = []
for dataset, scenes in [('Weather', ['sr020_bl8', 'sr010_bl16']),
                        ('ETTh1', ['sr010_bl8', 'sr015_bl8', 'sr020_bl8', 'sr010_bl16'])]:
    for scene in scenes:
        data = SharedData(dataset, scene, ROOT / 'shared_assets')
        output = ROOT / dataset / scene / 'CSDI' / 'seed_2023'
        model = build_adapter('CSDI', data, device)
        model.load_state_dict(torch.load(output / 'stage1/selected.pt', map_location=device, weights_only=True))
        plan = json.loads((output / 'plan.json').read_text())
        model.n_samples = plan['model_config']['sampling']['samples']
        model.sample_chunk_size = plan['model_config']['sampling']['sample_chunk_size']
        model.requires_grad_(False).eval()
        calls = [0]
        def forbid_repair(*args, **kwargs):
            calls[0] += 1
            raise AssertionError('cache reuse invoked repair model')
        model.forward = forbid_repair
        cache = RepairCache(model, data, output, device, training_profile('CSDI')['batch_size'])
        for seed in data.test_seeds:
            assert (cache.directory / f'test_{seed}' / 'manifest.json').is_file()
            completed, observed = cache.load('test', seed)
            assert completed.shape == observed.shape == (len(data.starts['test']), 96, data.n_vars)
        cases.append(dict(dataset=dataset, scene=scene, test_mask_groups=len(cache.records), repair_calls=calls[0], passed=True))
        print(json.dumps(cases[-1]), flush=True)
write_json(ROOT / 'prediction_cache_read_only_preflight.json', dict(passed=True, cases=cases,
    run_sha256=file_sha256(Path('/home/xyw/paper/E24/integrations/repair_baselines/run.py')),
    validation='real checkpoint, complete manifests, binding and file hashes, repair forward forbidden'))
