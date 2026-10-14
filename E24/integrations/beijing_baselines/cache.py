"""Immutable FP32 repair caches; targets are never passed to the repairer."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import file_sha256, write_json


class RepairCache:
    def __init__(self, model, data, output, device, batch_size, *, allow_generation=True):
        self.model, self.data, self.device = model, data, device
        self.allow_generation = allow_generation
        self.directory = Path(output) / 'repair_cache'
        self.directory.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.checkpoint_sha = file_sha256(Path(output) / 'stage1/selected.pt')
        self.records = {}
        self.arrays = {}

    def load(self, split, mask_seed=None):
        from .run import repair, state_sha
        samples = (1 if split == 'train' else self.model.n_samples) if hasattr(self.model, 'n_samples') else None
        key = '{}_{}'.format(split, mask_seed or 2021)
        if key in self.arrays:
            return self.arrays[key]
        directory = self.directory / key
        if not self.allow_generation and not (directory / 'manifest.json').is_file():
            raise FileNotFoundError('forecasting requires a sealed existing repair cache: {}'.format(directory))
        directory.mkdir(exist_ok=True)
        binding = dict(schema=1, split=split, mask_seed=mask_seed or 2021,
                       checkpoint_sha256=self.checkpoint_sha, state_sha256=state_sha(self.model),
                       data_manifest_sha256=self.data.manifest_sha, batch_size=self.batch_size,
                       implementation_sha256={p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
                       samples=samples, fixed_training_epoch=0, training_mask_seed=2023,
                       sampling_seed_rule='910000 + mask_seed_or_2021 + first_window_start',
                       dtype='float32', shape=[len(self.data.starts[split]), 96, self.data.n_vars],
                       starts_sha256=hashlib.sha256(self.data.starts[split].tobytes()).hexdigest())
        manifest = directory / 'manifest.json'
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if record['binding'] != binding:
                raise ValueError('stale repair cache: {}'.format(directory))
            for name, digest in record['files'].items():
                if file_sha256(directory / name) != digest:
                    raise ValueError('corrupt repair cache: {}'.format(name))
        else:
            started = time.perf_counter()
            shape = tuple(binding['shape'])
            completed = np.lib.format.open_memmap(directory / 'completed.npy', mode='w+', dtype=np.float32, shape=shape)
            masks = np.lib.format.open_memmap(directory / 'observed.npy', mode='w+', dtype=bool, shape=shape)
            original_samples = getattr(self.model, 'n_samples', None)
            if original_samples is not None:
                self.model.n_samples = samples
            self.model.eval()
            offset = 0
            try:
                with torch.no_grad():
                    for truth, observed, starts in self.data.batches(split, self.batch_size, mask_seed=mask_seed,
                            fixed_training=True, ordered=True):
                        truth, observed = truth.to(self.device), observed.to(self.device)
                        visible = torch.where(observed, truth, 0.)
                        # TorchScript profiling changes fusion after the first calls.
                        # Match the already-trained BRITS execution path before sealing.
                        if offset == 0 and self.model.__class__.__name__ == 'BritsAdapter':
                            for _ in range(2):
                                repair(self.model, visible, observed, starts, 910000 + (mask_seed or 2021) + int(starts[0]))
                        result = repair(self.model, visible, observed, starts, 910000 + (mask_seed or 2021) + int(starts[0]))
                        if not torch.isfinite(result).all() or not torch.equal(result[observed], visible[observed]):
                            raise AssertionError('invalid cached repair')
                        n = len(starts)
                        if not np.array_equal(starts, self.data.starts[split][offset:offset+n]):
                            raise AssertionError('cache window ordering differs')
                        completed[offset:offset+n] = result.cpu().numpy()
                        masks[offset:offset+n] = observed.cpu().numpy()
                        offset += n
            finally:
                if original_samples is not None:
                    self.model.n_samples = original_samples
            if offset != shape[0] or state_sha(self.model) != binding['state_sha256']:
                raise AssertionError('cache support or frozen model changed')
            completed.flush()
            masks.flush()
            del completed, masks
            record = dict(binding=binding, generation_seconds=time.perf_counter()-started,
                          observed_sha256=file_sha256(directory / 'observed.npy'),
                          files={name: file_sha256(directory / name) for name in ('completed.npy', 'observed.npy')})
            write_json(manifest, record)
        self.records[key] = record
        self.arrays[key] = (np.load(directory / 'completed.npy', mmap_mode='r'),
                            np.load(directory / 'observed.npy', mmap_mode='r'))
        return self.arrays[key]

    def batches(self, split, batch_size, mask_seed=None, epoch=0):
        completed, observed = self.load(split, mask_seed)
        indices = np.arange(len(completed))
        if split == 'train':
            np.random.default_rng(2023 + epoch).shuffle(indices)
            indices = indices[:len(indices)//batch_size*batch_size]
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset+batch_size]
            yield (torch.from_numpy(completed[selected]), torch.from_numpy(observed[selected]),
                   self.data.starts[split][selected])
