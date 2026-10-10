"""Synchronized wall time and PyTorch CUDA peaks for complete training commands.

Sidecars keep resource measurements separate from immutable selection manifests.
CUDA peaks include parameters, gradients, optimizer state and temporary tensors;
they exclude CUDA context and allocations made outside the PyTorch allocator.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

import torch

from malr.evaluation.report import file_sha256


class TrainingResources:
    def __init__(self, output: Path, *, device: str | torch.device, batch_size: int,
                 model: str, mode: str = 'training', started_at: float | None = None,
                 precision: str = 'float32', amp: bool = False):
        self.output = Path(output)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = model
        self.mode = mode
        self.precision = precision
        self.amp = amp
        self.started_at = started_at
        self.stages: dict[str, dict] = {}
        self.peak_allocated: int | None = None
        self.peak_reserved: int | None = None
        self.selection_manifest: Path | None = None
        self.report_path: Path | None = None

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _peaks(self) -> dict:
        if self.device.type != 'cuda':
            return {'peak_allocated_bytes': None, 'peak_reserved_bytes': None}
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        self.peak_allocated = max(self.peak_allocated or 0, allocated)
        self.peak_reserved = max(self.peak_reserved or 0, reserved)
        return {'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved}

    def __enter__(self):
        self._sync()
        self.started_at = perf_counter() if self.started_at is None else self.started_at
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.device_info = {'type': self.device.type, 'requested': str(self.device)}
        if self.device.type == 'cuda':
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            self.device_info.update(index=index, name=props.name, total_memory_bytes=props.total_memory)
            self.device_info['uuid'] = str(getattr(props, 'uuid', 'unavailable'))
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self.device_info['name'] = platform.processor() or platform.machine()
        return self

    @contextmanager
    def stage(self, name: str):
        if name in self.stages:
            raise ValueError(f'duplicate training stage: {name}')
        self._sync()
        self._peaks()  # Preserve preparation/previous-stage peaks before resetting.
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        status = 'failed'
        try:
            yield
            status = 'completed'
        finally:
            self._sync()
            self.stages[name] = {'wall_seconds': perf_counter() - started,
                                 'status': status, **self._peaks()}

    def __exit__(self, exc_type, exc_value, traceback):
        self._sync()
        full_seconds = perf_counter() - self.started_at
        self._peaks()
        stage_seconds = sum(row['wall_seconds'] for row in self.stages.values())
        report = {
            'schema_version': 1, 'model': self.model, 'mode': self.mode,
            'status': 'completed' if exc_type is None else 'failed',
            'started_utc': self.started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'full_workflow_seconds': full_seconds,
            'training_stages_seconds': stage_seconds,
            'preparation_and_reporting_seconds': max(0., full_seconds - stage_seconds),
            'stages': self.stages, 'peak_allocated_bytes': self.peak_allocated,
            'peak_reserved_bytes': self.peak_reserved, 'device': self.device_info,
            'batch_size': self.batch_size, 'precision': self.precision, 'amp': self.amp,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'timing_scope': 'entrypoint imports, data preparation, model/optimizer setup, all training '
                            'epochs, validation and checkpoint/selection saving; excludes final test '
                            'and the resource sidecar write',
            'memory_scope': 'PyTorch CUDA allocator peaks for this process, not whole-device usage',
        }
        if self.selection_manifest is not None and exc_type is None:
            path = Path(self.selection_manifest).resolve()
            selected = json.loads(path.read_text(encoding='utf-8'))
            report.update(selection_manifest=str(path), selection_manifest_sha256=file_sha256(path),
                          checkpoint_sha256=selected.get('checkpoint_sha256'),
                          epochs_completed=selected.get('epochs_completed'))
        if exc_type is not None:
            report['error_type'] = exc_type.__name__
        self.report_path = self.output / 'resources' / f'{uuid4().hex}.json'
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(f'TRAINING_RESOURCES {self.report_path}', flush=True)
        return False


def summarize_resources(records: list[dict]) -> dict:
    """Sum sequential workflows, take peak memory; missing stages stay unknown."""
    if not records or any(row.get('status') != 'completed' or row.get('mode') != 'training'
                          for row in records):
        raise ValueError('resource summary requires completed training records')

    def aggregate(key, operation):
        values = [row.get(key) for row in records]
        return None if any(value is None for value in values) else operation(values)

    allocated = aggregate('peak_allocated_bytes', max)
    reserved = aggregate('peak_reserved_bytes', max)
    devices = sorted({row['device']['name'] for row in records})
    batches = sorted({row['batch_size'] for row in records})
    return {
        'full_workflow_seconds': aggregate('full_workflow_seconds', sum),
        'training_stages_seconds': aggregate('training_stages_seconds', sum),
        'peak_allocated_gib': None if allocated is None else allocated / 1024**3,
        'peak_reserved_gib': None if reserved is None else reserved / 1024**3,
        'gpu': ';'.join(devices), 'batch_size': ';'.join(map(str, batches)),
        'hardware_consistent': len(devices) == 1 and len(batches) == 1,
    }


def selected_training_resources(manifests: list[Path]) -> dict:
    """Attach only sidecars tied to these exact manifests; never infer old peaks."""
    loops = []
    measured = []
    record_paths = []
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        selected = json.loads(manifest.read_text(encoding='utf-8'))
        loops.append(selected.get('training_seconds', selected.get('training', {}).get('training_seconds')))
        digest = file_sha256(manifest)
        matches = []
        for directory in (manifest.parent / 'resources', manifest.parent.parent / 'resources'):
            for path in directory.glob('*.json'):
                row = json.loads(path.read_text(encoding='utf-8'))
                if (row.get('selection_manifest_sha256') == digest and row.get('status') == 'completed'
                        and row.get('mode') == 'training'):
                    matches.append((row.get('finished_utc', ''), path, row))
        if matches:
            _, path, row = max(matches, key=lambda item: item[0])
            measured.append(row)
            record_paths.append(str(path))
    complete = len(measured) == len(manifests) and bool(manifests)
    summary = summarize_resources(measured) if complete else {
        'full_workflow_seconds': None, 'training_stages_seconds': None,
        'peak_allocated_gib': None, 'peak_reserved_gib': None, 'gpu': None,
        'batch_size': None, 'hardware_consistent': None,
    }
    summary.update(
        legacy_training_loop_seconds=sum(loops) if loops and all(v is not None for v in loops) else None,
        stage1_legacy_loop_seconds=loops[0] if len(loops) == 2 else None,
        stage2_legacy_loop_seconds=loops[1] if len(loops) == 2 else None,
        selection_manifest_paths=[str(Path(path).resolve()) for path in manifests],
        resource_record_paths=record_paths,
        resource_status='measured_all_stages' if complete else 'legacy_time_only_training_memory_unrecorded',
    )
    return summary
