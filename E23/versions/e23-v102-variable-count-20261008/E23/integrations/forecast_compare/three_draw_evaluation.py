"""Evaluate one selected model on three frozen test draws without selecting again."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from malr.evaluation.report import file_sha256

from .protocol import P4Protocol
from .test_mask_bank import EXTRA_TEST_SEEDS, load_test_mask_bank


def evaluate_three_draws(protocol: P4Protocol, bank: Path, selection: Path,
                        score: Callable[[P4Protocol], dict]) -> dict:
    """The caller verifies source/config and loads the selected weights once."""
    bank, selection = Path(bank).resolve(), Path(selection).resolve()
    protocols = load_test_mask_bank(protocol, bank)
    entries = json.loads(bank.read_text())['groups']
    if tuple(entry['seed'] for entry in entries) != (2021, *EXTRA_TEST_SEEDS):
        raise ValueError('formal test mask seeds differ from the predeclared protocol')
    record = json.loads(selection.read_text())
    checkpoint = Path(record['checkpoint'])
    checkpoint_sha = record['checkpoint_sha256']
    selection_sha, bank_sha = file_sha256(selection), file_sha256(bank)

    def verify_locked():
        if file_sha256(checkpoint) != checkpoint_sha:
            raise ValueError('selected checkpoint changed during three-draw evaluation')
        if file_sha256(selection) != selection_sha or file_sha256(bank) != bank_sha:
            raise ValueError('selection or mask bank changed during three-draw evaluation')

    verify_locked()
    rows = []
    for group, entry in zip(protocols, entries):
        metrics = score(group)
        verify_locked()
        if any(not math.isfinite(float(metrics[key])) or float(metrics[key]) < 0
               for key in ('mse', 'mae')):
            raise ValueError('test metrics must be finite and nonnegative')
        windows = len(protocol.refs['test']) // 128 * 128
        if metrics.get('windows') != windows or metrics.get('target_elements') != windows * 96 * 7:
            raise ValueError('each test draw must evaluate the same complete P4 window set')
        rows.append({**entry, **metrics})
    return {
        'schema_version': 1, 'dataset': 'ETTh1', 'scenario': 'P4',
        'training_runs': 1, 'test_draws': 3, 'test_groups': rows,
        'test_mean': {key: float(np.mean([row[key] for row in rows])) for key in ('mse', 'mae')},
        'test_std': {key: float(np.std([row[key] for row in rows], ddof=0)) for key in ('mse', 'mae')},
        'std_definition': 'population standard deviation across three test missing-position draws (ddof=0)',
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
        'selection_sha256': selection_sha, 'test_mask_bank': str(bank),
        'test_mask_bank_sha256': bank_sha,
        'source_csv_sha256': protocol.source_sha256,
        'validation_mask_sha256': protocol.mask_sha256['validation'],
        'metric_space': 'train_standardized',
        'checkpoint_selection': 'original fixed validation only; same selected weights in every test group',
    }
