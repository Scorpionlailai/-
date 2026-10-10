"""Read-only mask geometry probe; no model training or checkpoint selection."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil

import numpy as np
from integrations.e23.config import load_e23_p4_protocol
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_preserving_values
from malr.migration import MigrationCatalog

BASE = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments')
OUT = BASE / 'e23_long_gap_focus_diagnosis_20261005'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runs(observed):
    edges = np.flatnonzero(np.diff(np.r_[False, ~observed, False].astype(np.int8)))
    return edges[::2], edges[1::2]


def main():
    # Explicit edges and fully missing lookbacks verify the binning convention.
    a, b = runs(np.array([False, False, True, False, True], dtype=bool))
    assert a.tolist() == [0, 3] and b.tolist() == [2, 4]
    a, b = runs(np.zeros(96, dtype=bool))
    assert a.tolist() == [0] and b.tolist() == [96]
    a, b = runs(np.ones(96, dtype=bool))
    assert len(a) == len(b) == 0
    work = Path.cwd()
    rows = []
    inputs = {}
    for name in ['sr015_bl8', 'sr020_bl8', 'sr010_bl16']:
        scenario = BASE / 't1_v41_higher_missing_seed502_20261004' / name / 'scenario.json'
        os.environ['MALR_MISSING_SCENARIO'] = str(scenario)
        protocol = load_e23_p4_protocol(
            catalog=MigrationCatalog.load(work.parent / 'E22/migration/paper_catalog.jsonl', project='E23'),
            protocol_config=work / 'configs/stage1_seq96_dynamic_temporal_repair.yaml',
            mask_artifacts=work.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',
            training_seed=2023,
        )
        if name == 'sr015_bl8':
            diag_path = BASE / 'e23_v69_signature_position_sr015_bl8_20261005/seed_2023_frozen_mae/recovery-diagnostics.json'
        else:
            diag_path = BASE / 'e23_v69_other_missing_seed2023_20261005' / name / 'E23-v69-mae/recovery-diagnostics.json'
        diag = next(d for d in json.loads(diag_path.read_text()) if d['stage'] == 'stage1')
        record_path = Path(diag['selection_manifest'])
        record = json.loads(record_path.read_text())
        checkpoint = Path(record['checkpoint'])
        mask_path = protocol.artifacts / 'ETTh1_validation.npz'
        # The scenario manifest supplies the authoritative relocated mask path.
        manifest = json.loads(scenario.read_text())
        mask_path = scenario.parent / manifest['masks']['validation']['path']
        for path in [scenario, diag_path, record_path, checkpoint, mask_path, protocol.config_path]:
            inputs[str(path)] = sha(path)
        assert sha(mask_path) == protocol.mask_sha256['validation']
        assert sha(checkpoint) == record['checkpoint_sha256']
        hidden = long_hidden = interior_hidden = full = damaged = 0
        max_gaps = []
        no_variable_observation = 0
        for batch in iter_p4_batches(protocol, 'validation', batch_size=128):
            patches = patchify_preserving_values(batch.normalized_observed, batch.observed_mask, length=16, stride=8)
            full += (~patches.effective_mask.any(-1)).sum().item()
            damaged += patches.effective_mask.ne(patches.validity).any(-1).sum().item()
            for mask in batch.observed_mask.numpy().transpose(0, 2, 1).reshape(-1, 96):
                missing = ~mask
                hidden += int(missing.sum())
                visible = np.flatnonzero(mask)
                starts, stops = runs(mask)
                lengths = stops - starts
                max_gaps.append(int(lengths.max(initial=0)))
                long_hidden += int(lengths[lengths >= 32].sum())
                if len(visible):
                    distance = np.min(np.abs(np.arange(96)[:, None] - visible[None, :]), axis=1)
                    interior_hidden += int((missing & (distance >= 9)).sum())
                else:
                    no_variable_observation += 1
                    interior_hidden += 96
        assert damaged == diag['damaged_patches']
        assert hidden == diag['metrics']['all96']['hidden_coordinates']
        rows.append(dict(
            scene=name, validation_mask_sha256=protocol.mask_sha256['validation'],
            hidden_coordinates=hidden, damaged_patches=damaged, fully_hidden_patches=full,
            fully_hidden_damaged_fraction=full / damaged,
            h_available_damaged_fraction=diag['h_available_damaged_patches'] / damaged,
            window_long_gap_ge32_hidden_fraction=long_hidden / hidden,
            nearest_visible_ge9_or_no_visible_hidden_fraction=interior_hidden / hidden,
            median_per_variable_window_longest_gap=float(np.median(max_gaps)),
            variable_windows_without_observation=no_variable_observation,
            coordinate_count_unit='occurrences across overlapping validation lookbacks',
            gap_length_unit='consecutive missing positions clipped to each 96-point lookback',
            training_runs=0,
        ))
    assert all(sha(path) == digest for path, digest in inputs.items())
    OUT.mkdir(exist_ok=False)
    shutil.copy2(__file__, OUT / 'probe.py')
    result = dict(
        kind='read_only_fixed_validation_mask_probe', model_inference=False,
        new_training_runs=0, forecast_targets_used=False, weights_changed=False,
        rows=rows, input_sha256_before_and_after=inputs,
        limitation='Descriptive geometry, not a causal explanation or a performance improvement.',
    )
    (OUT / 'diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'geometry.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
