from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from integrations.s4m.overlay.malr_fair.data import (
    FairS4MDataset,
    build_s4m_sample,
    fair_data_provider,
)
from malr.data.masks import (
    EpochMaskManager,
    generate_variable_missing_mask,
    save_fixed_mask,
)
from malr.evaluation.report import file_sha256


@pytest.fixture(scope="module")
def fair_fixture(tmp_path_factory: pytest.TempPathFactory) -> Namespace:
    root = tmp_path_factory.mktemp("s4m-fair-data")
    csv_path = root / "ETTh1.csv"
    length = 14_400
    time = np.arange(length, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=length, freq="h"),
            **{
                f"v{variable}":
                np.sin(time / (13.0 + variable)) + variable * 0.1 + time * 1e-4
                for variable in range(7)
            },
        }
    )
    frame.to_csv(csv_path, index=False)

    raw = yaml.safe_load(
        Path("configs/stage1_sr010_no_recloss.yaml").read_text(encoding="utf-8")
    )
    raw["data"]["datasets"]["ETTh1"]["path"] = str(csv_path)
    raw["data"]["datasets"]["ETTh1"]["expected_length"] = length
    raw["data"]["datasets"]["ETTh1"]["allow_length_override"] = False
    config_path = root / "stage1.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    artifacts = root / "artifacts"
    masks: dict[str, dict[str, object]] = {}
    for split, support_start, support_end in (
        ("validation", 8640 - 96, 11520),
        ("test", 11520 - 96, 14400),
    ):
        observed, metadata = generate_variable_missing_mask(
            length=support_end - support_start,
            n_vars=7,
            start_rate=0.10,
            block_len=5,
            seed=2021,
            dataset="ETTh1",
            split=split,
            support_start=support_start,
        )
        path = artifacts / "masks" / f"ETTh1_{split}.npz"
        save_fixed_mask(path, observed, metadata)
        masks[split] = {
            "path": path,
            "observed": observed,
            "file_sha256": file_sha256(path),
            "final_rate": metadata.final_rate,
        }

    return Namespace(
        malr_config=str(config_path),
        malr_dataset="ETTh1",
        malr_artifacts=str(artifacts),
        fair_variant="missing",
        fair_eval_reference_batch=128,
        batch_size=16,
        num_workers=0,
        seed=2021,
        masks=masks,
    )


def test_validation_uses_exact_fixed_mask(fair_fixture: Namespace) -> None:
    dataset = FairS4MDataset(fair_fixture, "val")

    assert dataset.refs[0].input_start == 8640 - 96
    assert dataset.refs[0].target_start == 8640
    assert dataset.refs[-1].target_end <= 11520
    assert len(dataset.refs) == 2688
    assert np.array_equal(
        dataset.observed_timeline,
        fair_fixture.masks["validation"]["observed"],
    )
    assert (
        dataset.identity()["mask_sha256"]
        == fair_fixture.masks["validation"]["file_sha256"]
    )


def test_training_mask_matches_malr_epoch_manager(fair_fixture: Namespace) -> None:
    dataset = FairS4MDataset(fair_fixture, "train")
    dataset.set_epoch(7)
    expected = EpochMaskManager(
        length=8640,
        n_vars=7,
        start_rate=0.10,
        block_len=5,
        base_seed=2021,
    ).mask_for_epoch(7)

    assert np.array_equal(dataset.observed_timeline, expected)
    dataset.set_epoch(7)
    assert np.array_equal(dataset.observed_timeline, expected)
    dataset.set_epoch(8)
    assert not np.array_equal(dataset.observed_timeline, expected)


def test_build_sample_does_not_leak_masked_values() -> None:
    clean = np.arange(24, dtype=np.float32).reshape(8, 3)
    target = np.arange(12, dtype=np.float32).reshape(4, 3)
    observed = np.ones_like(clean, dtype=bool)
    observed[2:6, 1] = False
    mutated = clean.copy()
    mutated[~observed] = 100_000.0

    original = build_s4m_sample(clean, target, observed)
    changed = build_s4m_sample(mutated, target, observed)

    for index in (0, 1, 2, 3, 6, 7, 8, 9):
        assert np.array_equal(original[index], changed[index])
    assert np.all(original[0][~observed] == 0.0)
    assert np.array_equal(original[1], observed.astype(np.float32))


def test_auxiliary_tensors_follow_official_custom4_formula() -> None:
    clean = np.asarray([[2.0, -2.0], [8.0, 4.0], [-3.0, 5.0]], dtype=np.float32)
    observed = np.asarray([[1, 1], [0, 1], [1, 0]], dtype=bool)
    sample = build_s4m_sample(clean, np.ones((2, 2), dtype=np.float32), observed)

    expected_x = np.asarray([[2.0, -2.0], [0.0, 4.0], [-3.0, 0.0]], dtype=np.float32)
    assert np.array_equal(sample[0], expected_x)
    assert np.array_equal(sample[6], [[0, 1], [1, 0], [2, 1]])
    assert np.array_equal(sample[7], [[2, 0], [1, 1], [0, 2]])
    assert np.array_equal(sample[8], [[2, 4], [2, 4], [2, 4]])
    assert np.array_equal(sample[9], [[-3, -2], [-3, -2], [-3, -2]])


def test_data_provider_keeps_canonical_eval_refs(fair_fixture: Namespace) -> None:
    dataset, loader = fair_data_provider(fair_fixture, "test")
    first = next(iter(loader))

    assert len(dataset) == 2688
    assert first[0].shape == (16, 96, 7)
    assert first[5].shape == (16, 96, 7)
    assert dataset.identity()["sample_count"] == 2688


def test_clean_variant_has_no_mask_dependency(fair_fixture: Namespace) -> None:
    args = Namespace(**{**vars(fair_fixture), "fair_variant": "clean"})
    dataset = FairS4MDataset(args, "test")

    assert dataset.observed_timeline.all()
    assert dataset.identity()["mask_sha256"] is None
    assert dataset.identity()["final_rate"] == 0.0
