from dataclasses import asdict, replace
from pathlib import Path

import pytest

from malr import cli
from malr.config import load_config
from malr.data.splits import SplitBounds, protocol_from_bounds


DATASETS = {
    "etth2": ("ETTh2", 7, 17420, (0, 8640, 11520, 14400)),
    "ettm1": ("ETTm1", 7, 69680, (0, 34560, 46080, 57600)),
    "ettm2": ("ETTm2", 7, 69680, (0, 34560, 46080, 57600)),
    "exchange": ("Exchange", 8, 7588, (0, 5311, 6071, 7588)),
    "appliances": ("Appliances", 26, 19735, (0, 13814, 15788, 19735)),
    "beijing_air_quality": (
        "BeijingAirQuality",
        12,
        35064,
        (0, 24544, 28052, 35064),
    ),
}


@pytest.mark.parametrize("folder, expected", DATASETS.items())
def test_multidataset_main_configs_load_one_dataset(
    folder: str, expected: tuple[object, ...]
) -> None:
    dataset_name, n_vars, length, boundaries = expected
    config = load_config(Path(f"configs/datasets/{folder}/sr010_bl8.yaml"))

    assert set(config.data.datasets) == {dataset_name}
    dataset = config.data.datasets[str(dataset_name)]
    assert (dataset.n_vars, dataset.expected_length) == (n_vars, length)
    assert (
        dataset.train_start,
        dataset.train_end,
        dataset.validation_end,
        dataset.test_end,
    ) == boundaries
    assert set(config.model.profiles) == {dataset_name}
    assert set(config.training.profiles) == {dataset_name}
    assert config.relations.online.protocol == "dynamic_context_v3"
    assert config.relations.online.stage1_repair_epochs == 100
    assert (config.data.seq_len, config.data.pred_len) == (96, 96)
    assert (config.missing.start_rate, config.missing.block_len) == (0.10, 8)


@pytest.mark.parametrize("folder", DATASETS)
def test_each_dataset_has_four_locked_scenarios_and_only_recorded_supplements(
    folder: str,
) -> None:
    root = Path("configs/datasets") / folder
    base = load_config(root / "sr010_bl8.yaml")
    scenarios = {
        "sr010_bl5.yaml": (0.10, 5),
        "sr010_bl8.yaml": (0.10, 8),
        "sr015_bl8.yaml": (0.15, 8),
        "sr020_bl8.yaml": (0.20, 8),
    }

    supplements = {"sr010_bl3.yaml"} if folder in {"ettm1", "exchange"} else set()
    assert {path.name for path in root.glob("*.yaml")} == set(scenarios) | supplements
    for filename, (start_rate, block_len) in scenarios.items():
        candidate = load_config(root / filename)
        assert (candidate.missing.start_rate, candidate.missing.block_len) == (
            start_rate,
            block_len,
        )
        assert replace(candidate, missing=base.missing) == base
    if folder in {"ettm1", "exchange"}:
        supplementary = load_config(root / "sr010_bl3.yaml")
        assert (
            supplementary.missing.start_rate,
            supplementary.missing.block_len,
        ) == (0.10, 3)
        assert replace(supplementary, missing=base.missing) == base


def test_configured_protocol_accepts_new_dataset_boundaries() -> None:
    protocol = protocol_from_bounds(
        "ETTm1",
        69680,
        train_start=0,
        train_end=34560,
        validation_end=46080,
        test_end=57600,
    )

    assert protocol.train == SplitBounds(0, 34560)
    assert protocol.validation == SplitBounds(34560, 46080)
    assert protocol.test == SplitBounds(46080, 57600)


def test_cli_accepts_any_dataset_key_declared_by_a_config() -> None:
    args = cli.build_parser().parse_args(
        [
            "validate-data",
            "--config",
            "configs/datasets/exchange/sr010_bl8.yaml",
            "--dataset",
            "Exchange",
        ]
    )

    assert args.dataset == "Exchange"
    assert cli._project_root(args.config) == Path.cwd()


def test_inherited_stress_config_is_accepted_as_a_single_parameter_change() -> None:
    base = load_config(Path("configs/datasets/ettm1/sr010_bl8.yaml"))
    lower = load_config(Path("configs/datasets/ettm1/sr010_bl5.yaml"))
    higher = load_config(Path("configs/datasets/ettm1/sr015_bl8.yaml"))

    cli.validate_stress_config(base, lower)
    cli.validate_stress_config(base, higher)
    assert asdict(base)["relations"] == asdict(lower)["relations"]
