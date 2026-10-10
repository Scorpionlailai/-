from dataclasses import replace
from pathlib import Path

import pytest

import malr.cli as cli
from malr.config import load_config


def test_stress_protocol_accepts_only_higher_start_rate() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(
        source, missing=replace(source.missing, start_rate=0.15)
    )

    cli.validate_stress_config(source, target)


def test_sr015_config_changes_only_missing_start_rate() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_sr015.yaml")
    )

    cli.validate_stress_config(source, target)
    assert target.missing.start_rate == 0.15
    assert target.missing.block_len == 8


def test_sr020_config_changes_only_missing_start_rate() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_sr020.yaml")
    )

    cli.validate_stress_config(source, target)
    assert target.missing.start_rate == 0.20
    assert target.missing.block_len == 8


def test_stress_protocol_accepts_shorter_block_at_same_rate() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(source, missing=replace(source.missing, block_len=5))

    cli.validate_stress_config(source, target)


def test_stress_protocol_rejects_simultaneous_rate_and_block_change() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(
        source,
        missing=replace(source.missing, start_rate=0.15, block_len=5),
    )

    with pytest.raises(ValueError, match="only one missingness parameter"):
        cli.validate_stress_config(source, target)


def test_sr010_bl5_e22_config_changes_only_missing_block_len() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_bl5.yaml")
    )

    cli.validate_stress_config(source, target)
    assert target.missing.start_rate == 0.10
    assert target.missing.block_len == 5


def test_sr010_bl3_e22_config_changes_only_missing_block_len() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_bl3.yaml")
    )

    cli.validate_stress_config(source, target)
    assert target.missing.start_rate == 0.10
    assert target.missing.block_len == 3


@pytest.mark.parametrize(
    ("block_len", "message"),
    ((8, "missingness parameter change"), (9, "shorter missing block_len")),
)
def test_stress_protocol_rejects_unchanged_or_longer_block(
    block_len: int, message: str
) -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(source, missing=replace(source.missing, block_len=block_len))

    with pytest.raises(ValueError, match=message):
        cli.validate_stress_config(source, target)


def test_stress_protocol_rejects_architecture_change_with_shorter_block() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(
        source,
        missing=replace(source.missing, block_len=5),
        patch=replace(source.patch, stride=4),
    )

    with pytest.raises(ValueError, match="only missing block_len"):
        cli.validate_stress_config(source, target)


@pytest.mark.parametrize(
    ("rate", "message"),
    ((0.10, "missingness parameter change"), (0.05, "higher missing start_rate")),
)
def test_stress_protocol_rejects_unchanged_or_lower_pressure(
    rate: float, message: str
) -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(source, missing=replace(source.missing, start_rate=rate))

    with pytest.raises(ValueError, match=message):
        cli.validate_stress_config(source, target)


def test_stress_protocol_rejects_other_architecture_changes() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(
        source,
        missing=replace(source.missing, start_rate=0.15),
        patch=replace(source.patch, stride=4),
    )

    with pytest.raises(ValueError, match="only missing start_rate"):
        cli.validate_stress_config(source, target)
