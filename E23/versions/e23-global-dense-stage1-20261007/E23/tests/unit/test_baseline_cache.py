from dataclasses import replace
from pathlib import Path

import malr.cli as cli


CONFIG = Path("configs/stage1_seq96_relational_recovery.yaml")


def test_baseline_fingerprints_ignore_relation_only_changes() -> None:
    config, series, protocol = cli._load_context(CONFIG, "ETTh1")
    clean = cli._baseline_fingerprint(config, series, protocol, "ETTh1", "clean")
    missing = cli._baseline_fingerprint(config, series, protocol, "ETTh1", "missing")
    changed = replace(
        config,
        relations=replace(
            config.relations,
            online=replace(config.relations.online, gain_loss_weight=0.37),
        ),
    )
    assert cli._baseline_fingerprint(
        changed, series, protocol, "ETTh1", "clean"
    ) == clean
    assert cli._baseline_fingerprint(
        changed, series, protocol, "ETTh1", "missing"
    ) == missing


def test_only_missing_fingerprint_changes_with_missing_protocol() -> None:
    config, series, protocol = cli._load_context(CONFIG, "ETTh1")
    changed = replace(
        config,
        missing=replace(config.missing, start_rate=0.15),
    )
    assert cli._baseline_fingerprint(
        changed, series, protocol, "ETTh1", "clean"
    ) == cli._baseline_fingerprint(config, series, protocol, "ETTh1", "clean")
    assert cli._baseline_fingerprint(
        changed, series, protocol, "ETTh1", "missing"
    ) != cli._baseline_fingerprint(config, series, protocol, "ETTh1", "missing")


def test_baseline_cache_uses_dataset_sequence_and_missing_hierarchy(
    tmp_path: Path,
) -> None:
    config, series, protocol = cli._load_context(CONFIG, "ETTh1")
    metadata = cli._base_metadata(config, series, protocol, "ETTh1")
    experiment = tmp_path / "artifacts" / "legacy-experiment"
    clean_path = cli._baseline_cache_path(
        experiment,
        "ETTh1",
        "clean",
        metadata["clean_baseline_fingerprint"],
        metadata,
    )
    missing_path = cli._baseline_cache_path(
        experiment,
        "ETTh1",
        "missing",
        metadata["missing_baseline_fingerprint"],
        metadata,
    )
    assert clean_path == (
        tmp_path
        / "artifacts"
        / "ETTh1"
        / "seq96_pred96"
        / "clean"
        / metadata["clean_baseline_fingerprint"]
        / "baseline.json"
    )
    assert missing_path == (
        tmp_path
        / "artifacts"
        / "ETTh1"
        / "seq96_pred96"
        / "sr010_bl8"
        / "missing"
        / metadata["missing_baseline_fingerprint"]
        / "baseline.json"
    )
