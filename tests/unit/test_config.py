from pathlib import Path

import pytest

from malr.config import config_sha256, load_config


def test_stage1_defaults_are_locked() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))

    assert (cfg.data.seq_len, cfg.data.pred_len) == (96, 96)
    assert (cfg.patch.length, cfg.patch.stride, cfg.patch.count) == (16, 8, 12)
    assert cfg.missing.start_rate == pytest.approx(0.06)
    assert cfg.relations.topk == {
        "temporal": 6,
        "variable": 4,
        "historical": 4,
    }
    assert cfg.training.seed == 2021


def test_invalid_multilayer_recovery_is_rejected(tmp_path: Path) -> None:
    text = Path("configs/stage1.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        text.replace(
            "num_message_passing_layers: 1",
            "num_message_passing_layers: 2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_message_passing_layers must equal 1"):
        load_config(bad)


def test_config_hash_is_stable() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))

    assert config_sha256(cfg) == config_sha256(cfg)
    assert len(config_sha256(cfg)) == 64
