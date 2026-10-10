from pathlib import Path
import shutil

import pytest

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.forecast_compare.protocol import load_p4_protocol
from malr.migration import MigrationCatalog


ROOT = Path(__file__).resolve().parents[2]
CATALOG = MigrationCatalog.load(ROOT.parent / "E22/migration/paper_catalog.jsonl", project="E23")
PROTOCOL_CONFIG = ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml"
E23_CONFIG = ROOT / "configs/e23_multi_expert_p4.yaml"
WIDTH64_CONFIG = ROOT / "configs/e23_multi_expert_p4_width64.yaml"
WIDTH64_TOPP = ROOT / "configs/e23_multi_expert_p4_width64_topp.yaml"
WIDTH128_TOPK = ROOT / "configs/e23_multi_expert_p4_width128_topk.yaml"
WIDTH128_TOPP = ROOT / "configs/e23_multi_expert_p4_width128_topp.yaml"
WINDOW_V12 = ROOT / "configs/e23_multi_expert_p4_width64_window_v12.yaml"
WINDOW_V15 = ROOT / "configs/e23_multi_expert_p4_width64_strong_window_v15.yaml"
TWO_GROUP_V13 = ROOT / "configs/e23_multi_expert_p4_width64_window_two_group_v13.yaml"
MASKS = ROOT.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"


def test_e23_config_separates_model_from_p4(tmp_path):
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    settings = load_e23_settings(E23_CONFIG, protocol=protocol)
    assert (settings.d_model, settings.historical_topk, settings.variable_topk) == (16, 4, 4)
    assert settings.variable_candidate_limit == 12
    assert settings.temporal_candidate_limit == 10
    assert (settings.stage1_lr, settings.stage2_lr) == (0.001, 0.0001)
    assert (settings.batch_size, settings.seed, settings.stage1_epochs, settings.stage2_epochs) == (128, 2021, 100, 100)
    assert len(settings.config_sha256) == 64

    bad = tmp_path / "bad.yaml"
    bad.write_text(E23_CONFIG.read_text() + "\ndata: {train_end: 9000}\n")
    with pytest.raises(ValueError, match="unknown|data|override"):
        load_e23_settings(bad, protocol=protocol)


def test_e23_p4_loads_cataloged_masks(tmp_path):
    assert not (
        ROOT / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_v1"
        / "masks/ETTh1_validation.npz"
    ).exists()
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    assert protocol.source_sha256 == "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
    assert protocol.mask_sha256 == {
        "validation": "cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c",
        "test": "f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8",
    }
    masks_copy = tmp_path / "masks"
    masks_copy.mkdir()
    for split in ("validation", "test"):
        shutil.copyfile(MASKS / "masks" / f"ETTh1_{split}.npz", masks_copy / f"ETTh1_{split}.npz")
    changed = masks_copy / "ETTh1_validation.npz"
    data = bytearray(changed.read_bytes())
    data[-10] ^= 1
    changed.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256"):
        load_p4_protocol(PROTOCOL_CONFIG, tmp_path)
    with pytest.raises(ValueError, match="must come from E22"):
        load_e23_p4_protocol(
            catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=tmp_path,
        )


def test_e23_config_rejects_fractional_model_width(tmp_path):
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    bad = tmp_path / "fractional.yaml"
    bad.write_text(E23_CONFIG.read_text().replace("d_model: 16", "d_model: 16.5"))
    with pytest.raises(ValueError, match="d_model"):
        load_e23_settings(bad, protocol=protocol)


def test_width64_config_changes_only_repair_capacity(tmp_path):
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    base = load_e23_settings(E23_CONFIG, protocol=protocol)
    wider = load_e23_settings(WIDTH64_CONFIG, protocol=protocol)
    assert wider.d_model == 64
    assert wider.config_sha256 != base.config_sha256
    assert {
        name: value for name, value in vars(wider).items()
        if name not in {"d_model", "config_sha256"}
    } == {
        name: value for name, value in vars(base).items()
        if name not in {"d_model", "config_sha256"}
    }
    unsupported = tmp_path / "width32.yaml"
    unsupported.write_text(WIDTH64_CONFIG.read_text().replace("d_model: 64", "d_model: 32"))
    with pytest.raises(ValueError, match="locked|d_model"):
        load_e23_settings(unsupported, protocol=protocol)


def test_trial_configs_are_distinct_and_preserve_locked_training():
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    settings = [load_e23_settings(path, protocol=protocol) for path in
                (WIDTH64_CONFIG, WIDTH64_TOPP, WIDTH128_TOPK, WIDTH128_TOPP)]
    assert [(s.d_model, s.historical_top_p, s.variable_top_p) for s in settings] == [
        (64, None, None), (64, 0.9, 0.9), (128, None, None), (128, 0.9, 0.9),
    ]
    assert len({s.config_sha256 for s in settings}) == 4
    assert all(s.stage1_epochs == s.stage2_epochs == 100 for s in settings)
    assert all(s.batch_size == 128 and s.seed == 2021 for s in settings)


def test_trial_config_rejects_one_sided_top_p(tmp_path):
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    bad = tmp_path / "one_sided.yaml"
    bad.write_text(WIDTH64_TOPP.read_text().replace("  variable_top_p: 0.9\n", ""))
    with pytest.raises(ValueError, match="top-p|setting"):
        load_e23_settings(bad, protocol=protocol)


def test_window_conditioned_v_config_has_distinct_identity_and_fixed_predictor():
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    base = load_e23_settings(WIDTH64_CONFIG, protocol=protocol)
    dynamic = load_e23_settings(WINDOW_V12, protocol=protocol)
    assert dynamic.variable_selection == "window_context_topk_v1"
    assert dynamic.config_sha256 != base.config_sha256
    assert dynamic.d_model == base.d_model == 64
    assert dynamic.stage1_epochs == dynamic.stage2_epochs == 100


def test_two_group_h_config_inherits_window_v_and_locked_training(tmp_path):
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    v12 = load_e23_settings(WINDOW_V12, protocol=protocol)
    v13 = load_e23_settings(TWO_GROUP_V13, protocol=protocol)
    assert v13.historical_retrieval == "two_group_visible_rerank_v1"
    assert v13.variable_selection == v12.variable_selection == "window_context_topk_v1"
    assert v13.d_model == v12.d_model == 64
    assert v13.stage1_epochs == v13.stage2_epochs == 100
    assert v13.config_sha256 != v12.config_sha256
    bad = tmp_path / "bad-h.yaml"
    bad.write_text(TWO_GROUP_V13.read_text().replace(
        "two_group_visible_rerank_v1", "unbounded_all_groups"))
    with pytest.raises(ValueError, match="historical retrieval"):
        load_e23_settings(bad, protocol=protocol)


def test_stronger_window_v_config_is_distinct_from_v12():
    protocol = load_e23_p4_protocol(
        catalog=CATALOG, protocol_config=PROTOCOL_CONFIG, mask_artifacts=MASKS,
    )
    v12 = load_e23_settings(WINDOW_V12, protocol=protocol)
    v15 = load_e23_settings(WINDOW_V15, protocol=protocol)
    assert v15.variable_selection == "window_context_topk_v2"
    assert v15.d_model == v12.d_model == 64
    assert v15.stage1_epochs == v15.stage2_epochs == 100
    assert v15.config_sha256 != v12.config_sha256
