from pathlib import Path
import json

import pytest

from integrations.forecast_compare.s4m_p4 import (
    build_s4m_p4_commands, preflight_s4m_p4, require_s4m_selection,
    s4m_subprocess_environment,
)
from integrations.forecast_compare.protocol import load_p4_protocol


CONFIG = Path("configs/stage1_seq96_dynamic_temporal_repair.yaml")
MASKS = Path("../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context")
S4M = Path("/home/xyw/missing_aware_latent_recovery/external/S4M")


def test_s4m_p4_preflight():
    actual = preflight_s4m_p4(config=CONFIG, mask_artifacts=MASKS, s4m_root=S4M)
    common = load_p4_protocol(CONFIG, MASKS)
    assert actual["upstream_commit"] == "a718823addd3606e763dfc261174e0135b2535f4"
    assert actual["source_csv_sha256"] == common.source_sha256
    assert actual["test_mask_sha256"] == common.mask_sha256["test"]
    assert actual["validation_mask_sha256"] == common.mask_sha256["validation"]
    assert actual["validation_windows"] == actual["test_windows"] == 2688
    assert actual["test_refs_sha256"] == actual["common_test_refs_sha256"]


def test_s4m_rejects_old_bl5_masks():
    old = Path("artifacts/ETTh1/seq96_pred96/sr010_bl5/experiments")
    with pytest.raises(ValueError):
        preflight_s4m_p4(config=CONFIG, mask_artifacts=old, s4m_root=S4M)


def test_s4m_commands_use_official_100_epoch_profile(tmp_path):
    smoke, search, missing, clean = build_s4m_p4_commands(
        config=CONFIG, mask_artifacts=MASKS, s4m_root=S4M,
        output=tmp_path / "new-p4", python="/python")
    assert "--smoke" in smoke
    assert search[search.index("--profile") + 1] == "official"
    assert search[search.index("--train-epochs") + 1] == "100"
    assert search[search.index("--patience") + 1] == "100"
    assert search[search.index("--artifacts") + 1] == str(MASKS.resolve())
    assert missing[missing.index("--selection") + 1].endswith("selection-official.json")
    assert clean[clean.index("--selection") + 1].endswith("selection-official-clean.json")


def test_s4m_test_only_after_selection(tmp_path):
    with pytest.raises(ValueError, match="selection"):
        require_s4m_selection(tmp_path, variant="missing")
    (tmp_path / "selection-official.json").write_text(json.dumps({
        "selected_run_id": "chosen", "checkpoint_sha256": "a" * 64,
        "selected_manifest": "chosen.json", "test_metrics_seen": False,
    }))
    assert require_s4m_selection(tmp_path, variant="missing").name == "selection-official.json"
    with pytest.raises(ValueError, match="selection"):
        require_s4m_selection(tmp_path, variant="clean")


def test_s4m_progress_is_unbuffered_and_gpu_order_stable():
    result = s4m_subprocess_environment({"CUDA_VISIBLE_DEVICES": "1"})
    assert result["PYTHONUNBUFFERED"] == "1"
    assert result["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert result["CUDA_VISIBLE_DEVICES"] == "1"
