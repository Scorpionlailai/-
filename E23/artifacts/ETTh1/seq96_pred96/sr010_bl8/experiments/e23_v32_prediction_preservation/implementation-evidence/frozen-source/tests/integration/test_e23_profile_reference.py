from dataclasses import replace
from pathlib import Path

import pytest
import torch

from integrations.e23.config import load_e23_p4_protocol
from integrations.e23.profile_reference import ProfileTarget, profile_selected_reference
from malr.migration import MigrationCatalog


ROOT = Path(__file__).resolve().parents[2]
CATALOG = MigrationCatalog.load(ROOT.parent / "E22/migration/paper_catalog.jsonl", project="E23")
MANIFEST = CATALOG.paper_root / "SAITS/artifacts/ETTh1/seq96_pred96/sr010_bl8/saits/stage2_p4_v2/runs/saits-stage2.json"
MASKS = ROOT.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"


def test_profile_rejects_wrong_mask_and_preserves_memory():
    protocol = load_e23_p4_protocol(
        catalog=CATALOG,
        protocol_config=ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=MASKS,
    )
    calls = []
    memory = torch.tensor([3.0])

    def run_batch(batch):
        calls.append(batch.normalized_observed.shape[0])
        return batch.normalized_observed.mean()

    target = ProfileTarget("synthetic", run_batch, lambda: memory.clone())
    wrong = replace(protocol, mask_sha256={**protocol.mask_sha256, "test": "0" * 64})
    with pytest.raises(ValueError, match="mask"):
        profile_selected_reference(
            protocol=wrong, selected_manifest=MANIFEST, catalog=CATALOG,
            target=target, device=torch.device("cpu"), warmup_batches=1, batches=2,
        )
    assert calls == []

    report = profile_selected_reference(
        protocol=protocol, selected_manifest=MANIFEST, catalog=CATALOG,
        target=target, device=torch.device("cpu"), warmup_batches=1, batches=2,
    )
    assert calls == [128, 128, 128]
    assert torch.equal(memory, torch.tensor([3.0]))
    assert report["warmup_batches"] == 1
    assert report["timed_batches"] == 2
    assert report["split"] == "validation"
    assert report["cold_start_seconds"] >= 0
    assert report["pipeline_seconds"] >= report["model_seconds"] >= 0
    assert report["throughput_windows_per_second"] > 0
    assert report["peak_memory_bytes"] >= 0
    assert len(report["selection_sha256"]) == 64

    test_report = profile_selected_reference(
        protocol=protocol, selected_manifest=MANIFEST, catalog=CATALOG,
        target=target, device=torch.device("cpu"), warmup_batches=1, batches=2,
        split="test",
    )
    assert test_report["split"] == "test"
    assert calls == [128] * 6


def test_profile_rejects_stage1_reference_before_execution():
    protocol = load_e23_p4_protocol(
        catalog=CATALOG,
        protocol_config=ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=MASKS,
    )
    called = []

    def run_batch(batch):
        called.append(True)
        return batch.normalized_observed.mean()

    stage1 = CATALOG.paper_root / "SAITS/artifacts/ETTh1/seq96_pred96/sr010_bl8/saits/stage1_p4_v2/runs/saits-stage1.json"
    with pytest.raises(ValueError, match="forecast Stage 2"):
        profile_selected_reference(
            protocol=protocol, selected_manifest=stage1, catalog=CATALOG,
            target=ProfileTarget("synthetic", run_batch, lambda: torch.zeros(1)),
            device=torch.device("cpu"), warmup_batches=0, batches=1,
        )
    assert called == []
