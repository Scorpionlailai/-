import json
from pathlib import Path

import pytest
import torch
from torch import nn

from integrations.forecast_compare import saits_run
from integrations.forecast_compare.protocol import P4Batch
from malr.config import load_config


class TinySaits(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs, stage):
        # One optimizer step per epoch moves the parameter above zero.
        loss = (self.weight - 1).square()
        return {"reconstruction_loss": loss, "imputation_loss": loss,
                "imputed_data": torch.zeros_like(inputs["X"]) + self.weight}


def tiny_batch():
    observed = torch.tensor([[[True], [False]]])
    return P4Batch(torch.zeros(1, 2, 1), torch.zeros(1, 2, 1), observed,
                   torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1, 1))


class TinyProtocol:
    source_sha256 = "csv"
    mask_sha256 = {"validation": "valmask", "test": "testmask"}
    config_path = Path("configs/stage1_seq96_dynamic_temporal_repair.yaml")
    config = load_config(config_path)


def test_repair_checkpoint_uses_lowest_validation_imputation_mae(monkeypatch, tmp_path):
    monkeypatch.setattr(saits_run, "build_official_saits", lambda *a, **kw: TinySaits())
    monkeypatch.setattr(saits_run, "iter_p4_batches", lambda p, split, **kw: iter([tiny_batch()]))
    monkeypatch.setattr(saits_run, "_repair_validation_mae", lambda *a, **kw: [0.4, 0.2, 0.3][kw["epoch"]])
    manifest = saits_run.train_repair(TinyProtocol(), output=tmp_path,
                                      repo_root=tmp_path, device=torch.device("cpu"), epochs=3)
    record = json.loads(manifest.read_text())
    assert record["best_epoch"] == 2
    assert record["best_val_imputation_mae"] == 0.2
    assert len(record["checkpoint_sha256"]) == 64


def test_repair_training_never_asks_for_test_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(saits_run, "build_official_saits", lambda *a, **kw: TinySaits())

    def batches(_protocol, split, **_):
        if split == "test":
            raise AssertionError("test batches must not be opened")
        return iter([tiny_batch()])

    monkeypatch.setattr(saits_run, "iter_p4_batches", batches)
    saits_run.train_repair(TinyProtocol(), output=tmp_path, repo_root=tmp_path,
                           device=torch.device("cpu"), epochs=1)


def test_repair_progress(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(saits_run, "build_official_saits", lambda *a, **kw: TinySaits())
    monkeypatch.setattr(saits_run, "iter_p4_batches", lambda p, split, **kw: iter([tiny_batch()]))
    saits_run.train_repair(TinyProtocol(), output=tmp_path, repo_root=tmp_path,
                           device=torch.device("cpu"), epochs=1)
    output = capsys.readouterr().out
    assert "epoch=1/1" in output
    assert "batch=1/1" in output
    assert "val_imputation_mae=" in output


def test_repair_manifest_binds_protocol(monkeypatch, tmp_path):
    monkeypatch.setattr(saits_run, "build_official_saits", lambda *a, **kw: TinySaits())
    monkeypatch.setattr(saits_run, "iter_p4_batches", lambda p, split, **kw: iter([tiny_batch()]))
    record = json.loads(saits_run.train_repair(TinyProtocol(), output=tmp_path,
                        repo_root=tmp_path, device=torch.device("cpu"), epochs=1).read_text())
    assert record["source_csv_sha256"] == "csv"
    assert record["validation_mask_sha256"] == "valmask"
    assert record["settings"]["d_time"] == 96
    assert record["stage"] == "repair"
    assert len(record["model_config_sha256"]) == 64
    assert len(record["adapter_code_sha256"]) == 64
