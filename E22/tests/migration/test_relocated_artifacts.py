import hashlib
import json
from pathlib import Path

import pytest

from malr.migration import MigrationCatalog, load_migrated_run
from malr.cli import main
from integrations.forecast_compare.protocol import load_p4_protocol


def _row(source: Path, destination: str, data: bytes) -> dict:
    return {
        "source": str(source), "destination_relative": destination,
        "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "owner": "E22", "role": "asset", "dataset": "", "scenario": "", "run_id": "",
    }


def _catalog(tmp_path: Path, *, original: bytes = b"old", copied: bytes = b"new"):
    old = tmp_path / "old" / "checkpoint.pt"
    old.parent.mkdir()
    old.write_bytes(original)
    paper = tmp_path / "paper"
    new = paper / "E22" / "artifacts" / "checkpoint.pt"
    new.parent.mkdir(parents=True)
    new.write_bytes(copied)
    catalog_path = paper / "E22" / "migration" / "paper_catalog.jsonl"
    catalog_path.parent.mkdir()
    catalog_path.write_text(json.dumps(_row(old, "E22/artifacts/checkpoint.pt", copied)) + "\n")
    return old, new, catalog_path


def test_resolver_chooses_new_copy_even_if_old_exists(tmp_path: Path):
    old, new, catalog_path = _catalog(tmp_path)
    catalog = MigrationCatalog.load(catalog_path)
    assert catalog.resolve(old) == new
    assert old.read_bytes() == b"old"


def test_catalog_path_with_parent_segment_returns_canonical_destination(tmp_path: Path):
    old, new, catalog_path = _catalog(tmp_path)
    alias = catalog_path.parent / ".." / "migration" / catalog_path.name
    assert MigrationCatalog.load(alias).resolve(old) == new.resolve()
    assert ".." not in MigrationCatalog.load(alias).resolve(old).parts


def test_resolver_rejects_hash_or_path_mismatch(tmp_path: Path):
    old, new, catalog_path = _catalog(tmp_path)
    catalog = MigrationCatalog.load(catalog_path)
    with pytest.raises(ValueError, match="unmapped"):
        catalog.resolve(tmp_path / "old" / "missing.pt")
    with pytest.raises(ValueError, match="SHA-256"):
        catalog.resolve(old, expected_sha256="0" * 64)
    new.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256|size"):
        catalog.resolve(old)


def test_catalog_rejects_path_escape(tmp_path: Path):
    old, _, catalog_path = _catalog(tmp_path)
    row = _row(old, "E22/../escape.pt", b"x")
    catalog_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="destination"):
        MigrationCatalog.load(catalog_path)


def test_original_manifest_hash_is_unchanged(tmp_path: Path):
    old, new, catalog_path = _catalog(tmp_path)
    manifest = tmp_path / "old" / "run.json"
    payload = {"checkpoint": str(old), "checkpoint_sha256": hashlib.sha256(b"new").hexdigest()}
    manifest.write_text(json.dumps(payload))
    migrated_manifest = catalog_path.parent.parent / "artifacts" / "run.json"
    migrated_manifest.write_bytes(manifest.read_bytes())
    with catalog_path.open("a") as stream:
        stream.write(json.dumps(_row(manifest, "E22/artifacts/run.json", manifest.read_bytes())) + "\n")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    loaded, paths = load_migrated_run(manifest, MigrationCatalog.load(catalog_path))
    assert loaded == payload
    assert paths["checkpoint"] == new
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before


def test_selected_stage1_stage2_chain_resolves_to_new_copies():
    root = Path(__file__).resolve().parents[2]
    selection = json.loads((root / "migration/paper_selection.json").read_text())
    run = selection["runs"][0]
    manifest_path = root / run["manifest"].removeprefix("checkout:")
    catalog = MigrationCatalog.load(root / "migration/paper_catalog.jsonl")
    manifest, paths = load_migrated_run(manifest_path, catalog)
    assert paths["checkpoint"].is_file()
    assert paths["stage1_checkpoint"].is_file()
    assert paths["checkpoint"].is_relative_to(root)
    assert paths["stage1_checkpoint"].is_relative_to(root)
    assert manifest["run_id"] == run["run_id"]


def test_cli_inspects_selected_chain_and_masks_read_only(capsys, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    run = json.loads((root / "migration/paper_selection.json").read_text())["runs"][0]
    manifest_path = root / run["manifest"].removeprefix("checkout:")
    real_open = Path.open

    def forbid_old_checkout(self, *args, **kwargs):
        if "/missing_aware_latent_recovery/" in str(self):
            raise AssertionError(f"read old checkout: {self}")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_old_checkout)
    assert main([
        "inspect-migrated", "--migration-catalog", str(root / "migration/paper_catalog.jsonl"),
        "--run-manifest", str(manifest_path), "--mask-split", "validation",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] == run["run_id"]
    assert result["checkpoint_epoch"] >= 0
    assert result["mask_sha256"] == "cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c"


def test_p4_protocol_catalog_mode_uses_checked_copies():
    root = Path(__file__).resolve().parents[2]
    catalog = MigrationCatalog.load(root / "migration/paper_catalog.jsonl")
    protocol = load_p4_protocol(
        root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        root / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
        catalog=catalog,
    )
    assert protocol.source_sha256 == "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"


def test_documented_relative_manifest_inspection(monkeypatch, capsys):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    run = json.loads((root / "migration/paper_selection.json").read_text())["runs"][0]
    relative_manifest = run["manifest"].removeprefix("checkout:")
    assert main([
        "inspect-migrated", "--migration-catalog", "migration/paper_catalog.jsonl",
        "--run-manifest", relative_manifest, "--mask-split", "validation",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == run["run_id"]
