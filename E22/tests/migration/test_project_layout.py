from pathlib import Path

import pytest

from scripts.paper_migration.layout import verify_project_layout


def _base(tmp_path: Path) -> Path:
    root = tmp_path / "paper"
    (root / "data/raw").mkdir(parents=True)
    for name in ("E22", "E23"):
        project = root / name
        (project / "src/malr").mkdir(parents=True)
        (project / "src/malr/__init__.py").write_text("", encoding="utf-8")
        (project / "configs").mkdir()
        (project / "data").symlink_to("../data", target_is_directory=True)
    return root


def test_layout_rejects_windows_gitdir_and_old_absolute_data_link(tmp_path: Path) -> None:
    root = _base(tmp_path)
    e22 = root / "E22"
    (e22 / ".git").write_text("gitdir: C:/Users/Administrator/old/.git\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Windows gitdir"):
        verify_project_layout(root)
    (e22 / ".git").unlink()
    (e22 / "data").unlink()
    (e22 / "data").symlink_to("/home/xyw/missing_aware_latent_recovery/data", target_is_directory=True)
    with pytest.raises(ValueError, match="relative data link"):
        verify_project_layout(root)


def test_e23_has_source_but_no_e22_artifacts(tmp_path: Path) -> None:
    root = _base(tmp_path)
    artifact = root / "E23/artifacts/old.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"old checkpoint")
    with pytest.raises(ValueError, match="E23 artifacts"):
        verify_project_layout(root)
    artifact.unlink()
    report = verify_project_layout(root)
    assert report.projects == ("E22", "E23")
    assert report.data_root == root / "data"
