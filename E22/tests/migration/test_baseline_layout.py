from pathlib import Path
import subprocess

import pytest

from malr.migration import MigrationCatalog
from scripts.paper_migration.baseline import verify_baseline


GIT = "/home/xyw/miniforge3/envs/malr/bin/git"
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "migration/paper_catalog.jsonl"


def test_baseline_rejects_wrong_upstream_commit(tmp_path: Path):
    subprocess.run([GIT, "init", "-q", "-b", "test", str(tmp_path / "wrong")], check=True)
    project = tmp_path / "wrong"
    subprocess.run([GIT, "-C", str(project), "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "--allow-empty", "-m", "wrong"], check=True)
    with pytest.raises(ValueError, match="upstream commit"):
        verify_baseline("SAITS", project=project, catalog=MigrationCatalog.load(CATALOG))


def test_s4m_overlay_does_not_change_protected_semantics(tmp_path: Path):
    source = ROOT.parent / "S4M"
    project = tmp_path / "s4m"
    subprocess.run([GIT, "clone", "--local", "-q", str(source), str(project)], check=True)
    protected = project / "s4m/model/S4M.py"
    protected.write_bytes(protected.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="protected S4M"):
        verify_baseline("S4M", project=project, catalog=MigrationCatalog.load(CATALOG))


def test_selected_baselines_have_checked_bridge_and_evaluations():
    catalog = MigrationCatalog.load(CATALOG)
    for name in ("SAITS", "S4M"):
        report = verify_baseline(name, project=ROOT.parent / name, catalog=catalog)
        assert report.portable
        assert report.bridge_hash and report.selected_evaluation_hash


def test_s4m_approved_patch_is_installed():
    project = ROOT.parent / "S4M"
    patch = project / "bridge/integrations/s4m/patches/s4m-malr-fair.patch"
    result = subprocess.run(
        [GIT, "-C", str(project), "apply", "--reverse", "--ignore-space-change", "--check", str(patch)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_s4m_bridge_preflight_accepts_migrated_commit():
    project = ROOT.parent / "S4M"
    result = subprocess.run(
        ["/home/xyw/miniforge3/envs/malr-s4m/bin/python", "-c",
         "from pathlib import Path; from integrations.forecast_compare.s4m_p4 import _verify_s4m_checkout; print(_verify_s4m_checkout(Path('.'))['commit'])"],
        cwd=project, env={"PYTHONPATH": "bridge:../E22/src"}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "a718823addd3606e763dfc261174e0135b2535f4" in result.stdout


def test_baseline_rejects_s4m_clone_without_reviewed_patch(tmp_path: Path):
    project = tmp_path / "s4m"
    subprocess.run([GIT, "clone", "--local", "-q", str(ROOT.parent / "S4M"), str(project)], check=True)
    first_bridge = subprocess.check_output(
        [GIT, "-C", str(project), "rev-list", "--reverse",
         "a718823addd3606e763dfc261174e0135b2535f4..HEAD"], text=True,
    ).splitlines()[0]
    subprocess.run([GIT, "-C", str(project), "checkout", "-q", first_bridge], check=True)
    with pytest.raises(ValueError, match="reviewed S4M patch"):
        verify_baseline("S4M", project=project, catalog=MigrationCatalog.load(CATALOG))
