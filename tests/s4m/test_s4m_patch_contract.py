from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from integrations.s4m.verify_upstream import patch_paths


PATCH = Path("integrations/s4m/patches/s4m-malr-fair.patch")
ALLOWED = {
    "s4m/run.py",
    "s4m/data_provider/data_factory.py",
    "s4m/experiments/exp_pretrain1.py",
}


def test_official_patch_changes_only_declared_entry_files() -> None:
    assert patch_paths(PATCH) == ALLOWED


@pytest.mark.skipif(
    not os.getenv("S4M_OFFICIAL_ROOT"), reason="official checkout not bootstrapped"
)
def test_official_patch_is_applied_and_python_syntax_is_valid() -> None:
    root = Path(os.environ["S4M_OFFICIAL_ROOT"]).resolve()
    assert (root / "s4m" / "malr_fair" / "data.py").is_file()
    assert not (root / "s4m" / "malr_fair" / "__pycache__").exists()
    reverse = subprocess.run(
        ["git", "-C", str(root), "apply", "--reverse", "--check", str(PATCH.resolve())],
        text=True,
        capture_output=True,
    )
    assert reverse.returncode == 0, reverse.stderr
    targets = [root / path for path in sorted(ALLOWED)]
    syntax_check = (
        "import ast,pathlib,sys;"
        "[ast.parse(pathlib.Path(p).read_text(encoding='utf-8'),filename=p) "
        "for p in sys.argv[1:]]"
    )
    compiled = subprocess.run(
        [sys.executable, "-c", syntax_check, *map(str, targets)],
        text=True,
        capture_output=True,
    )
    assert compiled.returncode == 0, compiled.stderr
