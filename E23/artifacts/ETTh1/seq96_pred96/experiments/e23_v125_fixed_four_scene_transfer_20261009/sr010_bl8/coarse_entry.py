import sys,os
from pathlib import Path
S=Path('/home/xyw/.cache/codex-worktrees/e23-v125-shared-calibration-20261009')
sys.path[:0]=[str(S/'src'),str(S)]
os.chdir('/home/xyw/paper/E23')
from integrations.e23 import run
assert Path(run.__file__).resolve().is_relative_to(S),run.__file__
if __name__=='__main__':raise SystemExit(run.main())
