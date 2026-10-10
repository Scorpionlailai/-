import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
W=Path('/home/xyw/.cache/codex-worktrees/e23-v125-shared-calibration-20261009')
sys.path[:0]=[str(W/'src'),str(W)]
from integrations.e23.config import load_e23_p4_protocol
from malr.migration import MigrationCatalog
def protocol_for(plan,scene):
 os.environ.pop('MALR_MISSING_SCENARIO',None)
 if scene.get('scenario_manifest'):os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
 previous=os.getcwd()
 try:
  os.chdir('/home/xyw/paper/E23')
  return load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
 finally:os.chdir(previous)
