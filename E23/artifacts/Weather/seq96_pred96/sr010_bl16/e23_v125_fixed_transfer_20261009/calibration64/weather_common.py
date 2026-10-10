import sys,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
W=Path('/home/xyw/.cache/codex-worktrees/e23-v125-shared-calibration-20261009')
sys.path[:0]=[str(W/'src'),str(W)]
spec=importlib.util.spec_from_file_location('readonly_weather_data','/home/xyw/paper/E24/integrations/e24_sr010_bl16/data.py')
data=importlib.util.module_from_spec(spec);sys.modules[spec.name]=data;spec.loader.exec_module(data)
def protocol_for(plan,scene):
 manifest=Path(plan['dataset_manifest']);record=json.loads(manifest.read_text())
 return data.load_dataset_protocol(manifest,config_path=Path(record['config_path']),artifacts=manifest.parent)
