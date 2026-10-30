"""Self-contained, repository-relative frozen implementation bootstrap."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
SNAP=ROOT/'vendor'
sys.path[:0]=[str(SNAP/'E24/src'),str(SNAP/'E24')]
from integrations.v125_ablation.adapter import V125Adapter

def verify_sources():
    for name,h in json.loads((ROOT/'vendor_lock.json').read_text())['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('vendored source changed: '+name)

def strip_postprocessing(model):
    model.core.point_refiner=None;model.core.recent_mean_projector=None
    model.coarse_projector=None;model.inference_path='coarse'
    return model
