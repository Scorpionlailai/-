"""E6 strong context capacity control; E7 summary after original R3 selection."""
import variants
from aligned_h_model import make_repair as make_r3
from variants import ContextAdapter,R2SummaryReadout

def make_repair(data,variant,seed=2023):
 if variant not in ['E0','E6','E7']:raise ValueError(variant)
 model=make_r3(data,'H_GLOBAL',seed)
 if variant=='E6':
  model=ContextAdapter(data,model.core,seed)
  model.disabled=frozenset(['historical','variable','temporal'])
  assert model.core.capacity_hidden>0  # Dataset-specific analytic parameter matching, validated within 1% before training.
 elif variant=='E7':model.core.point_candidate_readout.__class__=R2SummaryReadout
 return model

def effective_parameters(full):
 repair=full.frozen.repair
 if isinstance(repair,ContextAdapter):
  # H/V/T alternatives, weights and available columns are constant zero (3x3).
  return sum(p.numel() for p in full.parameters())-9*repair.core.overlap_confidence.score[0].out_features
 return sum(p.numel() for n,p in full.named_parameters() if not n.startswith(('frozen.repair.ema_encoder.','frozen.repair.core.historical.key_encoder.')))

def install():variants.make_repair=make_repair
