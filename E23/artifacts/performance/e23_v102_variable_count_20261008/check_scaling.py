"""Check old-weight compatibility and actual repair computation at C=7/21/64."""
import importlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

W = Path('/home/xyw/.cache/codex-worktrees/e23-v102-variable-count-20261008')
OLD = Path('/home/xyw/paper/E23/versions/e23-v100-stage1-validation-cache-20261008/E23')
R = Path(__file__).resolve().parent
sys.path[:0] = [str(W/'src'),str(W)]
import torch
from malr.models.e23.model import MultiExpertRepairEncoder
from malr.models.e23.recent_mean import RecentMeanProjector
from malr.data.patching import patchify_preserving_values
from malr.models.explicit_repair import build_explicit_repair

torch.set_num_threads(4)
device = torch.device('cuda:0')
pkg = types.ModuleType('legacy_e23_scaling')
pkg.__path__ = [str(OLD/'src/malr/models/e23')]
sys.modules[pkg.__name__] = pkg
Legacy = importlib.import_module('legacy_e23_scaling.recent_mean').RecentMeanProjector
scene_root = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_mean_and_dense_20261008/E23-v97/sr015_bl8/stage1')
selection = json.loads((scene_root/'runs/e23-stage1.json').read_text())
payload = torch.load(selection['checkpoint'], map_location='cpu', weights_only=False)
prefix = 'recent_mean_projector.'
state = {key[len(prefix):]:value for key,value in payload['model_state'].items() if key.startswith(prefix)}
old = Legacy(7,mode='global_dense_time_v4').eval()
new = RecentMeanProjector(7,mode='global_dense_time_v4').eval()
old.load_state_dict(state,strict=True)
new.load_state_dict(state,strict=True)
torch.manual_seed(2023)
values = torch.randn(3,96,7)
mask = torch.rand_like(values) > .73
visible = values.masked_fill(~mask,0)
patches = patchify_preserving_values(visible,mask,16,8)
explicit = build_explicit_repair(torch.randn_like(patches.values),patches.values,
                                patches.effective_mask,patches.validity,
                                torch.zeros(3,7),torch.ones(3,7),window_len=96,patch_len=16,stride=8)
args = (explicit, types.SimpleNamespace(latents=torch.randn(3,7,patches.values.shape[2],64)),
        torch.randn(21,64,96),torch.zeros(3,7),torch.ones(3,7))
with torch.no_grad():
    a,_ = old(*args)
    b,_ = new(*args)
assert torch.equal(a.normalized_sequence,b.normalized_sequence)
del old,new,payload

rows = []
for variables in [7,21,64]:
    for topk in [4,8]:
        torch.manual_seed(2023)
        model = MultiExpertRepairEncoder(variable_count=variables,window_length=96,patch_length=16,stride=8,
            d_model=64,retrieval_dim=64,historical_topk=4,historical_threshold=.7,variable_topk=topk,
            variable_offsets=(-1,0,1),variable_selection='window_context_topk_v1',
            temporal_local_radius=2,temporal_boundaries=(2,4,8,16),
            recovery_schedule='iterative_tv_h_v1',recovery_rounds=1,point_refiner_span=32,
            front_context='window_cnn_v1',point_router='pointwise_v1',overlap_fusion='confidence_v1',
            point_candidate_readout='eight_head_v1',front_variable_context='signature_plain_pos_v3',
            repair_coordinates='robust_local_v1',recent_mean_projection='global_dense_time_v4').to(device)
        # Seven variables have only six possible non-self sources: report clipping.
        batch_size=16
        values=torch.randn(batch_size,96,variables,device=device)
        mask=torch.rand_like(values)>.73
        mask[1]=False
        visible=values.masked_fill(~mask,0)
        mean=torch.zeros(batch_size,variables,device=device)
        std=torch.ones_like(mean)
        count=model.patch_count
        starts=torch.arange(count,device=device)[None,:].expand(batch_size,-1)*8
        timings=[]
        torch.cuda.reset_peak_memory_stats(device)
        for repeat in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started=time.perf_counter()
            output=model(visible,mask,mean,std,starts,epoch=1,explore_variable=False,specialist_exposure=0)
            sequence=output.explicit.normalized_sequence
            assert torch.equal(sequence[mask],visible[mask])
            loss=(sequence-values).square()[~mask].mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter()-started)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        evidence=output.evidence['variable'].candidates
        dense=model.recent_mean_projector.global_dense_refiner
        row=dict(variables=variables,configured_variable_topk=topk,
                 effective_variable_topk=model.variable.topk,
                 variable_candidate_slots=evidence.messages.shape[-2],
                 dense_parameters=dense.parameter_count,repair_parameters=sum(p.numel() for p in model.parameters()),
                 batch_size=batch_size,step_ms_median=statistics.median(timings[1:])*1000,
                 peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                 observations_unchanged=True,finite_outputs_and_gradients=True,
                 scope='synthetic full repair forward/backward; no optimizer; empty H bank; not training accuracy')
        rows.append(row)
        print(json.dumps(row),flush=True)
        del model,output,evidence,loss,sequence,values,visible,mask
        torch.cuda.empty_cache()
result=dict(seven_variable_saved_v97_suffix_state_strict_load=True,
            seven_variable_trained_suffix_outputs_exact_equal=True,
            source_checkpoint_sha256=selection['checkpoint_sha256'],
            device='cuda:0',rows=rows,formal_training=False,test_used=False)
(R/'scaling_checks.json').write_text(json.dumps(result,indent=2)+'\n')
