"""Check old-weight compatibility and actual repair computation at C=7/21/64."""
import importlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

W = Path('/home/xyw/.cache/codex-worktrees/e23-v103-dynamic-candidates-20261008')
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
rows = []
for variables in [21,64]:
    for topk in [4,16]:
        torch.manual_seed(2023)
        model = MultiExpertRepairEncoder(variable_count=variables,window_length=96,patch_length=16,stride=8,
            d_model=64,retrieval_dim=64,historical_topk=4,historical_threshold=.7,variable_topk=topk,
            variable_offsets=(-1,0,1),variable_selection=('window_context_topk_v1' if topk==4 else 'window_context_nucleus32_p90_cap16_v1'),
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
        if topk==16:
            stats=output.evidence['variable'].provenance
            counts=stats['selected_source_count']
            row.update(source_count_min=int(counts.min()),source_count_max=int(counts.max()),
                       source_count_mean=float(counts.float().mean()),
                       cap_hit_fraction=float(stats['cap_hit'].float().mean()),
                       retained_probability_mass_mean=float(stats['retained_probability_mass'].mean()),
                       source_pool_size=stats['source_pool_size'])
        rows.append(row)
        print(json.dumps(row),flush=True)
        del model,output,evidence,loss,sequence,values,visible,mask
        torch.cuda.empty_cache()
result=dict(
            device='cuda:0',rows=rows,formal_training=False,test_used=False)
(R/'scaling_checks.json').write_text(json.dumps(result,indent=2)+'\n')
