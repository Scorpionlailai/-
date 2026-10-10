import numpy as np
import pytest
import torch

from malr.models.e23.historical_context import context_windows, input_context
from malr.models.e23.model import MultiExpertRepairEncoder
from malr.prototypes.online_bank import OnlinePrototypeMemory
from malr.prototypes.tensor_index import TensorPrototypeIndex


def contextual_memory():
    memory = OnlinePrototypeMemory(patch_len=4, patch_stride=2, train_end=100,
                                    variable_count=1)
    context = np.arange(12, dtype=float)
    memory.stage_patch(0, 16, context[4:8], [1., 0.], observed=True,
                       context_values=context, context_mask=np.ones(12, bool))
    memory.publish_epoch(1)
    return memory


def test_member_context_roundtrip_rekey_and_full_evidence_cutoff():
    memory = contextual_memory()
    memory.rekey_batch(lambda members: np.array([[0., 1.] for _ in members]))
    restored = OnlinePrototypeMemory(patch_len=4, patch_stride=2, train_end=100,
                                     variable_count=1)
    restored.load_state_dict(memory.state_dict())
    assert restored.snapshot_hash() == memory.snapshot_hash()
    member = restored.snapshot().groups[0][0].members[0]
    assert member.evidence_end == 24
    assert not member.context_values.flags.writeable
    assert not member.context_mask.flags.writeable
    assert restored.query(0, 22, [0., 1.]) == ()
    assert restored.query(0, 24, [0., 1.])[0].identity == (0, 16)
    index = TensorPrototypeIndex.from_snapshot(restored.snapshot(), device=torch.device('cpu'))
    keys = torch.tensor([[[[0., 1.], [0., 1.]]]])
    result = index.query(keys, torch.tensor([[22, 24]]), torch.ones(1,1,2,dtype=torch.bool))
    assert not result.valid[0,0,0].any()
    assert result.valid[0,0,1].any()


def test_hidden_context_cannot_be_stored_or_cross_training_boundary():
    memory = OnlinePrototypeMemory(patch_len=4, patch_stride=2, train_end=24,
                                    variable_count=1)
    context = np.arange(12,dtype=float); mask = np.ones(12,bool); mask[0] = False
    context[0] = 123.
    with pytest.raises(ValueError, match='zero missing'):
        memory.stage_patch(0,16,context[4:8],[1.,0.],observed=True,
                           context_values=context,context_mask=mask)
    with pytest.raises(ValueError,match='training boundary'):
        memory.stage_patch(0,18,context[4:8],[1.,0.],observed=True,
                           context_values=np.arange(12,dtype=float),context_mask=np.ones(12,bool))


def test_non_axis_keys_keep_exact_snapshot_hash_after_loading():
    memory=OnlinePrototypeMemory(patch_len=4,patch_stride=2,train_end=100,variable_count=1)
    memory.stage_patch(0,16,np.arange(4,dtype=float),[.1,.3,.6,.9],observed=True)
    memory.publish_epoch(1)
    restored=OnlinePrototypeMemory(patch_len=4,patch_stride=2,train_end=100,variable_count=1)
    restored.load_state_dict(memory.state_dict())
    assert restored.snapshot_hash()==memory.snapshot_hash()


def test_context_inputs_ignore_hidden_values_and_zero_pad():
    values = torch.arange(32.).reshape(1,32,1).requires_grad_()
    mask = torch.ones_like(values,dtype=torch.bool); mask[:,8:24] = False
    a, am = context_windows(values,mask,patch_len=16,stride=8,patches=4)
    changed = values.detach().clone(); changed[~mask] = 999.
    b, bm = context_windows(changed,mask,patch_len=16,stride=8,patches=4)
    assert torch.equal(a,b) and torch.equal(am,bm)
    assert not am[:,:,0,:16].any()
    assert not bm[:,:,-1,24:].any()
    a.sum().backward()
    assert torch.equal(values.grad[~mask],torch.zeros_like(values.grad[~mask]))


def test_contextual_query_can_retrieve_for_fully_hidden_patch():
    torch.manual_seed(2021)
    model = MultiExpertRepairEncoder(1,32,16,8,64,64,
        variable_selection='window_context_topk_v1',recovery_schedule='iterative_tv_h_v1',
        historical_query_mode='global_context_v1')
    values = torch.sin(torch.arange(32.)).reshape(1,32,1)
    mask = torch.ones_like(values,dtype=torch.bool); mask[:,8:24] = False
    mean, std = torch.zeros(1,1),torch.ones(1,1)
    starts = torch.tensor([[100,108,116,124]])
    out = model(values,mask,mean,std,starts)
    from malr.data.patching import patchify_with_state
    from malr.models.e23.contracts import ExpertInputs
    p = patchify_with_state(values,mask,16,8)
    inputs = ExpertInputs(out.base_latents,p.values.masked_fill(~p.effective_mask,0),
        p.effective_mask,p.validity,p.rho,starts,mean,std,p.rho.lt(1))
    contexts, context_masks = input_context(inputs,patch_len=16,stride=8)
    # Use a complete observed center with matching real-visible neighbors.
    stored = contexts[0,0,1].clone(); sm = context_masks[0,0,1].clone()
    sm[16:32] = True; stored[16:32] = 0
    key = model.historical.retrieval.encode_frozen(stored[None],sm[None])[0].numpy()
    memory = OnlinePrototypeMemory(patch_len=16,patch_stride=8,train_end=100,variable_count=1)
    memory.stage_patch(0,32,np.zeros(16),key,observed=True,
                       context_values=stored.numpy(),context_mask=sm.numpy())
    memory.publish_epoch(1); model.historical.refresh_snapshot(memory.snapshot())
    # Availability is a threshold-dependent choice; force the diagnostic threshold.
    model.historical.similarity_threshold = -1
    cache = model.historical.prepare_candidates(inputs)
    assert not p.effective_mask[0,0,1].any()
    assert cache.available[0,0,1]
    with pytest.raises(ValueError,match='patch-only'):
        old = OnlinePrototypeMemory(patch_len=16,patch_stride=8,train_end=100,variable_count=1)
        old.stage_patch(0,32,np.zeros(16),key,observed=True); old.publish_epoch(1)
        model.historical.refresh_snapshot(old.snapshot())
