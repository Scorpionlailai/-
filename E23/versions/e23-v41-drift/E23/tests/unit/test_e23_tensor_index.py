import numpy as np
import pytest
import torch

from malr.prototypes.online_bank import (
    OnlinePrototypeGroup, OnlinePrototypeMember, OnlinePrototypeMemory,
    OnlinePrototypeSnapshot,
)
from malr.prototypes.tensor_index import TensorPrototypeIndex


def _memory():
    return OnlinePrototypeMemory(
        patch_len=4, patch_stride=2, train_end=128, variable_count=2,
        group_similarity_threshold=0.5,
    )


def _member(variable, start, key):
    return OnlinePrototypeMember(variable, start, np.full(4, start, dtype=np.float64), np.array(key))


def _query(index, keys, starts, damaged):
    return index.query(
        torch.tensor(keys, dtype=torch.float64),
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(damaged, dtype=torch.bool), topk=4,
    )


def test_tensor_index_matches_online_bank_query_batch_on_ties_and_empty_groups():
    memory = _memory()
    for variable, start, key in (
        (0, 0, [1, 0]), (0, 4, [1, 0]), (0, 8, [0, 1]),
    ):
        member = _member(variable, start, key)
        memory.stage_patch(variable, start, member.raw_patch, member.clean_key, observed=True)
    snapshot = memory.publish_epoch(1)
    index = TensorPrototypeIndex.from_snapshot(snapshot, device=torch.device("cpu"))
    keys = np.array([[[[1., 0.], [0., 1.]], [[1., 0.], [1., 0.]]]])
    starts = np.array([[8, 12]])
    damaged = np.ones((1, 2, 2), dtype=bool)
    result = _query(index, keys, starts, damaged)
    ids = np.array([0, 0, 1, 1])
    legacy = memory.query_batch(ids, np.tile(starts, (1, 2)).reshape(-1),
                                keys.reshape(-1, 2), max_items=4)
    for position, row in enumerate(legacy):
        got = result.starts.reshape(-1, 4)[position][result.valid.reshape(-1, 4)[position]].tolist()
        assert got == [member.start for member in row]
    assert not result.valid[:, 1].any()
    assert result.raw_patches.device.type == "cpu"


def test_tensor_index_full_center_selects_group_then_filters_future_member():
    memory = _memory()
    group_a = OnlinePrototypeGroup(0, 0, (
        _member(0, 0, [0.6, 0.8]), _member(0, 80, [1.0, 0.0]),
    ))
    group_b = OnlinePrototypeGroup(0, 1, (_member(0, 4, [0.8, -0.6]),))
    memory._snapshot = OnlinePrototypeSnapshot(1, ((group_a, group_b), ()),
                                                frozenset({(0, 0), (0, 80), (0, 4)}), {})
    index = TensorPrototypeIndex.from_snapshot(memory.snapshot(), device=torch.device("cpu"))
    result = _query(index, [[[[1., 0.]]]], [[16]], [[[True]]])
    legacy = memory.query_batch([0], [16], [[1., 0.]])[0]
    assert [member.start for member in legacy] == [0]
    assert result.starts[0, 0, 0][result.valid[0, 0, 0]].tolist() == [0]
    assert result.group_support[0, 0, 0, 0] == 2
    assert 80 not in result.starts[0, 0, 0][result.valid[0, 0, 0]].tolist()


def test_tensor_index_empty_bank_and_undamaged_queries():
    memory = _memory()
    index = TensorPrototypeIndex.from_snapshot(memory.snapshot(), device=torch.device("cpu"))
    result = _query(index, np.ones((1, 2, 3, 2)), [[4, 8, 12]], np.ones((1, 2, 3), dtype=bool))
    assert not result.valid.any()

    member = _member(0, 0, [1, 0])
    memory.stage_patch(0, 0, member.raw_patch, member.clean_key, observed=True)
    index = TensorPrototypeIndex.from_snapshot(memory.publish_epoch(1), device=torch.device("cpu"))
    result = _query(index, np.ones((1, 2, 3, 2)), [[4, 8, 12]], np.zeros((1, 2, 3), dtype=bool))
    assert not result.valid.any()


def test_tensor_index_matches_legacy_across_variables_and_causal_starts():
    random = np.random.default_rng(13)
    memory = _memory()
    for variable in range(2):
        for start in range(40):
            memory.stage_patch(variable, start, random.normal(size=4),
                               random.normal(size=8), observed=True)
    index = TensorPrototypeIndex.from_snapshot(memory.publish_epoch(1), device=torch.device("cpu"))
    keys = random.normal(size=(2, 2, 3, 8))
    starts = np.array([[0, 12, 28], [8, 20, 48]], dtype=np.int64)
    damaged = random.uniform(size=(2, 2, 3)) > 0.2
    result = _query(index, keys, starts, damaged)
    for batch in range(2):
        for variable in range(2):
            for patch in range(3):
                actual = result.starts[batch, variable, patch][
                    result.valid[batch, variable, patch]
                ].tolist()
                expected = ([member.start for member in memory.query_batch(
                    [variable], [starts[batch, patch]],
                    [keys[batch, variable, patch]], max_items=4,
                )[0]] if damaged[batch, variable, patch] else [])
                assert actual == expected

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tensor_index_cuda_matches_cpu_identity():
    memory = _memory()
    for start in (0, 4, 8):
        member = _member(0, start, [1, 0])
        memory.stage_patch(0, start, member.raw_patch, member.clean_key, observed=True)
    snapshot = memory.publish_epoch(1)
    keys = torch.tensor([[[[1., 0.]]]], dtype=torch.float32)
    starts = torch.tensor([[16]])
    damaged = torch.tensor([[[True]]])
    cpu = TensorPrototypeIndex.from_snapshot(snapshot, device=torch.device("cpu")).query(keys, starts, damaged)
    gpu = TensorPrototypeIndex.from_snapshot(snapshot, device=torch.device("cuda")).query(
        keys.cuda(), starts.cuda(), damaged.cuda()
    )
    torch.testing.assert_close(cpu.starts, gpu.starts.cpu())
    torch.testing.assert_close(cpu.valid, gpu.valid.cpu())
    torch.testing.assert_close(cpu.key_similarity, gpu.key_similarity.cpu(), atol=1e-6, rtol=1e-6)


def test_two_group_query_uses_stable_causal_same_variable_shortlist():
    first = OnlinePrototypeGroup(0, 0, tuple(_member(0, start, [1, 0])
                                             for start in (0, 2, 4, 6, 8)))
    second = OnlinePrototypeGroup(0, 1, tuple(_member(0, start, [1, 0])
                                              for start in (10, 12, 14, 16)))
    future = OnlinePrototypeGroup(0, 2, (_member(0, 40, [1, 0]),))
    other = OnlinePrototypeGroup(1, 3, (_member(1, 0, [1, 0]),))
    snapshot = OnlinePrototypeSnapshot(1, ((first, second, future), (other,)),
                                       frozenset(), {})
    index = TensorPrototypeIndex.from_snapshot(snapshot, device=torch.device("cpu"))
    result = index.query_two_groups(
        torch.tensor([[[[1., 0.]], [[1., 0.]]]], dtype=torch.float32),
        torch.tensor([[20]]), torch.ones(1, 2, 1, dtype=torch.bool), member_topk=4,
    )
    assert result.valid.shape == (1, 2, 1, 8)
    assert result.starts[0, 0, 0][result.valid[0, 0, 0]].tolist() == [0, 2, 4, 6, 10, 12, 14, 16]
    assert result.group_ids[0, 0, 0][result.valid[0, 0, 0]].tolist() == [0]*4 + [1]*4
    assert result.starts[0, 1, 0][result.valid[0, 1, 0]].tolist() == [0]
    assert (result.starts[result.valid] + 4 <= 20).all()


def test_two_group_query_does_not_duplicate_an_only_group():
    group = OnlinePrototypeGroup(0, 0, (_member(0, 0, [1, 0]),))
    snapshot = OnlinePrototypeSnapshot(1, ((group,),), frozenset(), {})
    index = TensorPrototypeIndex.from_snapshot(snapshot, device=torch.device("cpu"))
    result = index.query_two_groups(
        torch.tensor([[[[1., 0.]]]], dtype=torch.float32),
        torch.tensor([[8]]), torch.ones(1, 1, 1, dtype=torch.bool), member_topk=4,
    )
    assert result.valid.sum().item() == 1
    assert result.starts[result.valid].tolist() == [0]
