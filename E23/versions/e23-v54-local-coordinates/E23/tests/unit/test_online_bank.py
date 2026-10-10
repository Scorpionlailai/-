from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

from malr.prototypes.online_bank import OnlinePrototypeMemory, OnlinePrototypeMember


def make_memory(**kwargs: object) -> OnlinePrototypeMemory:
    return OnlinePrototypeMemory(
        patch_len=16,
        patch_stride=8,
        train_end=128,
        variable_count=1,
        **kwargs,
    )


def make_member(
    variable_id: int,
    start: int,
    *,
    clean_key: list[float] | None = None,
) -> OnlinePrototypeMember:
    key = np.asarray(clean_key or [1.0, 0.0], dtype=np.float64)
    raw_patch = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    return OnlinePrototypeMember(variable_id, start, raw_patch, key)


def test_epoch_one_starts_empty_and_pending_patch_is_not_readable() -> None:
    memory = make_memory()
    member = make_member(variable_id=0, start=0)

    assert memory.snapshot().epoch == 0
    assert memory.query(0, 32, [1.0, 0.0]) == ()
    assert memory.stage_patch(0, 0, member.raw_patch, member.clean_key, observed=True)
    assert memory.query(0, 32, [1.0, 0.0]) == ()

    published = memory.publish_epoch(1)

    assert published.epoch == 1
    assert [item.start for item in memory.query(0, 32, [1.0, 0.0])] == [0]


def test_masked_patch_is_not_staged_and_duplicate_identity_is_permanent() -> None:
    memory = make_memory()
    member = make_member(variable_id=0, start=8)

    assert not memory.stage_patch(0, 8, member.raw_patch, member.clean_key, observed=False)
    assert memory.publish_epoch(1).groups == ((),)
    assert memory.stage_patch(0, 8, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(2)

    assert not memory.stage_patch(0, 8, member.raw_patch, member.clean_key, observed=True)


def test_query_enforces_strict_nonoverlap_causality() -> None:
    memory = make_memory()
    for start in (0, 16, 32):
        member = make_member(variable_id=0, start=start)
        memory.stage_patch(0, start, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(1)

    assert [item.start for item in memory.query(0, 32, [1.0, 0.0])] == [0, 16]


def test_group_center_is_recomputed_from_retained_members_only() -> None:
    memory = make_memory(group_capacity=2, group_similarity_threshold=0.8)
    for start, key in (
        (0, [1.0, 0.0]),
        (8, [1.0, 0.0]),
        (16, [0.8660254, 0.5]),
    ):
        member = make_member(variable_id=0, start=start, clean_key=key)
        memory.stage_patch(0, start, member.raw_patch, member.clean_key, observed=True)

    snapshot = memory.publish_epoch(1)
    retained = snapshot.groups[0][0]

    assert [member.start for member in retained.members] == [8, 16]
    assert np.allclose(retained.center, np.array([0.9659258, 0.2588190]), atol=1e-5)
    assert not np.allclose(retained.center, np.array([0.9850713, 0.1721495]), atol=1e-5)


def test_publish_is_independent_of_pending_submission_order() -> None:
    first = make_memory()
    second = make_memory()
    members = [make_member(0, start) for start in (32, 0, 16)]

    for member in members:
        first.stage_patch(0, member.start, member.raw_patch, member.clean_key, observed=True)
    for member in reversed(members):
        second.stage_patch(0, member.start, member.raw_patch, member.clean_key, observed=True)

    first.publish_epoch(1)
    second.publish_epoch(1)

    assert first.snapshot_hash() == second.snapshot_hash()


def test_evicted_identity_cannot_be_inserted_again() -> None:
    memory = make_memory(group_capacity=1, group_similarity_threshold=1.0)
    first = make_member(0, 0)
    second = make_member(0, 16, clean_key=[1.0, 0.0])

    memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)
    memory.publish_epoch(1)
    memory.stage_patch(0, 16, second.raw_patch, second.clean_key, observed=True)
    memory.publish_epoch(2)

    assert not memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)


def test_state_round_trip_preserves_snapshot_and_hash() -> None:
    memory = make_memory()
    member = make_member(0, 0)
    memory.stage_patch(0, 0, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(1)

    state = memory.state_dict()
    restored = make_memory()
    restored.load_state_dict(state)

    assert restored.snapshot_hash() == memory.snapshot_hash()


def test_query_is_read_only_and_diagnostics_track_staging_events() -> None:
    memory = make_memory(group_capacity=1, group_similarity_threshold=1.0)
    member = make_member(0, 0)
    memory.stage_patch(0, 0, member.raw_patch, member.clean_key, observed=False)
    memory.stage_patch(0, 0, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(1)

    before = memory.snapshot_hash()
    assert [item.start for item in memory.query(0, 32, [1.0, 0.0])] == [0]
    assert memory.snapshot_hash() == before
    assert memory.diagnostics()["skipped_masked"] == 1
    assert memory.diagnostics()["staged_observed"] == 1


def test_rekey_updates_published_and_pending_members_before_publish() -> None:
    memory = make_memory(group_similarity_threshold=0.8)
    first = make_member(0, 0, clean_key=[1.0, 0.0])
    second = make_member(0, 16, clean_key=[1.0, 0.0])

    memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)
    memory.publish_epoch(1)
    memory.stage_patch(0, 16, second.raw_patch, second.clean_key, observed=True)

    memory.rekey(lambda _member: np.array([0.0, 1.0], dtype=np.float64))

    published_member = memory.snapshot().groups[0][0].members[0]
    assert np.allclose(published_member.clean_key, [0.0, 1.0])
    assert [item.start for item in memory.query(0, 32, [0.0, 1.0])] == [0]

    memory.publish_epoch(2)

    assert len(memory.snapshot().groups[0]) == 1
    retained = memory.snapshot().groups[0][0].members
    assert [member.start for member in retained] == [0, 16]
    assert all(np.allclose(member.clean_key, [0.0, 1.0]) for member in retained)


def test_rekey_validates_all_members_before_mutating_memory() -> None:
    memory = make_memory()
    first = make_member(0, 0, clean_key=[1.0, 0.0])
    second = make_member(0, 16, clean_key=[1.0, 0.0])
    memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)
    memory.publish_epoch(1)
    memory.stage_patch(0, 16, second.raw_patch, second.clean_key, observed=True)
    before_hash = memory.snapshot_hash()
    before_pending = np.array(memory.state_dict()["pending"][0]["clean_key"], copy=True)

    with pytest.raises(ValueError, match="non-zero norm"):
        memory.rekey(
            lambda member: [0.0, 1.0] if member.start == 0 else [0.0, 0.0]
        )

    assert memory.snapshot_hash() == before_hash
    assert np.allclose(memory.snapshot().groups[0][0].members[0].clean_key, [1.0, 0.0])
    after_pending = memory.state_dict()["pending"][0]["clean_key"]
    assert np.array_equal(after_pending, before_pending)


def test_batched_rekey_updates_all_members_with_one_callback() -> None:
    memory = make_memory(group_similarity_threshold=0.8)
    first = make_member(0, 0, clean_key=[1.0, 0.0])
    second = make_member(0, 16, clean_key=[1.0, 0.0])
    memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)
    memory.publish_epoch(1)
    memory.stage_patch(0, 16, second.raw_patch, second.clean_key, observed=True)
    callback_calls = 0

    def new_keys(members: tuple[OnlinePrototypeMember, ...]) -> np.ndarray:
        nonlocal callback_calls
        callback_calls += 1
        return np.tile(np.array([[0.0, 1.0]]), (len(members), 1))

    memory.rekey_batch(new_keys)
    memory.publish_epoch(2)

    retained = memory.snapshot().groups[0][0].members
    assert callback_calls == 1
    assert [member.start for member in retained] == [0, 16]
    assert all(np.allclose(member.clean_key, [0.0, 1.0]) for member in retained)


def test_epoch_publish_scales_to_a_training_sized_pending_buffer() -> None:
    memory = OnlinePrototypeMemory(
        patch_len=16,
        patch_stride=8,
        train_end=1_000,
        variable_count=7,
        group_capacity=16,
        max_groups_per_variable=128,
    )
    random = np.random.default_rng(7)
    for index in range(5_000):
        memory.stage_patch(
            index % 7,
            index // 7,
            random.normal(size=16),
            random.normal(size=32),
            observed=True,
        )

    started = perf_counter()
    snapshot = memory.publish_epoch(1)
    elapsed = perf_counter() - started

    assert snapshot.epoch == 1
    assert elapsed < 2.0


def test_batched_query_matches_scalar_causality_and_ranking() -> None:
    memory = make_memory(group_similarity_threshold=0.8)
    for start, key in (
        (0, [1.0, 0.0]),
        (16, [0.9, 0.1]),
        (32, [0.0, 1.0]),
    ):
        member = make_member(0, start, clean_key=key)
        memory.stage_patch(0, start, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(1)
    starts = np.array([16, 32, 64], dtype=np.int64)
    keys = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    batched = memory.query_batch(
        np.zeros(3, dtype=np.int64), starts, keys, max_items=2
    )
    scalar = tuple(
        memory.query(0, int(start), key, max_items=2)
        for start, key in zip(starts, keys, strict=True)
    )

    assert tuple(tuple(member.identity for member in row) for row in batched) == tuple(
        tuple(member.identity for member in row) for row in scalar
    )
