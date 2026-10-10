from __future__ import annotations


def test_online_memory_public_protocol_is_importable() -> None:
    from malr.prototypes import OnlinePrototypeMemory, OnlinePrototypeSnapshot

    memory = OnlinePrototypeMemory(
        patch_len=16,
        patch_stride=8,
        train_end=128,
        variable_count=1,
    )

    assert isinstance(memory.snapshot(), OnlinePrototypeSnapshot)
