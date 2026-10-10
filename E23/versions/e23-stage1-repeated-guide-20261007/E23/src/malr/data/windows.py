from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .splits import DatasetProtocol, SplitName


@dataclass(frozen=True)
class WindowRef:
    input_start: int
    input_end: int
    target_start: int
    target_end: int
    split: Literal["train", "validation", "test"]

    def __post_init__(self) -> None:
        if not (
            0 <= self.input_start < self.input_end <= self.target_start < self.target_end
        ):
            raise ValueError("window intervals must be ordered and non-overlapping")


def build_window_index(
    protocol: DatasetProtocol,
    split: SplitName,
    seq_len: int,
    pred_len: int,
) -> list[WindowRef]:
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")
    bounds = protocol.bounds_for(split)
    first_target = bounds.start + seq_len if split == "train" else bounds.start
    last_target = bounds.end - pred_len
    if first_target > last_target:
        return []

    refs = [
        WindowRef(
            input_start=target_start - seq_len,
            input_end=target_start,
            target_start=target_start,
            target_end=target_start + pred_len,
            split=split,
        )
        for target_start in range(first_target, last_target + 1)
    ]
    if any(ref.input_start < 0 for ref in refs):
        raise ValueError("split does not contain enough prior history")
    if any(
        ref.target_start < bounds.start or ref.target_end > bounds.end
        for ref in refs
    ):
        raise AssertionError("window target crossed its split boundary")
    return refs

