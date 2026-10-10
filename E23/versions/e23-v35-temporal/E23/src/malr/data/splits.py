from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, order=True)
class SplitBounds:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("split start must be non-negative")
        if self.end <= self.start:
            raise ValueError("split end must be greater than split start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    source_length: int
    train: SplitBounds
    validation: SplitBounds
    test: SplitBounds

    def __post_init__(self) -> None:
        if self.source_length < self.test.end:
            raise ValueError("source length is shorter than the test boundary")
        if self.train.end != self.validation.start:
            raise ValueError("train and validation splits must be contiguous")
        if self.validation.end != self.test.start:
            raise ValueError("validation and test splits must be contiguous")

    def bounds_for(self, split: SplitName) -> SplitBounds:
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        if split == "test":
            return self.test
        raise ValueError(f"unknown split: {split}")


def protocol_from_bounds(
    name: str,
    length: int,
    *,
    train_start: int,
    train_end: int,
    validation_end: int,
    test_end: int,
) -> DatasetProtocol:
    """Build a protocol from the boundaries recorded in an experiment config."""

    if length < test_end:
        raise ValueError(
            f"{name} requires at least {test_end} rows for the configured protocol, "
            f"found {length}"
        )
    return DatasetProtocol(
        name=name,
        source_length=length,
        train=SplitBounds(train_start, train_end),
        validation=SplitBounds(train_end, validation_end),
        test=SplitBounds(validation_end, test_end),
    )


def protocol_for(name: str, length: int) -> DatasetProtocol:
    """Return the two legacy protocols kept for backwards-compatible callers."""

    if name == "ETTh1":
        if length < 14400:
            raise ValueError("ETTh1 requires at least 14400 rows")
        return DatasetProtocol(
            name=name,
            source_length=length,
            train=SplitBounds(0, 8640),
            validation=SplitBounds(8640, 11520),
            test=SplitBounds(11520, 14400),
        )
    if name == "Weather":
        if length != 52696:
            raise ValueError("Weather requires exactly 52696 rows")
        return DatasetProtocol(
            name=name,
            source_length=length,
            train=SplitBounds(0, 36887),
            validation=SplitBounds(36887, 42157),
            test=SplitBounds(42157, 52696),
        )
    raise ValueError(f"unsupported dataset: {name}")

