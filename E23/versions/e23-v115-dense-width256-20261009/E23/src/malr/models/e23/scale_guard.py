"""Limit extreme normalized repair tails before a frozen forecast interface."""
import math

import torch


def compress_hidden_normalized_tails(sequence, observed, validity, *, threshold=16.0):
    """Keep the central range exact and compress only missing-value tails.

    The tail is sign(z) * (threshold + asinh(abs(z) - threshold)). It is
    continuous with a continuous first derivative at the threshold and keeps a
    nonzero gradient in the tail. The operation belongs to the repair output;
    it changes neither visible coordinates nor predictor statistics/config.
    """
    if sequence.shape != observed.shape or sequence.shape != validity.shape:
        raise ValueError('repair guard masks must match the sequence')
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('repair guard threshold must be finite and positive')
    hidden = validity.to(torch.bool) & ~observed.to(torch.bool)
    safe = torch.where(hidden, sequence, torch.zeros_like(sequence))
    magnitude = safe.abs()
    tail = safe.sign()*(threshold+torch.asinh((magnitude-threshold).clamp_min(0)))
    compressed = torch.where(magnitude > threshold, tail, safe)
    return torch.where(hidden, compressed, sequence)
