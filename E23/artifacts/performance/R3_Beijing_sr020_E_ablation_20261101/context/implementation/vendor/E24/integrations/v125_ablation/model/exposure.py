"""Deterministic specialist exposure shared by parallel and iterative repair."""

import torch
from torch import Tensor


def sample_specialist_exposure(
    damaged: Tensor, availability: Tensor, *, fraction: float, seed: int,
) -> tuple[Tensor, Tensor]:
    """Expose exactly one usable expert for a fixed-seed subset of damaged targets."""
    if availability.shape != (*damaged.shape, 3):
        raise ValueError("specialist availability must have shape [B,C,N,3]")
    if not 0 <= fraction <= 1 or seed < 0:
        raise ValueError("specialist fraction/seed is invalid")
    generator = torch.Generator(device=damaged.device).manual_seed(seed)
    eligible = damaged.to(torch.bool).flatten().nonzero().flatten()
    selected = torch.zeros(damaged.numel(), dtype=torch.bool, device=damaged.device)
    count = round(fraction * len(eligible))
    if count:
        choice = torch.randperm(len(eligible), generator=generator, device=damaged.device)[:count]
        selected[eligible[choice]] = True
    selected = selected.reshape_as(damaged)
    random = torch.rand(availability.shape, generator=generator, device=damaged.device)
    winner = random.masked_fill(~availability.to(torch.bool), -1).argmax(dim=-1)
    singleton = torch.nn.functional.one_hot(winner, num_classes=3).to(torch.bool)
    allowed = torch.where(selected[..., None], singleton & availability, availability)
    return allowed, selected
