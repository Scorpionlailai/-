import torch
import pytest
from torch import nn

from malr.prototypes.retrieval import (
    MaskAwareRetrievalEncoder,
    RetrievalEncoderEMA,
    masked_pair_loss,
    normalized_clean_keys,
    retrieval_state_tuple,
)


def test_mask_aware_encoder_has_fixed_normalized_output() -> None:
    encoder = MaskAwareRetrievalEncoder(patch_len=4, retrieval_dim=3, hidden_dim=8)
    values = torch.randn(5, 4)
    mask = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1]],
        dtype=torch.bool,
    )
    keys = encoder(values, mask)

    assert keys.shape == (5, 3)
    assert torch.allclose(keys.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_clean_keys_stop_gradient_and_pair_loss_is_differentiable() -> None:
    projection = nn.Linear(4, 3, bias=False)
    clean = torch.randn(5, 4)
    keys = normalized_clean_keys(projection, clean)
    assert not keys.requires_grad

    queries = keys.detach().clone().requires_grad_(True)
    loss = masked_pair_loss(queries, keys)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    loss.backward()
    assert queries.grad is not None


def _retrieval_states_equal(
    first: tuple[tuple[str, object], ...],
    second: tuple[tuple[str, object], ...],
) -> bool:
    if [name for name, _ in first] != [name for name, _ in second]:
        return False
    return all(
        torch.equal(torch.as_tensor(first_value), torch.as_tensor(second_value))
        for (_, first_value), (_, second_value) in zip(first, second)
    )


def test_ema_updates_after_step_but_frozen_view_waits_for_next_epoch() -> None:
    encoder = MaskAwareRetrievalEncoder(patch_len=4, retrieval_dim=2, hidden_dim=3)
    manager = RetrievalEncoderEMA(encoder, decay=0.9)
    initial_frozen = retrieval_state_tuple(manager.frozen_encoder)

    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.fill_(1.0)
    manager.update_ema()

    assert not _retrieval_states_equal(
        retrieval_state_tuple(manager.ema_encoder), initial_frozen
    )
    assert _retrieval_states_equal(
        retrieval_state_tuple(manager.frozen_encoder), initial_frozen
    )

    manager.begin_epoch(1)

    assert _retrieval_states_equal(
        retrieval_state_tuple(manager.frozen_encoder),
        retrieval_state_tuple(manager.ema_encoder),
    )


def test_ema_state_round_trip_preserves_frozen_epoch() -> None:
    encoder = MaskAwareRetrievalEncoder(patch_len=4, retrieval_dim=2, hidden_dim=3)
    manager = RetrievalEncoderEMA(encoder)
    manager.begin_epoch(1)
    state = manager.state_dict()

    restored = RetrievalEncoderEMA(MaskAwareRetrievalEncoder(4, 2, 3))
    restored.load_state_dict(state)

    assert restored.frozen_epoch == 1
    assert _retrieval_states_equal(
        retrieval_state_tuple(restored.frozen_encoder),
        retrieval_state_tuple(manager.frozen_encoder),
    )
