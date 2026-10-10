from pathlib import Path

import pytest
import torch

from integrations.forecast_compare.author_profiles import (
    load_author_profiles, epoch_end_learning_rate, build_author_model, author_coifnet_loss, _OriginalBRITS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_ctf_and_t1_scheduler_semantics_differ():
    profiles = load_author_profiles()
    ctf = profiles['models']['ChannelTokenFormer']['training']
    t1 = profiles['models']['T1']['training']
    assert [epoch_end_learning_rate(ctf, e) for e in (1, 2, 3)] == [1e-4, 5e-5, 2.5e-5]
    assert [epoch_end_learning_rate(t1, e) for e in (1, 2, 3)] == [1e-3] * 3


def test_paper_coifnet_weight_changes_loss():
    prediction = torch.cat((torch.ones(2, 96, 7) * 2, torch.ones(2, 96, 7)), dim=1)
    observed = torch.zeros(2, 96, 7)
    loss = author_coifnet_loss(prediction, observed, torch.ones_like(observed), observed, weight=0.2)
    assert loss.item() == pytest.approx(1.8)


def test_original_brits_runs_on_cpu_and_hidden_targets_are_not_inputs():
    cfg = load_author_profiles()['models']['BRITS']
    model = build_author_model('BRITS', cfg, ROOT / cfg['root'], torch.device('cpu')).eval()
    values, mask = torch.randn(2, 96, 7), torch.ones(2, 96, 7)
    mask[:, 20:28, 1] = 0
    values = values * mask
    result = model(values, mask=mask)
    assert result.shape == values.shape
    assert torch.equal(result[mask.bool()], values[mask.bool()])
    loss = model.original_loss(values, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_original_brits_delta_uses_current_mask_at_block_boundaries():
    # Original input_process.py parse_delta: 1+(1-masks[h])*deltas[h-1].
    mask = torch.tensor([[[1.], [0.], [0.], [1.]]])
    result = _OriginalBRITS._direction(torch.zeros_like(mask), mask)
    assert result['deltas'].flatten().tolist() == [1., 2., 3., 1.]
