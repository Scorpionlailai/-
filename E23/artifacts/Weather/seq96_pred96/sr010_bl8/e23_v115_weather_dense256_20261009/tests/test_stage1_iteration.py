import numpy as np
import pytest
import torch

from stage1_iteration import repair_loss, selection_updates, repair_totals, summarize_draws


def test_separate_selection_preserves_two_actual_best_epochs():
    best = {'mse': float('inf'), 'mae': float('inf')}
    assert selection_updates({'global_mse': .2, 'global_mae': .1}, best) == ['mse', 'mae']
    assert selection_updates({'global_mse': .19, 'global_mae': .11}, best) == ['mse']
    assert selection_updates({'global_mse': .21, 'global_mae': .09}, best) == ['mae']
    assert best == {'mse': .19, 'mae': .09}
    assert selection_updates({'global_mse': .19, 'global_mae': .09}, best) == []
    with pytest.raises(ValueError):
        selection_updates({'global_mse': float('nan'), 'global_mae': .08}, best)
    assert best == {'mse': .19, 'mae': .09}


def test_global_scale_hidden_and_validity_and_tail_counts():
    target = torch.zeros(1, 96, 2)
    repaired = target.clone()
    repaired[0, 0, 0] = 2
    repaired[0, 95, 1] = -3
    repaired[0, 94, 1] = 900  # invalid coordinate must not be scored
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[0, 0, 0] = observed[0, 95, 1] = observed[0, 94, 1] = False
    valid = torch.ones_like(observed)
    valid[0, 94, 1] = False
    std = torch.tensor([[2., .5]])
    totals = repair_totals(repaired, target, observed, valid, std)
    assert totals == {'repair96_squared': 18.25, 'repair96_absolute': 5.5,
                      'hidden96_coordinates': 2, 'repair32_squared': 2.25,
                      'repair32_absolute': 1.5, 'hidden32_coordinates': 1}
    loss = repair_loss(repaired, target, observed, valid, std,
                       {'full_mse': 0., 'full_mae': 1., 'tail_mse': 0., 'tail_mae': .25})
    assert loss.item() == 3.125


def test_equal_mask_mean_std_are_separate_from_coordinate_weighted_mean():
    rows = [dict(repair96_mse=1., repair96_mae=1., repair32_mse=2., repair32_mae=2.,
                 hidden96_coordinates=1, hidden32_coordinates=1),
            dict(repair96_mse=3., repair96_mae=3., repair32_mse=4., repair32_mae=4.,
                 hidden96_coordinates=3, hidden32_coordinates=3)]
    result = summarize_draws(rows)
    assert result['repair96_mse']['mean'] == 2.
    assert result['repair32_mse']['mean'] == 3.
    assert result['repair96_mae']['mask_std_ddof1'] == pytest.approx(np.sqrt(2))
    assert result['coordinate_weighted']['repair96_mse'] == 2.5
    assert result['coordinate_weighted']['repair32_mae'] == 3.5
