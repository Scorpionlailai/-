import torch

from malr.data.normalization import (
    TrainVariableStats,
    denormalize_forecast,
    fit_train_variable_stats,
    masked_window_normalize,
    standardize_values,
)
from malr.data.patching import (
    PATCH_MISSING,
    PATCH_OBSERVED,
    PATCH_PARTIAL,
    patchify_with_state,
)
from malr.data.windows import WindowRef
from malr.models.stage1 import ModelVariant
from malr.training.data import iter_prepared_batches


def test_masked_values_cannot_change_student_statistics() -> None:
    clean = torch.tensor([[[1.0], [2.0], [100.0], [200.0]]])
    mask = torch.tensor([[[True], [True], [False], [False]]])
    fallback = TrainVariableStats(
        mean=torch.tensor([5.0]),
        std=torch.tensor([2.0]),
    )

    first = masked_window_normalize(clean, mask, fallback, eps=1e-5)
    changed = clean.clone()
    changed[:, 2:] = -9999
    second = masked_window_normalize(changed, mask, fallback, eps=1e-5)

    torch.testing.assert_close(first.mean, second.mean)
    torch.testing.assert_close(first.std, second.std)
    torch.testing.assert_close(first.student, second.student)
    torch.testing.assert_close(first.student[:, 2:], torch.zeros(1, 2, 1))


def test_complete_target_uses_same_observed_window_coordinate() -> None:
    clean = torch.tensor([[[1.0], [3.0], [100.0], [200.0]]])
    mask = torch.tensor([[[True], [True], [False], [False]]])
    fallback = TrainVariableStats(
        mean=torch.tensor([5.0]),
        std=torch.tensor([2.0]),
    )

    result = masked_window_normalize(clean, mask, fallback, eps=1e-5)

    expected_target = (clean - result.mean.unsqueeze(1)) / result.std.unsqueeze(1)
    torch.testing.assert_close(result.target, expected_target)
    torch.testing.assert_close(
        result.student[:, :2], result.target[:, :2]
    )
    torch.testing.assert_close(
        result.student[:, 2:], torch.zeros_like(result.student[:, 2:])
    )


def test_hidden_values_change_target_but_not_missing_input_or_statistics() -> None:
    clean = torch.tensor([[[1.0], [3.0], [100.0], [200.0]]])
    mask = torch.tensor([[[True], [True], [False], [False]]])
    fallback = TrainVariableStats(
        mean=torch.tensor([5.0]),
        std=torch.tensor([2.0]),
    )

    first = masked_window_normalize(clean, mask, fallback, eps=1e-5)
    changed = clean.clone()
    changed[:, 2:] = torch.tensor([[-7.0], [11.0]])
    second = masked_window_normalize(changed, mask, fallback, eps=1e-5)

    torch.testing.assert_close(first.mean, second.mean)
    torch.testing.assert_close(first.std, second.std)
    torch.testing.assert_close(first.student, second.student)
    assert not torch.allclose(first.target[:, 2:], second.target[:, 2:])


def test_fully_missing_variable_uses_train_fallback() -> None:
    clean = torch.tensor([[[1.0, 10.0], [3.0, 20.0]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    fallback = TrainVariableStats(
        mean=torch.tensor([7.0, 15.0]),
        std=torch.tensor([4.0, 5.0]),
    )

    result = masked_window_normalize(clean, mask, fallback, eps=1e-5)

    assert result.used_fallback.tolist() == [[False, True]]
    torch.testing.assert_close(result.mean[0, 1], torch.tensor(15.0))
    torch.testing.assert_close(result.std[0, 1], torch.tensor(5.0))
    torch.testing.assert_close(result.student[0, :, 1], torch.zeros(2))
    torch.testing.assert_close(
        result.target[0, :, 1], (clean[0, :, 1] - 15.0) / 5.0
    )


def test_train_variable_stats_and_denormalization() -> None:
    train = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    stats = fit_train_variable_stats(train, eps=1e-5)
    normalized_forecast = torch.zeros(1, 4, 2)

    restored = denormalize_forecast(
        normalized_forecast,
        stats.mean.unsqueeze(0),
        stats.std.unsqueeze(0),
    )

    torch.testing.assert_close(stats.mean, torch.tensor([3.0, 14.0]))
    torch.testing.assert_close(restored[0, 0], stats.mean)


def test_official_scaling_is_fit_on_training_values_only() -> None:
    values = torch.tensor([[1.0], [3.0], [101.0], [-9999.0]])
    train_stats = fit_train_variable_stats(values[:2], eps=0.0)

    standardized = standardize_values(values, train_stats)

    torch.testing.assert_close(standardized[:2].mean(0), torch.zeros(1))
    torch.testing.assert_close(
        standardized[:2].square().mean(0).sqrt(), torch.ones(1)
    )
    assert standardized[2, 0].item() == 99.0
    assert torch.isfinite(standardized[3, 0])


def test_patchtst_geometry_and_states_exclude_padding() -> None:
    values = torch.ones(1, 96, 1)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 16:32] = False

    patches = patchify_with_state(values, observed, length=16, stride=8)

    assert patches.values.shape == (1, 1, 12, 16)
    assert patches.validity[:, :, -1].sum().item() == 8
    assert set(patches.state.flatten().tolist()) == {
        PATCH_OBSERVED,
        PATCH_PARTIAL,
        PATCH_MISSING,
    }
    assert patches.rho[0, 0, 2].item() == 0.0
    assert patches.state[0, 0, 2].item() == PATCH_MISSING
    assert patches.rho[0, 0, -1].item() == 1.0


def test_padding_replicates_last_value_but_is_excluded_from_observation_counts() -> None:
    values = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    observed = torch.ones_like(values, dtype=torch.bool)

    patches = patchify_with_state(values, observed, length=16, stride=8)

    torch.testing.assert_close(
        patches.values[0, 0, -1, 8:], torch.full((8,), 95.0)
    )
    assert not patches.effective_mask[0, 0, -1, 8:].any()
    assert not patches.validity[0, 0, -1, 8:].any()


def test_training_batch_optionally_carries_shared_coordinate_target() -> None:
    series = torch.arange(36, dtype=torch.float32).view(36, 1).numpy()
    timeline_mask = torch.ones((32, 1), dtype=bool)
    timeline_mask[8:16] = False
    refs = [WindowRef(0, 32, 32, 36, "train")]
    stats = fit_train_variable_stats(torch.from_numpy(series), eps=0.0)

    with_target = next(
        iter_prepared_batches(
            series,
            refs,
            timeline_mask,
            mask_support_start=0,
            train_stats=stats,
            variant=ModelVariant.OURS,
            batch_size=1,
            patch_len=16,
            patch_stride=8,
            include_imputation_target=True,
        )
    )
    without_target = next(
        iter_prepared_batches(
            series,
            refs,
            timeline_mask,
            mask_support_start=0,
            train_stats=stats,
            variant=ModelVariant.OURS,
            batch_size=1,
            patch_len=16,
            patch_stride=8,
            include_imputation_target=False,
        )
    )

    assert with_target.imputation_target is not None
    assert with_target.imputation_target.shape == (1, 32, 1)
    assert without_target.imputation_target is None
