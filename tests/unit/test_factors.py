import numpy as np

from malr.prototypes.factors import compute_context_factors


def test_factors_use_context_not_center():
    segment = np.arange(64, dtype=np.float64)
    valid = np.ones(64, dtype=bool)
    original = compute_context_factors(
        segment, valid, center_slice=slice(24, 40), train_std=2.0
    )

    changed = segment.copy()
    changed[24:40] = 1e9
    after = compute_context_factors(
        changed, valid, center_slice=slice(24, 40), train_std=2.0
    )

    np.testing.assert_allclose(original.values, after.values)
    np.testing.assert_array_equal(original.available, after.available)


def test_unavailable_dimensions_are_masked():
    valid = np.zeros(64, dtype=bool)
    valid[:8] = True
    factors = compute_context_factors(
        np.arange(64, dtype=np.float64),
        valid,
        center_slice=slice(24, 40),
        train_std=1.0,
    )

    assert factors.available.shape == (7,)
    assert not factors.available[3:].any()
    assert np.isfinite(factors.values).all()


def test_boundary_gap_is_scaled_by_train_standard_deviation():
    segment = np.zeros(64, dtype=np.float64)
    segment[:24] = 2.0
    segment[40:] = 8.0
    valid = np.ones(64, dtype=bool)

    factors = compute_context_factors(
        segment, valid, center_slice=slice(24, 40), train_std=3.0
    )

    assert factors.available[-1]
    assert factors.values[-1] == 2.0
