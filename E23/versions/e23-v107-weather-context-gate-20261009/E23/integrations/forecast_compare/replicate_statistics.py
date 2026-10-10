"""Separate seed and mask uncertainty; preserve the paired crossed design."""
import numpy as np

TRAINING_SEEDS = (102, 202, 302, 402, 502)


def summarize_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or min(values.shape) < 2:
        raise ValueError('a finite seed-by-mask matrix with at least two of each is required')
    return dict(mean=float(values.mean()),
                training_seed_std_ddof1=float(values.mean(axis=1).std(ddof=1)),
                test_mask_std_ddof1=float(values.mean(axis=0).std(ddof=1)),
                training_runs=values.shape[0], test_masks=values.shape[1])


def crossed_paired_ci(ours, baseline, *, draws=10000, seed=20261004):
    ours, baseline = np.asarray(ours, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if ours.shape != baseline.shape or ours.ndim != 2 or min(ours.shape) < 2:
        raise ValueError('paired seed-by-mask matrices are required')
    delta = ours-baseline
    if not np.isfinite(delta).all() or draws < 100:
        raise ValueError('nonfinite errors or too few bootstrap draws')
    rng = np.random.default_rng(seed)
    row = rng.integers(0, len(delta), size=(draws, len(delta)))
    col = rng.integers(0, delta.shape[1], size=(draws, delta.shape[1]))
    means = delta[row[:, :, None], col[:, None, :]].mean((1, 2))
    lower, upper = np.quantile(means, [.025, .975])
    return dict(mean_difference_ours_minus_T1=float(delta.mean()),
                crossed_bootstrap_ci95=[float(lower), float(upper)], bootstrap_draws=draws,
                bootstrap_seed=seed, seed_pair_win_count=int((delta.mean(axis=1) < 0).sum()),
                mask_pair_win_count=int((delta.mean(axis=0) < 0).sum()),
                uncertainty_scope='training RNG and missing positions conditional on this fixed ETTh1 split')
