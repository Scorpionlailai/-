# Post-Recovery Global Refinement Design

## Objective

Improve the ETTh1 missing-data model without changing PatchTST itself. The new
front-end must remove the contribution of an unreliable masked-patch embedding
when a patch is missing, then reconcile all recovered patch latents before the
unchanged PatchTST encoder. Success is measured against the `missing` baseline
under the same fixed masks: both MSE and MAE must decrease.

## Current limitation

The existing residual fusion computes

`z_residual = z_base + (1 - rho) * delta_z`.

For a fully missing patch (`rho = 0`), the masked-input projection `z_base`
remains in the result. Recovery is also performed independently per target
after relation aggregation; there is no recovery-specific pass that checks the
joint consistency of all variable-patch nodes before PatchTST.

## Approved architecture

### Replacement fusion

The relation front-end still produces temporal, variable, and historical
messages using the existing hard top-k rules. A replacement MLP consumes those
three messages, the node prior, and `z_base`, and emits an absolute candidate
latent `z_hat`.

The mixed latent is

`z_mix = rho * z_base + (1 - rho) * z_hat`.

Consequently, observed patches exactly preserve `z_base`, partially observed
patches interpolate between evidence and recovery, and fully missing patches
do not retain the defective masked-input projection.

### Optional global refiner

An optional single global attention block receives every variable-patch node in
one `[batch, variables * patches, d_model]` sequence. Variable, position, and
state priors are included in the attention input. The block contains pre-norm
multi-head self-attention, a feed-forward network, and a zero-initialized output
projection.

All nodes participate as context. Only damaged nodes are changed:

`z_final = z_mix + (1 - rho) * delta_global`.

The zero output initialization makes the refiner an exact identity at startup.
Observed patches remain fixed anchors even though they participate as keys and
values.

## Configuration and experiment isolation

- Preserve `additive` and `residual_mlp` behavior and checkpoint keys.
- Add `replacement_mlp` as a third fusion mode.
- Add opt-in refiner settings with backward-compatible disabled defaults.
- Provide two ETTh1 `start_rate=0.10`, no-recovery-loss configs:
  - replacement fusion only;
  - replacement fusion plus global refiner.
- Do not change masks, PatchTST hyperparameters, the forecasting loss, dataset
  split, seed, prototype rules, or relation top-k values.

## Validation requirements

- Exact replacement interpolation at `rho` values 1, 0.5, and 0.
- Fully missing output must be independent of `z_base` when the candidate is
  held fixed.
- Refiner must be identity at initialization.
- Observed nodes must remain bitwise unchanged after refinement.
- Refiner output projection must receive gradient at the first optimization
  step; inner layers may begin receiving gradient after the output gate opens.
- Legacy additive and residual tests must remain green.
- Model construction and a synthetic training step must work with the refiner.

## Experimental decision rule

Tune on ETTh1 validation only. Compare the existing `missing`, current
`residual_mlp`, replacement-only, and replacement-plus-refiner variants under
the same fixed mask hashes. Do not inspect the test set until the architecture
choice is locked. A candidate advances only if validation MSE and validation
MAE both beat `missing`.
