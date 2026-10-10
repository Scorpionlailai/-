# E23 v25: full-width heads with learned selection

## Intent

Test whether the weak v24 8×8 result was caused by narrow heads rather than the multihead idea. Keep the final repair latent at width 64, but give each of eight H/V/T candidate-reading heads its own 64-dimensional computation. A learned gate decides each head's contribution for each damaged patch. Keep the original explicit repair decoder and PatchTST prediction head.

## Architecture

The H, V, and T candidate pools and one-time candidate selection remain as in v12/v23/v24. Each expert projects the shared candidates into eight separate 64-dimensional query/key/value spaces. Each head independently attends to candidates and routes none/H/V/T. Its weighted H/V/T residual is a 64-dimensional proposal. A second router scores the eight proposals, their candidate quality, the base patch, and local observation geometry. Softmax over eight proposals plus a no-change option gives nonnegative weights summing to one. The final latent is the original 64-dimensional base plus the weighted sum of proposals. Undamaged patches and patches with no valid expert evidence keep the base exactly.

The head gate is learned jointly in the original two-stage schedule; no hand-pruned head, diversity penalty, extra candidate, or prediction-head change is included. Public router weights summarize the effective none/H/V/T contribution after the head gate, and diagnostics expose per-head routes, gate weights, and contributions.

## Evaluation

Train fresh Stage 1 and Stage 2, 100 epochs each, on locked ETTh1 P4 with seed 2021, same v12 candidate policies, MSE + 0.2 MAE Stage 2 objective, validation MAE selection, and original PatchTST. Compare validation against selected v12 MAE 0.5882120694703256 and MSE 0.7454477286392869. Promote only if MAE is strictly lower and MSE no higher. Replay the selected checkpoint and verify H state; test only after the gate passes. Include v24 and v23 validation results as context, not as fresh controls.

## Compatibility and checks

Implement as an opt-in router mode restricted to width 64 and v12 candidate policies. Keep v12/v23/v24 checkpoint shapes and behavior unchanged. Test gate normalization, candidate-level gradient flow, exact no-evidence fallback, output width, config identity, observed-value preservation, and strict v12 loading.
