# E23: eight candidate-reading H/V/T heads

## Goal and decision

Improve ETTh1 P4 forecast accuracy beyond selected E23 v12 while keeping its original PatchTST forecast head and fixed data/masks. The approved design uses eight parallel heads of width 16, concatenated to a 128-dimensional repair latent. Each head reads the selected H/V/T candidates before their current within-expert weighted average. Candidate retrieval and source selection happen once per forward pass.

## Architecture

Each expert publishes its selected candidate messages, raw candidate quality, validity, and original candidate weights as differentiable evidence. H has four candidates; V has four source variables and three offsets (flattened to 12); T has ten directional bands. No candidates outside these pools are added in the first trial.

For each expert, eight learned query/key/value projections produce independent masked attention over its candidates. The original within-expert weights act as a finite log prior. Each head receives one H, V, and T summary, then independently chooses none/H/V/T using geometry and candidate quality. A head with no valid candidates gives that expert zero probability; an undamaged patch takes the none path and retains its base latent. The eight residual head outputs concatenate to 128 dimensions, then the existing explicit repair decoder and unchanged PatchTST predictor run.

The public router weights remain the mean of eight head route weights, preserving existing logging. Separate head weights and candidate attention remain available for diagnosis. Gradients must reach candidate messages and their projectors; diagnostic provenance stays detached.

## Controlled evaluation

Train a fresh single-head 128-dimensional control and the eight-head 128-dimensional trial with otherwise identical v12 candidate policies, ETTh1 P4 masks, seed 2021, 100 Stage 1 epochs, 100 Stage 2 epochs, and Stage 2 MSE + 0.2 MAE with validation-MAE checkpoint selection. No new prediction head, auxiliary diversity loss, or candidate cap change. Compare selected validation MAE and MSE to v12 (0.5882120694703256 and 0.7454477286392869). Test only a trial that improves MAE strictly without increasing MSE; otherwise v12 stays selected. Report per-head routing/attention and ablations as diagnostic evidence, not as a promotion metric.

## Safety and compatibility

The new router is opt-in through a config key allowed only for width 128 with v12 H/V/T policies. Existing configs and checkpoints retain the original router and strict loading. Candidate tensors retain gradients only during the forward graph and are not serialized in checkpoints. Tests cover shape, masking, candidate dependence, gradient flow, config identity, and legacy behavior.
