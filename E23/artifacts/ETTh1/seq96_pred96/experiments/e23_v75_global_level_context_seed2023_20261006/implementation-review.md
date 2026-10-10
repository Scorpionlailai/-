# E23-v75 implementation review

Scope: bounded implementation and pre-run evidence review against the v75 design and implementation plan. Read-only source inspection; no GPU, optimizer, formal training, or test-bank scoring. Only this requested review artifact was written.

## Findings

Critical: none found.

Important: none found.

Minor:

1. `execute.py:131` writes `only_change="Full96 long-gap attention repair readout; original v72 losses, experts and Stage2"` into the new primary cell plan; the secondary inherits it. This describes the preceding attention experiment, whereas the actual frozen configuration disables attention and adds the 128-parameter global-level branch. Effect: future cell metadata will misdescribe the intervention. Correct the wording through an explicitly recorded controller revision and renewed freeze before producing immutable cell outputs; do not silently alter the currently frozen controller.
2. `publish.py:151` renders successful validation promotion as `保留v73候选`. Its numeric gate and machine-readable `retained_candidate` use the current candidate correctly, but a successful v75 run would publish the wrong human-readable version. Correct the publisher wording to v75/current name before publication and record its reviewed revision. This does not require changing model source or old trial artifacts.

## Implementation assessment

- Exactly four copied source files changed: `integrations/e23/config.py`, `integrations/e23/train.py`, `src/malr/models/e23/model.py`, and `src/malr/models/e23/window_context.py`. New source-tree files are the declared configuration and seven-test module.
- `window_context.py:62-67` constructs the two-input, 64-output, bias-free projection after all original modules, inside a CPU RNG fork, and zeros its 128 weights. The original five-channel convolution/GELU operators and existing state keys remain intact; `none` retains the original execution path.
- `window_context.py:118-127` derives global visible values from masked original normalized values multiplied by query standard deviation plus query mean. It intersects observed and valid masks before arithmetic, derives the mean only from those values, and clears invalid positions. An entirely unobserved variable supplies zero in both new channels. No target tensor or future statistic is introduced.
- The selected runtime uses `repair_input_scaling=legacy_v1`, so the supplied normalized values and statistics reconstruct global train-standardized values once. The existing robust local five-feature branch remains separate; it is not rescaled twice.
- The downstream returned evidence remains five channels. The parallel input projection is mathematically a seven-channel pre-GELU input while preserving the original five-channel operator. Existing signature source reliability masking is unchanged, and the activated-branch test demonstrates cross-variable level sensitivity capacity.
- `_trial_identity` binds mode, seven CNN/five evidence channels, two level features, 128 parameters, source, units, projection, and no-observation behavior. Existing manifest and checkpoint verification loops enforce these entries. Default `none` omits the added identity and retains backward compatibility.
- The formal configuration contains no attention readout and retains the original predictor and v72 losses/full Stage2 freeze. The existing preflight records repair 421101/predictor 35427, finite losses, positive second-step level gradient, and no test scoring. This review did not rerun its optimizer steps.

## Independent verification

- Five new non-training tests passed on CPU in 1.39 seconds; the optimizer/roundtrip-training tests were deliberately deselected to honor review scope. These five cover configuration, exact zero-start parameters/RNG/head/forecast, exact channels and five-feature interface, cross-variable sensitivity, and hidden/invalid payload isolation.
- An additional small CPU forward/backward probe used an activated projection and payload variants zero, NaN, positive Inf, and negative Inf at hidden/invalid locations. Outputs and every populated parameter gradient matched exactly; excluded input gradients were zero and all gradients finite. No optimizer was constructed or stepped.
- All 528 frozen source hashes, all three controller hashes, and the plan hash matched. A 26-file sample of the protected manifest matched. The copied-parent hash comparison identified exactly the four declared changed source files.
- Inspected full-suite log reports 1173 passed, nine failed, one skipped; its nine failure names match the disclosed inherited list. No full-suite rerun was performed.

## Limits and readiness

No evidence yet establishes forecasting benefit, T1 superiority, optimization stability over 100 epochs, or independent-target/multiple-seed generalization. The cross-variable test establishes representational capacity after manual activation, not learned effectiveness. Checkpoint roundtrip training, full freeze, and CUDA evidence were inspected from existing tests/preflight rather than re-executed.

The model implementation is ready for the bounded formal experiment under the locked validation policy. Resolve or explicitly annotate the two metadata defects before corresponding immutable cell/report outputs are produced. No model-source change is required by this review; it is not a merge recommendation or a claim of experimental success.
