# E23-v76 independent implementation review

## Findings and readiness

- **Critical: none.**
- **Important: none.**
- **Minor, corrected: parameter-count label.** The original frozen `plan.json` and `preflight.py/json` call 456400 Stage1 parameters “trainable”. Independent construction finds 420973 trainable repair parameters plus 35424 trainable auxiliary parameters, totaling **456397 trainable / 456400 total**. The auxiliary head's three scalar `core.backbone.encoder.layers.{0,1,2}.self_attn.sdp_attn.scale` parameters have `requires_grad=False`. Independently verified `parameter-count-correction.json` and the corrected external publisher: actual trainable and total counts are separate; cached T1's trainable count is marked `not_measured`. Frozen evidence remains intact, and the correction changes neither optimization nor checkpoints. No unresolved Minor remains.
- The publisher's copied “retain v75” caption was corrected to v76 and independently rechecked; no residual version-label issue found.

**Ready for the sole formal run.** Preserve the locked source/controller/plan files, record the review approval, and perform the planned hash gate before launching.

## Checked boundaries

- Read the approved design/plan, new YAML, reused auxiliary implementation and Stage1/Stage2 integration, controller, preflight, publisher, and existing auxiliary tests. The copied 528 parent source files are byte-identical; the only new source-tree file is the v76 YAML. Independently verified 529 current sources, 3821 protected files, 129 reference locks, controller/plan hashes, **4354 unique frozen paths**, and all 529 source-archive members.
- Runtime architecture is retained v72: repair 420973, original predictor 35427, no v75 context modules. Auxiliary initialization uses isolated seed2022; its CPU initial SHA matches recorded CUDA preflight. Independent original-head SHA equals `fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561`, and the final model contains no auxiliary state.
- Future targets enter auxiliary loss or no-gradient validation diagnostics only. Predictor inputs, repair queries, and observed history-bank staging do not consume future targets. Stage1 checkpoint selection remains original global repair MSE. Auxiliary metadata is checked between payload/manifest; its separately saved state is discarded when initializing Stage2.
- Independently ran six existing nontraining auxiliary tests and five controller checks: **6 passed / 3 training tests deselected; 5 passed**. An actual-architecture CPU forward/backward probe, with zero optimizer steps, confirms epoch10 repair gradient absent; epoch11 hidden gradient sum 0.4022467434, visible gradient exactly zero, finite loss/gradient, preserved RNG, and untouched original head.
- Inspected archived real CUDA preflight: warmup isolation, active hidden gradient, populated observed bank, original-head initialization, and complete Stage2 repair freeze passed. Stage2 retains the original MSE + .2 MAE, uniform weighting, `zero_invalid_v1`, seed2023, batch128 FP32, and 100 epochs. Stage1 also has 100 epochs and the original v72 losses plus .05 auxiliary loss.
- Controller enforces the auxiliary identity and schedule counters: 5940 post-warmup batches and 844800 window occurrences. These count scheduled exposures, not distinct windows or independently measured nonzero gradients on every batch. Dual validation selections are locked before the shared 20 test draws. Publisher includes auxiliary costs within the single Stage1 process and explicitly distinguishes native auxiliary padding from formal zero-invalid padding.

## Limits

This is a focused implementation and pre-run artifact review, with no GPU work, training, source changes, or full-suite rerun. The inherited 1173-pass / 9-known-failure / 1-skip evidence is reused because existing product sources are unchanged; it is not a fresh v76 full-suite result. The three existing tests that train were inspected but not independently rerun. Formal checkpoints, final resource totals, test statistics, and any performance claim require the later trained-artifact review. Reusing the old 20 draws and one seed does not establish untouched-test generalization or superiority to T1.
