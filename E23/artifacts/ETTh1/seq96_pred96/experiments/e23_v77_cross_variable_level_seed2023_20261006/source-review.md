# v77 preliminary source review

Independent read-only review using requesting-code-review; scope: candidate source and nontraining CPU behavior. No GPU, optimizer steps, training, full-suite rerun, source edits, or formal-run approval. The sole written artifact is this report.

## Assessment

Preliminary source review is clean: Critical 0, Important 0, Minor 0. Source is ready for the next preparation gate, not approved for formal training. Parent/v76 auxiliary choice, final configuration, controller/publisher, actual CUDA preflight and immutable pre-test locks remain pending.

## Requirements and change boundary

Read design `/home/xyw/paper/E23/docs/experiments/2026-10-06_E23_v77_cross_variable_level_design.md` and plan `/home/xyw/paper/E23/docs/superpowers/plans/2026-10-06-e23-v77-cross-variable-level.md`.

Recomputed all529 hashes in `copied-parent-sources.json`: only four existing files differ: `integrations/e23/config.py`, `integrations/e23/train.py`, `src/malr/models/e23/model.py`, `src/malr/models/e23/recent_mean.py`. Compared each changed file with an available source matching the frozen parent hash; no unrelated edits in these diffs. The 525 remaining listed files match their recorded parent hashes. New behavior test is additional to the parent inventory.

## Independently verified

- Ran the first five new tests only: `PYTHONPATH=src:. python -B -m pytest -p no:cacheprovider tests/integration/test_e23_cross_variable_level.py -q -k 'not original_repair_losses_train and not checkpoint_roundtrip'` -> 5 passed,2 deselected in1.84s. This verifies old state tensors, CPU RNG, head initialization and zero-start function equality; direct cross-variable access; global output units; hidden/invalid NaN and Inf isolation; empty-window zero new delta.
- Additional CPU probes used forward passes only. Actual repair/head parameter counts are424238/35427. The new network contributes3265 parameters and accepts49 features. Confirmed ordering of whole96 means, zero all six summaries for an entirely unavailable source variable,42 zero context features for a fully empty window, and target onehot7.
- `recent_mean.py:25-27` masks hidden/invalid payload before arithmetic. Features derive from original observed sequence, visible masks and existing query mean/std, not reconstructed hidden values or clean/future targets. Counts and clamped denominators handle empty recent/whole supports; absent sources have all six summaries zero. The branch is shared across targets with target identity and all-seven-variable summaries.
- `recent_mean.py:73-84` preserves the original mean_v1 computation and adds bounded nominal global delta directly in train-standardized units. Only valid hidden coordinates in the final32 points change; observed positions, first64 points, coverage and validity remain intact. A separate count1 probe confirmed global point change equals4*nominal delta, share cap4, and no change anywhere else. The design correctly says nominal delta is not the actual aggregate mean change when fewer than8 hidden points exist.
- New final linear output is zero initialized; nested CPU fork_rng preserves all original initialization draws. The original network/module keys and predictor code are unchanged; the independent test verifies their tensors and initial forward function.
- Static checkpoint review: the added seven global identity fields feed `_trial_identity`; both `_verify_identity` and `_verify_trial_payload` iterate expected fields. Stage1 and Stage2 loaders call these checks and strict state loading; frozen Stage2 compares every repair tensor with Stage1. New metadata includes49 features,3265 parameters, train_standardized units, bound2, source description, empty-window rule and summary order. Training save paths include the identity in payload and manifest. No missing new identity check found.

## Author evidence inspected, not rerun

`red-tests.log`: all7 new behaviors fail before implementation. `green-tests.log`:15 new/old recent-mean tests pass. `full-suite.log`:1180 passed,9 failed,1 skipped. Failure names agree with the9 inherited names recorded in v75 `test-audit.json`; v76 `auxiliary-test-audit.json` states no product changes and9 auxiliary checks passed. These are recorded author results, not an independently green full suite. The two optimizer/checkpoint roundtrip tests were deliberately not rerun under the no-training instruction.

## Declined to judge / remaining gate

- Final parent and Stage1 auxiliary inheritance: pending v76 fixed-validation ruling, explicitly outside this preliminary review. Source access and synthetic probes do not establish a winning model.
- Formal controller, final config, publisher, actual historical CUDA gradient/freeze/head preflight, resource accounting and pre-test locks: not yet finalized; must be reviewed after preparation.
- Trained accuracy, test-based promotion, T1 comparison, generalization and sealed confirmation: no v77 training/results yet; no claims approved here.

No source findings were deferred. No source or training artifacts were changed.
