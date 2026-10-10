# E23 Forecast-Side Variable Mixer Implementation Plan

**Goal:** Test whether a small patch-local cross-variable attention layer improves the width 64 E23 forecast while preserving the selected Stage 1 repair state and locked ETTh1 P4 protocol.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-forecast-variable-mixer-design.md`.

**Environment:** `/home/xyw/paper/E23`, Python `/home/xyw/miniforge3/envs/malr/bin/python`, `PYTHONPATH=.:src`, CUDA GPU 0. Use fresh ignored `artifacts/.../e23_multi_expert_width64_variable_mixer_v4/`; do not overwrite v2/v3 or baseline artifacts.

## Task 1: Optional forecast mixer model

**Files:** `src/malr/models/repaired_forecast.py`, `integrations/e23/train.py`, focused model tests.

- [x] Write RED tests proving that the mixer starts as an exact identity on the old forecast path, changes one variable's prediction when another variable's patch changes after its gate opens, and receives a finite gradient under forecast loss.
- [x] Implement one LayerNorm + four-head self-attention across the variable axis at each patch, followed by a zero-initialized scalar residual gate. Preserve the no-mixer module/state_dict path by default.
- [x] Expose `forecast_mixer="none" | "patch_attention_v1"` in `build_forecast_model`; run focused tests and inspect the diff.

## Task 2: Paired architecture provenance and CLI

**Files:** `integrations/e23/train.py`, `integrations/e23/run.py`, `integrations/e23/profile_reference.py`, integration tests.

- [x] Write RED tests for Stage 2 training with the new flag, manifest/checkpoint pairing and tamper rejection, old missing-field default `none`, CLI forwarding, selected evaluation and profile reconstruction.
- [x] Add `train_forecast(..., forecast_mixer="none")`, `train-forecast --forecast-mixer`, and paired mixer identity to checkpoint/manifest. Verify the pair in `load_stage2_selection`; default old missing fields to `none`.
- [x] Build the verified selected architecture in evaluation and profiling; record the mixer in selected evaluation. Run focused and applicable unit/integration suites, check diff, and commit.

## Task 3: Fixed validation trial and gated test

- [x] Launch fresh 100-epoch Stage 2 from the selected width 64 Stage 1 on GPU 0, with `--selection-metric mae --forecast-mixer patch_attention_v1`. Require exit 0, 100 epochs, paired hashes and a new output root.
- [x] Compare selected fixed-validation forecast MAE/MSE with v3. Run selected test only if MAE `< 0.5904396656148828` **and** MSE `<= 0.7647695325129157`.
- [x] If the gate passes, profile the selected model on fixed test inputs (batch 128, warmup 5, timed 10) and run exactly one selected test over 2,688 windows. If it fails, stop without a test.
- [x] Write a trial report from generated artifacts, including the adaptive test-set reuse limitation, verify all hashes and values, run final applicable tests and commit.

## Completion

The 100-epoch v4 selected checkpoint failed both predeclared validation gates (MAE `0.591542694`, MSE `0.773467380` versus v3 `0.590439666`/`0.764769533`). No v4 selected test or test-input speed profile was run. The read-only validation ablation, artifact hashes and unchanged H state are recorded in `docs/experiments/E23_width64_variable_mixer_P4_validation_trial.md`; the applicable suite passed 595 tests with five inherited CUDA warnings.
