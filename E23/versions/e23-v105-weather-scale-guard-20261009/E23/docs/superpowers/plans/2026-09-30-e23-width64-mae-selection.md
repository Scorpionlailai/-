# E23 Width 64 Validation-MAE Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a separately selected width 64 E23 checkpoint using fixed-validation forecast MAE while retaining the same MSE-only Stage 2 training.

**Architecture:** Reuse the selected width 64 Stage 1 checkpoint and all existing P4/model/training settings. Add an explicit Stage 2 checkpoint-selection policy (`mse` default, `mae` new) to the trainer, CLI and paired manifest/checkpoint contract. Run a fresh 100-epoch Stage 2, profile its selected checkpoint, then test it once.

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, pytest, E23 catalog-aware P4 runner.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-width64-mae-selection-design.md`

## Global Constraints

- Locked ETTh1 P4 CSV, validation/test masks, train-only normalization, 96/96 lengths, 16/8 patch, batch 128 and seed 2021 are unchanged.
- Stage 2 still trains for 100 epochs with Adam LR 0.0001 and forecast MSE only, with a fresh optimizer. The selected width 64 Stage 1 manifest and paired H memory/key are reused unchanged.
- The width 64 MSE-selected checkpoint and sole selected test remain immutable. MAE-selected Stage 2/test outputs live only under `e23_multi_expert_width64_mae_v3`.
- Test labels choose no checkpoint or hyperparameter. Baseline SAITS/E22 selected tests are read-only.

## Review Focus

- Old manifests missing `selection_metric` must still load as `mse`; Task 1 tests this.
- A tampered selector in the new manifest must disagree with its checkpoint and fail before test; Task 1 tests this.
- The MAE policy must select the lowest validation MAE even when MSE ranks epochs differently; Task 1 tests this.
- The CLI must pass the requested policy to the trainer, with `mse` as the default; Task 2 tests this.
- Selected evaluation must include the policy and remain write-once over 2,688 windows with unchanged H state; Task 2 and Task 3 check this.

---

### Task 1: Paired Stage 2 selector

**Files:** `integrations/e23/train.py`, `tests/integration/test_e23_stage2.py`.

**Interfaces:** `train_forecast(protocol, settings, *, stage1_manifest, output, device, epochs=100, selection_metric: Literal["mse", "mae"]="mse") -> Path`; `load_stage2_selection(...) -> dict` verifies the manifest/checkpoint selector pair and treats missing old fields as `mse`.

- [x] **Step 1: Write a failing two-epoch synthetic test.** Supply validation rows `(mse=.1,mae=.5)` then `(mse=.2,mae=.4)`; `selection_metric="mae"` must select epoch 2, record both selected metrics and the policy, and restore its checkpoint. Tampering the manifest policy to `mse` must fail before evaluation. An unsupported policy must fail before output creation.
- [x] **Step 2: Verify RED.** Run the specific new test; expect the trainer to reject the unknown `selection_metric` argument.
- [x] **Step 3: Implement.** Select by the requested validation field using strict `<` (earliest tie), save the selector in new checkpoint and manifest, verify it on load, and default absent old metadata to `mse`. Do not change forecast loss, seed, optimizer, H state or validation batch construction.
- [x] **Step 4: Verify GREEN.** Run the focused Stage 2 tests, then the applicable E23 unit/integration suite; check diff and commit.

### Task 2: CLI and selected-result provenance

**Files:** `integrations/e23/run.py`, `tests/integration/test_e23_run.py`.

**Interfaces:** `train-forecast --selection-metric {mse,mae}` defaults to `mse`; `evaluate_selected(...)` copies the verified selection metric into the write-once result.

- [x] **Step 1: Write failing CLI/evaluation tests.** Verify a `main([... --selection-metric mae])` call passes `mae` to the trainer, and a synthetic MAE-selected evaluation includes `selection_metric="mae"`. Existing MSE evaluation reports `mse`.
- [x] **Step 2: Verify RED, then implement the minimal argument pass-through/result field.** Run focused tests, the applicable suite and diff check; commit.

### Task 3: Complete independent MAE-selected trial

**Files:** New ignored `artifacts/.../e23_multi_expert_width64_mae_v3/` jobs/manifests/profile/test; new `docs/experiments/E23_width64_mae_selection_P4_trial.md`.

**Interfaces:** Existing `integrations.e23.run train-forecast` consumes the selected width 64 Stage 1 manifest with `--selection-metric mae`; `profile_e23_selected(..., split="test")`; selected `evaluate` writes exactly once.

- [x] **Step 1: Launch the full Stage 2 through `scripts/run_background.py`.** Require exit 0, 100 epochs, MAE-selected manifest, paired H hashes and no reuse of the old width 64 Stage 2 output.
- [x] **Step 2: Profile on idle GPU 0 with batch 128, warmup 5, timed 10 fixed test input batches, without computing metrics.** Save timing, cold start, memory and hashes.
- [x] **Step 3: Execute the only MAE-selected test through `scripts/run_background.py`.** Require 2,688 windows, both errors/DRRs, selector provenance and unchanged H state; never overwrite or reselect by test.
- [x] **Step 4: Write the report from generated JSON.** Compare accuracy with width 64 MSE-selected E23, SAITS and E22; label Stage 1 reuse and speed limits, verify every number/hash, run final tests and commit.

## Completion

The v3 Stage 2 selected epoch 87 by validation MAE after all 100 epochs. Its one selected test gave MSE `0.427264729` and MAE `0.434448576`, both below the archived SAITS values. Artifact identity checks and the applicable suite passed (`589 passed`, five inherited CUDA warnings). The report at `docs/experiments/E23_width64_mae_selection_P4_trial.md` records the adaptive reuse of the same test set after v2's result; independent held-out confirmation remains future work.
