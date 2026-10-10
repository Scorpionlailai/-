# E23 Grouped Historical and Current Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate one E23 variant whose historical H/V and current T/V evidence are routed as two groups.

**Architecture:** H starts anchor train-only aligned historical cross-variable patches; a new historical V encodes them. The existing V and T remain current-window experts. A two-level router combines leaf messages into group residuals and then chooses none/history/current.

**Tech Stack:** Python 3.11, PyTorch, pytest, ETTh1 P4, CUDA.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-grouped-historical-current-recovery-design.md`

**Final status:** The planned trial was completed, failed the predeclared validation gate, and its grouped code was subsequently reverted at the user's request. The checked tasks record the trial, not the active architecture.

## Global Constraints

- Existing flat E23 configs/checkpoints and H/V/T behavior must remain loadable.
- The historical archive is exactly train-only standardized source data, identity-bound to P4; no validation/test rows or target values.
- Selected historical starts must satisfy patch-end `<=` query start and train end.
- Current V and T use original observed current-window inputs only.
- Width 64, historical topk 4, variable topk 4, patch 16/stride 8, batch 128, seed 2021, 100+100 epochs, fresh optimizer and forecast MSE loss remain locked.
- Test only after fixed validation MAE `<0.5904396656148828` and MSE `<=0.7647695325129157`.

## Review Focus

- H candidate start zero when invalid: historical V must mask it, never treat it as a valid training patch.
- Causal boundary at exactly `start + patch_len == query_start`: allowed; one step later: rejected.
- Prototype source and archive source mismatch: reject or prove the aligned target patch identity.
- No available group and one available group: finite softmax and exact zero unavailable weights.
- Stage 1 and Stage 2 architecture/config mismatch: reject before reading validation or test batches.

### Task 1: Historical aligned variable expert

**Files:** Create `src/malr/models/e23/historical_variable.py`; modify `src/malr/models/e23/historical.py`; tests in `tests/unit/test_e23_historical_variable.py`.

**Interfaces:** `HistoricalVariableExpert(variable_count, patch_length, d_model, topk, archive)`; `forward(inputs: ExpertInputs, historical: ExpertEvidence, *, epoch: int, explore: bool) -> ExpertEvidence`; H provenance gains candidate raw patches.

- [x] Write RED tests with two-variable deterministic archive: valid H anchor reads aligned other-variable history, changed current source values do not change historical V, invalid/future/out-of-train anchors give zero evidence, gradients reach relation/message parameters, and archive is not a persistent trainable state.
- [x] Run focused test and confirm expected missing-symbol/behavior failure.
- [x] Implement minimal historical V gathering and H provenance; mask invalid and out-of-bounds starts before softmax, use source query normalization, expose starts/source variables/candidate count.
- [x] Run focused and existing H/V tests; commit.

### Task 2: Group routing and encoder

**Files:** Create `src/malr/models/e23/grouped_router.py`; modify `src/malr/models/e23/model.py`, `contracts.py`; tests in `tests/unit/test_e23_grouped_router.py` and `tests/integration/test_e23_grouped_model.py`.

**Interfaces:** `GroupedExpertRouter.forward(inputs, evidence) -> RouterOutput`; `MultiExpertRepairEncoder(..., architecture="flat_hvt_v1", historical_archive=None)`; grouped output evidence keys `historical`, `historical_variable`, `temporal`, `variable`, and router weights ordered `none, historical_group, current_group`.

- [x] Write RED tests for leaf/group availability, exact empty fallback, only one group usable, weight sums and finite nonzero gradients to both groups.
- [x] Run focused tests and confirm failure.
- [x] Implement hierarchical router and grouped model branch; apply specialist exposure at group level, preserving original flat branch and observed-value clamp.
- [x] Run focused and existing model/router tests; commit.

### Task 3: Config, paired training and reporting

**Files:** Modify `integrations/e23/config.py`, `train.py`, `run.py`, `profile_reference.py`; create `configs/e23_grouped_history_current_p4_width64.yaml`; integration tests.

**Interfaces:** New strict architecture config value; `build_repair_model` reconstructs train archive from P4, all existing train/load/evaluate/profile entrypoints accept it through settings.

- [x] Write RED tests for grouped config, Stage 1/2 one-epoch pairing, strict architecture mismatch and archive identity, grouped selected diagnostics/profile, plus legacy regression.
- [x] Run tests and confirm failure.
- [x] Implement config/model wiring and dynamic group/leaf reporting; keep legacy records defaulting to flat architecture.
- [x] Run focused and full applicable test suite; commit.

### Task 4: Fixed trial and evidence

- [x] Run fresh grouped Stage 1 for all 100 epochs; verify selected H memory/key pair, config/source hashes and finite validation repair MSE.
- [x] Run fresh grouped Stage 2 for all 100 epochs with MAE selection and no forecast mixer; verify paired hashes and chosen validation MSE/MAE.
- [x] Apply the predeclared dual-metric validation gate. If it passes, profile fixed test inputs and evaluate exactly one selected test; otherwise do not touch the test split.
- [x] Write `docs/experiments/E23_grouped_history_current_P4_trial.md` with result, provenance, expert/group diagnostics, accuracy/speed comparisons if available, and test-set reuse limit; run final verification and commit.
