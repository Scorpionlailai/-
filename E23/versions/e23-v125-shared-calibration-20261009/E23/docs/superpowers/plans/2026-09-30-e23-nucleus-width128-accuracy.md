# E23 Top-p and Width 128 Accuracy Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare E23 v3 against H/V top-p selection, repair width 128 and their combination on the locked ETTh1 P4 protocol.

**Architecture:** Add a deterministic nucleus prefix mask to H/V candidate selection under new explicit config identities. Preserve exact legacy top-k behavior and v3 selected checkpoint reconstruction. Train three independent two-stage variants and gate one selected test by predeclared validation metrics.

**Tech Stack:** Python, PyTorch, pytest, ETTh1 P4, CUDA.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-nucleus-width128-accuracy-design.md`

## Global Constraints

- Existing v3 selected checkpoint, config, masks and outputs remain immutable; the unrelated local CSV edit is untouched.
- Top-p is `p=0.9` for H and V. H retrieval compute cap is eight candidates within one best causal group; V scores all six other variables. T remains unchanged.
- The original top-k path must keep exactly four H and four V candidates and bitwise-compatible state keys/outputs.
- Repair/key/message width 128 starts Stage 1 from scratch; predictor remains width 16 and the 96/96, patch16/stride8 P4 geometry stays locked.
- Each new variant runs Stage 1/2 for 100 epochs; Stage 1 selects validation hidden repair MSE, Stage 2 trains forecast MSE and selects validation forecast MAE.
- Only one variant passing validation MAE `<0.5904396656148828` and MSE `<=0.7647695325129157` may receive one selected test and profile.

## Review Focus

- Invalid H/V candidates never enter the top-p probability denominator or get positive weights.
- Equal scores yield stable, deterministic prefixes, and a nonempty eligible pool always selects at least one candidate.
- Empty candidate pools give unavailable evidence, allowing the router's `none` path.
- The old v3 manifest still reconstructs its exact width64/top-k forecast and H state.
- New config/manifest identities prevent a D128/top-p checkpoint from loading as another variant.

### Task 1: Nucleus prefix helper and H selection

**Files:** `src/malr/models/e23/nucleus.py`, `historical.py`, tests/unit.

- [x] Write RED tests for p=0.9 cumulative prefix, ties, invalid slots, minimum one, empty pool, and H top-p candidate count/causal mask. Run the focused tests and inspect expected failures.
- [x] Implement `nucleus_prefix(logits: Tensor, eligible: Tensor, *, p: float) -> Tensor` using stable sorted order and masked softmax; introduce optional H top-p with compute cap eight, preserving the old path when absent.
- [x] Run focused H/index/legacy tests, inspect exact v3 checkpoint reconstruction, commit.

### Task 2: V top-p and width128 construction

**Files:** `src/malr/models/e23/variable.py`, `model.py`, `integrations/e23/config.py`, `train.py`, new variant YAMLs, tests.

- [x] Write RED tests for directed six-source top-p, self-exclusion, Stage 1 exploration, no valid local sources, width128 repair versus width16 predictor and strict config identities. Run expected failures.
- [x] Thread optional top-p through V and model builders; add D64/top-p, D128/top-k and D128/top-p configs with the same locked training fields. Keep original config schema and state keys intact.
- [x] Run focused and full applicable tests; commit.

### Task 3: Trial identity, training smoke and legacy compatibility

**Files:** `integrations/e23/train.py`, `run.py`, `profile_reference.py`, integration tests and ignored trial artifacts.

- [x] Write RED tests proving new selection manifests/checkpoints carry variant identity and reject wrong config/architecture before evaluation reads batches; verify old v3 loader unchanged.
- [x] Implement paired provenance checks for selection mode/p/cap and width; run focused tests and full suite.
- [x] Run a D128 one-batch CUDA forward/backward memory smoke under the real P4 input and verify finite loss/gradients; commit code and tests.

### Task 4: Three controlled 100+100 validation trials

- [x] Train `D64/top-p`, `D128/top-k`, `D128/top-p` in independent ignored artifact roots, each with fresh Stage 1 and Stage 2. Verify 100 epochs per stage, hashes, immutable H state and finite validation.
- [x] Compare selected validation metrics with the v3 dual gate; pick at most one winner by lowest validation MAE, using the fixed variant order for ties. Only if eligible, run one selected ETTh1 test and fixed test-input profile.
- [x] Produce a tracked experiment report with validation/test metrics, candidate distributions, resource use, artifact hashes, negative findings and test-adaptivity caveat. Run final focused/full verification and whole-branch review; commit.
