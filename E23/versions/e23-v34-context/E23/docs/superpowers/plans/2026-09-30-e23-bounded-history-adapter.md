# E23 Bounded Historical-Variable Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test a bounded historical-variable input transform while preserving the flat v3 parent and v6 adapter behavior.

**Architecture:** `aligned_bounded_v2` applies `8*tanh(x/8)` to normalized historical source/target scalars before the existing adapter MLP. All remaining structure and training are the v6 zero-gated adapter.

**Tech Stack:** Python, PyTorch, pytest, ETTh1 P4, CUDA.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-bounded-history-adapter-design.md`

## Global Constraints

- `none` and `aligned_v1` outputs/checkpoint state remain unchanged; v7 uses a new paired identity.
- Candidate alignment, train-only archive, strict pre-window cutoff, frozen v3 parent and epoch-0 exact baseline remain mandatory.
- Train 100 epochs, seed 2021, batch 128, adapter-only Adam LR `0.001`, forecast-MSE loss, MAE selection over epochs 0..100.
- Selected test gate remains MAE `<0.5904396656148828` and MSE `<=0.7647695325129157`.

## Review Focus

- Values above 100 or 2559 remain finite and at most 8 after the transform (float32 `tanh` can round to exactly 1).
- `aligned_v1` is unaffected, including its saved v6 checkpoint reconstruction.
- Tampering variant identity is rejected before selected evaluation reads data.
- Parent H bank/key, all flat v3 weights, archive source and mask identities are unchanged.
- Epoch-0 selected adapter is exactly the v3 forecast, so a failed trial leaves the baseline selected.

### Task 1: Bounded optional expert path

**Files:** `src/malr/models/e23/historical_variable.py`, `model.py`, `integrations/e23/train.py`, unit/integration tests.

- [x] Write RED tests for bounded extreme inputs and small-value behavior, v1 unchanged output, and zero-gate baseline equality with the v2 flag.
- [x] Run RED tests; add the optional fixed transform and `aligned_bounded_v2` model-builder path.
- [x] Run focused and legacy model tests; commit.

### Task 2: Variant provenance and reconstruction

**Files:** `integrations/e23/train.py`, `run.py`, `profile_reference.py`, integration tests.

- [x] Write RED tests for adapter-variant CLI forwarding, v7 manifest/checkpoint pairing, tamper rejection, selected evaluation/profile reconstruction, and v6/legacy loading.
- [x] Run RED tests; thread the explicit variant through training, loading, evaluation and profiling.
- [x] Run applicable and full suite; commit.

### Task 3: Fixed v7 trial and report

- [x] Train one fresh 100-epoch v7 adapter from the selected v3 Stage 2 checkpoint; verify epoch 0, frozen parent/H hashes and finite validation.
- [x] Apply the predeclared dual-metric gate. It failed; no selected ETTh1 test or test-input profile was run.
- [x] Run validation-only gate/availability diagnostics, write a tracked trial report with hashes and v3/v6 comparison, run final verification, commit and review.
