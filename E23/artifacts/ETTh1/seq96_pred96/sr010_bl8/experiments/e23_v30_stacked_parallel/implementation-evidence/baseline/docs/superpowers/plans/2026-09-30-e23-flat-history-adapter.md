# E23 Flat Historical-Variable Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether train-only aligned historical cross-variable evidence improves the selected flat E23 v3 forecast without changing its H/V/T router or trained baseline path.

**Architecture:** Add an optional zero-gated historical-variable residual after the v3 router. Initialize all old modules from the selected v3 Stage 2 checkpoint, freeze them, train the adapter alone, and select from epochs 0–100 by fixed-validation MAE.

**Tech Stack:** Python, PyTorch, pytest, locked ETTh1 P4, RTX 5090.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-flat-history-adapter-design.md`

## Global Constraints

- Default E23 construction and old checkpoints remain bitwise unchanged with `history_adapter="none"`.
- Historical-variable candidates require valid H provenance, target/archive identity, candidate end `<=` lookback-window start and `<=` train end.
- The selected v3 Stage 2 model, H bank and key stay frozen; only adapter parameters optimize.
- Adapter training: fresh Adam, LR `0.001`, zero weight decay, seed `2021`, batch `128`, `100` epochs, forecast-MSE loss; earliest fixed-validation MAE minimum over epochs `0..100`.
- Selected test only if validation MAE `<0.5904396656148828` and MSE `<=0.7647695325129157`.

## Review Focus

- Invalid H candidate start zero must not become a real archive candidate.
- Candidate ending exactly at window start is allowed; one step later is rejected, even if before the queried patch.
- Archive source rows and H target prototype must refer to the same training timestamp and raw standardized scale.
- Zero gate must preserve v3 exact predictions; unavailable evidence must add no correction despite MLP bias.
- Parent Stage 2 checkpoint, H key/bank, source/mask/config hashes and adapter architecture must be verified before any selected evaluation batch.

### Task 1: Optional historical-variable residual

**Files:** `src/malr/models/e23/historical.py`, new `src/malr/models/e23/historical_variable.py`, `src/malr/models/e23/model.py`, `integrations/e23/train.py`, focused unit/integration tests.

**Interfaces:** `HistoricalVariableExpert.forward(inputs, historical, *, epoch, explore) -> ExpertEvidence`; `HistoricalVariableAdapter.forward(routed_latents, inputs, historical) -> (delta, evidence)`; `MultiExpertRepairEncoder(..., historical_archive=None)`; `build_forecast_model(..., history_adapter="none" | "aligned_v1")`.

- [ ] Write RED tests for alignment, strict pre-window cutoff, invalid and mismatched H candidates, no current-source dependence, directed-relation gradients, unavailable zero correction and exact zero-gate equivalence to v3.
- [ ] Run focused RED tests and confirm failures are caused by missing behavior.
- [ ] Implement the minimal optional expert/adapter path after the flat router; default path must retain all legacy state keys.
- [ ] Run focused and legacy model/H tests; commit.

### Task 2: Parent-paired adapter training and selection

**Files:** `integrations/e23/train.py`, `integrations/e23/run.py`, focused integration tests.

**Interfaces:** `train_history_adapter(protocol, settings, *, parent_selection: Path, output: Path, device: torch.device, epochs: int=100) -> Path`; existing `load_stage2_selection` recognizes optional `history_adapter`/parent fields; CLI `train-adapter --parent-selection`.

- [ ] Write RED tests for strict parent identity, old selection default `none`, frozen base weights/H state, adapter-only gradients, epoch-0 fallback and paired manifest/checkpoint tamper rejection before batches.
- [ ] Run RED tests; implement training from the selected v3 Stage 2, with epoch-0 evaluation and only adapter optimizer parameters.
- [ ] Run focused Stage 1/2 and new adapter tests; commit.

### Task 3: Selected reconstruction and diagnostics

**Files:** `integrations/e23/run.py`, `integrations/e23/profile_reference.py`, integration tests.

**Interfaces:** `evaluate_selected` and `profile_e23_selected` reconstruct `history_adapter` from the paired selected record; report archive/H hashes and adapter gate/availability when present.

- [ ] Write RED tests for selected adapter reconstruction, paired identity rejection before test batches, old selected evaluation, and profile timing segment.
- [ ] Run RED tests; implement model reconstruction, adapter diagnostics and read-only H checks.
- [ ] Run focused and full applicable suite; commit.

### Task 4: Controlled v6 trial

- [x] Create fresh `e23_flat_history_adapter_width64_v6` artifacts and train 100 epochs from v3 selected Stage 2; confirm epoch-0 exact v3 validation metrics, parent hashes, frozen state and finite training.
- [x] Apply the predeclared dual-metric validation gate. It failed, so no selected ETTh1 test or test-input profile was run.
- [x] Inspect validation-only historical-variable availability and input range. Epoch 0 is the zero-gate ablation and exactly matches v3; selected gates are all zero. No speed comparison was run because the gate failed.
- [x] Write the tracked experiment report, verify hashes and focused tests. Whole-branch review follows the bounded-input follow-up trial.
