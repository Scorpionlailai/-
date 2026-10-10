# E23 v3 Targeted Accuracy Iterations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a Stage 2 MAE-aligned loss, window-conditioned V selection and two-group H reranking in that order while preserving selected v3 and the fixed width-16 predictor.

**Architecture:** Trial 1 adds one paired Stage 2 loss mode and reuses v3 Stage 1. Trial 3 adds a config-gated sparse V selector and fresh two-stage training. Trial 4 adds a config-gated H shortlist/rerank with four final candidates and fresh two-stage training. Each trial starts from the current validation winner, and one final selected test is allowed only after all validation decisions.

**Tech Stack:** Python, PyTorch, pytest, ETTh1 P4, CUDA.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-v3-targeted-accuracy-iterations-design.md`

## Global Constraints

- Original v3 selected artifacts and default config bytes remain immutable; unrelated local CSV remains untouched. Only scope listed files in commits.
- Fixed P4 geometry, masks, seed 2021, batch 128, 100 epochs per trained stage, and predictor architecture/width 16.
- Candidate gate: selected validation MAE strictly below current winner; selected validation MSE `<= 0.7647695325129157`. Start from v3 MAE `0.5904396656148828`.
- Trial 1: exact loss `MSE + 0.2*MAE`, one fresh Stage 2 from selected v3 Stage 1. Trials 3/4: fresh Stage 1/2, inheriting only the winner's loss/V mode as specified.
- At most one new ETTh1 test and test-input profile after all validation trials; none if v3 remains winner.

## Review Focus

- Old `forecast_mse_only` manifests/checkpoints still load and produce exact v3 validation metrics; test in Task 1.
- New loss manifests/checkpoints reject tampering and wrong mode before any evaluation batch; test in Task 1.
- V contextual source scores use observed values only and cannot read hidden or padding values; test in Task 2.
- H shortlist includes no future or wrong-variable member and never gives invalid members weight; test in Task 4.
- Conditional v13 config combines exactly the selected V/loss modes and cannot load another variant's checkpoint; test in Tasks 4/5.

---

### Task 1: Stage 2 loss mode and v11 validation trial

**Files:** `integrations/e23/train.py`, `integrations/e23/run.py`, `tests/integration/test_e23_stage2.py`, ignored v11 artifacts, report appendix.

**Interfaces:** `stage2_objective(forecast, target, *, loss_mode="mse")`; `train_forecast(..., forecast_loss="mse")`; CLI `--forecast-loss {mse,mse_plus_0p2_mae}`. The new mode records the same exact value in manifest and checkpoint; the old mode uses original field values/defaults.

- [ ] Add RED arithmetic/gradient and manifest tamper tests, including old v3 selected loader compatibility. Run focused tests and inspect intended failures.
- [ ] Implement `MSE + 0.2*MAE`, CLI choice and paired mode checking; preserve original MSE path. Run focused tests, then full suite; commit code/tests.
- [ ] Train 100-epoch Stage 2 from original v3 Stage 1 in a fresh ignored `e23_v3_mixed_loss_v11` root, selecting validation MAE. Verify pairing, 100 epochs, finite metrics and H immutability; record decision against current winner.

### Task 2: Contextual V source selection and strict identity

**Files:** `src/malr/models/e23/variable.py`, `model.py`, `integrations/e23/config.py`, `train.py`, new v12 config, unit/integration tests.

**Interfaces:** `SparseVariableExpert(..., selection="global_topk"|"window_context_topk_v1")`; the new selector returns four sources per `[B,C,N]` target and uses the spec's masked prior/visibility/consistency score. `E23Settings.variable_selection` defaults to `global_topk`; trial identity is paired in both stages.

- [ ] Add RED tests for per-window source changes, no self, stable ties, rotating Stage 1 exploration, hidden/padding invariance and all-invalid V; run and inspect intended failures.
- [ ] Implement the cheap six-source selector before the existing four-source candidate MLP; keep global top-k/top-p branches exact. Add strict config and paired identity tests (including wrong config before batches); run focused/full suite and commit.
- [ ] Run real P4 CUDA one-batch forward/backward smoke with finite gradients and memory peak.

### Task 3: v12 two-stage validation trial

- [ ] Choose v12 Stage 2 loss from Task 1's recorded winner. Train new Stage 1/2 for 100 epochs each in a fresh ignored v12 root; select Stage 1 by hidden repair MSE and Stage 2 by forecast MAE.
- [ ] Verify full epoch logs, selected paired hashes, unchanged H state and finite validation; record candidate-source distribution and gate decision against current winner. Commit a tracked validation report update.

### Task 4: Two-group causal H shortlist and visible-error rerank

**Files:** `src/malr/prototypes/tensor_index.py`, `src/malr/models/e23/historical.py`, `model.py`, `integrations/e23/config.py`, `train.py`, new v13 config, unit/integration tests.

**Interfaces:** `TensorPrototypeIndex.query_two_groups(..., member_topk=4) -> TensorRetrieval` returns eight stable slots; `HistoricalExpert(..., retrieval_strategy="one_group"|"two_group_visible_rerank_v1")` reduces eligible shortlisted members to four before messages. Original `query` is untouched.

- [ ] Add RED tests for two-group causal/same-variable shortlist, tie order, rerank by visible error, no valid member, fixed four final slots and no hidden-value effect; run and inspect intended failures.
- [ ] Implement new index method/H strategy, strict config, paired identity and wrong-variant guard; run focused/full suite and commit.
- [ ] Run real P4 CUDA one-batch forward/backward smoke, finite gradients and memory peak.

### Task 5: v13 trial, final decision and review

- [ ] Choose v13 V and Stage 2 loss modes from the recorded winner; train new Stage 1/2 for 100 epochs each in a fresh ignored v13 root.
- [ ] Verify manifests, hashes, selected metrics, H immutability and candidate diagnostics. Compare v11/v12/v13 to v3 under the predeclared gate. If a non-v3 winner remains, run exactly one selected test and fixed-input profile; otherwise no test.
- [ ] Write a tracked experiment report with full validation metrics, winner/rollback, artifact hashes, candidate distributions, resource caveats and test-adaptivity limits. Run final full suite and one fresh whole-branch review; fix material findings, commit and preserve the unrelated CSV.
