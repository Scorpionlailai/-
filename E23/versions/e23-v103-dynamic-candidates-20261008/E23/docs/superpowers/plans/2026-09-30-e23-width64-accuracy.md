# E23 Width 64 Accuracy Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a wider E23 repair encoder improves the locked ETTh1 P4 forecast accuracy beyond selected SAITS and E22 references.

**Architecture:** Reuse the reviewed explicit H/V/T repair and forecast pipeline, changing only the repair latent width from 16 to 64 in a separate E23 config and artifact root. Keep the predictor at the inherited 16-channel PatchTST width. Train both 100-epoch stages, select only by fixed validation metrics, then evaluate the selected model once.

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, pytest, RTX 5090, existing E23 P4/catalog runner.

**Spec:** `docs/superpowers/specs/2026-09-30-e23-width64-accuracy-design.md`

## Global Constraints

- ETTh1 P4 source hash `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; fixed validation/test mask hashes `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c` and `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`.
- P4 lengths `96/96`, patch `16/8`, batch `128`, seed `2021`, separate Stage 1/2 `100` epochs. Stage 1 LR `0.001`, Stage 2 LR `0.0001`; no selected-test tuning.
- E22/SAITS and their archived artifacts remain read-only. Width 16 E23 checkpoints, manifests and the single selected test remain untouched.
- Accuracy is the primary goal. Report training/inference speed without requiring a SAITS speed win.

## Review Focus

- Width 64 config must keep the same P4 source and fixed masks; Task 1 checks identity.
- Width 16 Stage 1 must be rejected when Stage 2 expects width 64; Task 1 checks config mismatch before a batch.
- Repair width 64 must still feed the width 16 predictor with the original mask and correct forecast shape; Task 1 checks a synthetic forward.
- The published H bank key width must match the width 64 frozen encoder; Task 2 checks paired checkpoint restoration.
- Width 64 selected evaluation must write once, cover 2,688 windows, and leave H state unchanged; Task 3 checks the output record.

---

### Task 1: Permit the reviewed width 64 variant

**Files:**
- Create: `configs/e23_multi_expert_p4_width64.yaml`
- Modify: `integrations/e23/config.py`
- Test: `tests/integration/test_e23_config.py`
- Test: `tests/integration/test_e23_model.py`

**Interfaces:** `load_e23_settings(path: Path, *, protocol: P4Protocol) -> E23Settings` accepts only `d_model in {16, 64}` while all other locked fields remain exact. `build_forecast_model(protocol, settings, *, device)` already uses the E23 repair width and unchanged P4 predictor width.

- [ ] **Step 1: Write failing tests.** Load the new width 64 YAML and assert all settings except `d_model` equal width 16, different config hashes, correct repair/predictor widths, exact original-mask clamp and forecast shape on a two-window synthetic batch. Assert width 32 and data overrides reject.
- [ ] **Step 2: Verify RED.** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/integration/test_e23_config.py tests/integration/test_e23_model.py -q`; expect the new width 64 load to fail.
- [ ] **Step 3: Implement.** Add the new YAML with only `model.d_model: 64` changed, and expand the strict parser's allowed widths to `{16, 64}`. Do not alter P4 loader, experts, losses or predictor.
- [ ] **Step 4: Verify GREEN.** Run the focused tests, then the applicable `tests/unit tests/integration` suite and the artifact-free P4 protocol test. Check `git diff --check` and commit only the config/parser/tests.

### Task 2: Train and select the two stages

**Files:** New ignored outputs under `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_width64_v2/`.

**Interfaces:** Existing `integrations.e23.run train-repair` and `train-forecast` CLIs; both use the new E23 config, the same P4 protocol YAML, E22 migration catalog and fixed mask root.

- [ ] **Step 1: Run a synthetic width 64 forward/backward.** Check finite gradients, `[B,7,96]` repair and `[B,7,96]` forecast shapes, exact observation clamp, and H index key width 64.
- [ ] **Step 2: Launch Stage 1 through `scripts/run_background.py`.** Use `--device cuda:0`, the new config, 100 epochs and a fresh output/job directory. Inspect `status.json` exit 0 and selected manifest with source/config/mask/key/memory hashes.
- [ ] **Step 3: Launch Stage 2 through `scripts/run_background.py`.** Consume only the selected Stage 1 manifest, use a fresh output/job directory, and inspect exit 0, 100 completed epochs, reset optimizer, frozen H hashes and fixed-validation MSE selection. Reject a width 16 Stage 1 manifest under the width 64 config before a batch.
- [ ] **Step 4: Record validation results and timing.** Compare the generated width 64 manifests against the width 16 and SAITS records; no test labels are read at this step.

### Task 3: Profile, evaluate once and report

**Files:**
- Create: `docs/experiments/E23_width64_P4_trial.md`
- Update: `docs/experiments/E23_multi_expert_P4_trial.md` only if its width 16 record still awaits completion.

**Interfaces:** Existing `profile_e23_selected(..., split="test")` profiles inputs without metrics; `integrations.e23.run evaluate --selection ... --output ...` writes the selected test record once.

- [ ] **Step 1: Profile selected width 64 on the fixed test input order.** GPU 0, batch 128, warmup 5, timed 10; save full-pipeline/model-core time, cold start, peak memory, H-state identity and environment.
- [ ] **Step 2: Execute the only selected width 64 test with `scripts/run_background.py`.** Require exit 0, 2,688 windows, source/mask/config/checkpoint hashes, finite MSE/MAE, both shared-reference DRRs and unchanged H state. Do not overwrite or rerun it.
- [ ] **Step 3: Write both trial reports from JSON.** State the width 16 and width 64 accuracy values against SAITS MSE `<0.4405843816917229` and E22 MAE `<0.4332482160624463`; include controlled speed as context and provenance. Commit the reports after checking every number against manifests and benchmark JSON.
