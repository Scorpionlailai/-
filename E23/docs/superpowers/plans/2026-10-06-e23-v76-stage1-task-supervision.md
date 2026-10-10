# E23-v76 Implementation Plan

> For agentic workers: use superpowers:executing-plans inline.

**Goal:** Train useful repair information before frozen formal prediction training.
**Architecture:** Reuse existing training-only auxiliary futureMSE, finalhead untouched.
**Tech Stack:** Python/PyTorch.
**Spec:** /home/xyw/paper/E23/docs/experiments/2026-10-06_E23_v76_stage1_task_supervision_design.md

## Global Constraints
No productcode modifications; auxmodeforecast_mse_v1 weight.05 warmup10 seed2022 discarded; original100+100/seed2023/Stage2fixed. Parent pendingv75validation only; no newGPU whilev75 active.

## Review Focus
- Auxfuture/cleanmissing targets neverenterinput.
- Epoch10detach/epoch11nonzero repairgradient.
- Auxdiscarded/formalhead initialization2023 unchanged.
- S1auxidentity and selectionglobalrepairMSE.
- Report allauxtraining resources and nativeauxpadding versus zero_invalidfinal.

### Task1: configure existing mode
- [x] Verified528-source isolation and config copy; zero productchanges.
- [ ] Run existingtests/integration/test_stage1_auxiliary_forecast.py; no newREDclaims.

### Task2: conditional formal run
- [ ] Awaitv75forecastvalidation, lock runtimeconfig/latestparent.
- [ ] Adapt externalcontroller Stage1--auxiliary-forecast, ensureS1identity/6600updates/5940postwarmupbatches; Stage2 unchanged. CUDAreal epoch10/11 lossgradient/fullfreeze check, freeze/archive/review.
- [ ] Formal100+100, then20developmenttests and validationdiagnostics.

### Task3: report
- [ ] Report accuracy/time/GPU/allauxparams, appendcentralledger, validationonlypromotes. Continue ifT1unmet.
