# E23-v75 Implementation Plan

> For agentic workers: use superpowers:executing-plans inline.

**Goal:** Make observed global source levels available to recovery experts without changing final predictor.
**Architecture:** Zero-start two-channel parallel input projection of existing window CNN; preserve 5-feature point evidence API.
**Tech Stack:** Python/PyTorch/pytest.
**Spec:** /home/xyw/paper/E23/docs/experiments/2026-10-06_E23_v75_global_level_context_design.md

## Global Constraints
Final35427 head/Stage2 unchanged; original none behavior/RNG; only true visible/valid values; no new GPU training until v74 complete; validation only promotion.

## Review Focus
- CNN combined7 channels versus downstream5 interface.
- Whole-feature RNG and original5-channel operator/parameters exact.
- Invalid NaN/Inf and entirely unseen variable statistics excluded.
- Global information can reach another variable via signature.
- Checkpoint identity prevents relabelled input policy.

### Task1: global level path
- [x] New7tests RED on missing policy: config; zero-start/head/RNG; exact visible-only channels and returned5; cross-variable level sensitivity after activation; invalid/hidden isolation; learning gradients; full frozen checkpoint identity.
- [x] Modify config.py/model.py/train.py/window_context.py; add front_level_context defaultnone, new mode constraints, zero2-channel input projection and input-only feature construction.
- [x] New7 GREEN and inherited relevant checks; full pytest compare known9 failures.

### Task2: conditional formal experiment
- [x] Await v74, lock latest validation parent and references without using test for parent.
- [ ] Adapt controller/preflight/report to runtimev72+128 strategy, exactparams421101; CUDApreflight after GPU is free, freeze/archive/review.
- [ ] Formal100+100, validation locks,20tests, fixed validation diagnostics.

### Task3: results
- [ ] Publish all scores/resources/frozen state and update central ledger; only validation promotes.
- [ ] If T1 goal still unmet continue to next evidence-based change.
