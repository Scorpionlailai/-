# E23-v74 Implementation Plan

> For agentic workers: use superpowers:executing-plans inline.

**Goal:** Preserve useful recent-window repair while improving long gaps.
**Architecture:** Same two-layer full96 attention moved before existing tail refinement.
**Tech Stack:** Python/PyTorch/pytest.
**Spec:** /home/xyw/paper/E23/docs/experiments/2026-10-06_E23_v74_pre_tail_attention_design.md

## Global Constraints
Original 35427-parameter predictor and all Stage2 configuration unchanged; seed2023; frozen repair; validation parent v72; all prior files protected; no test-selected weights.

## Review Focus
- Old mode call order unchanged.
- Zero new readout preserves old model and RNG.
- New readout is consumed by tail and mean refinement rather than appended last.
- Hidden NaN/Inf and invalid split gaps never leak.
- Checkpoint identity forbids old/new placement relabeling.

### Task1: placement mode and identity
- [x] Add seven new behavior tests by exercising new configuration with existing six contracts plus real forward hooks that verify earlier readout output reaches point_refiner.
- [x] Run new file RED: missing mode should fail all seven.
- [x] Change config.py, model.py and train.py only; add config YAML. Store mode, select same attention class for both modes, call pre-tail for new mode and late for old modes. Add identity placement only for new mode.
- [x] Run new seven and old six GREEN; full pytest compare against v73 known nine inherited failures and one skip.

### Task2: formal experiment
- [x] Four inherited/adapted controller checks GREEN. CUDA two-step finite/gradient and full Stage2 freeze preflight.
- [ ] Freeze files, archive, request independent implementation review.
- [ ] Run 100+100, validation locks then old20 tests and fixed-validation repair diagnostics.

### Task3: record and continue
- [ ] Verify scores/resources/243 repair tensor freeze and publish main4/draw80/paired60/selections2 CSV.
- [ ] Append long-gap diagnostic and central trial ledger.
- [ ] Review delivery; promote based only on fixed validation. Continue if T1 target not reached.
