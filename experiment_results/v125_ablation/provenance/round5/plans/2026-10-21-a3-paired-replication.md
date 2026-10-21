# A3 Paired Replication Implementation Plan

> For agentic workers: use superpowers:executing-plans inline.

**Goal:** Execute the approved A3 replication and compare all evidence sources.
**Architecture:** Independent round5 artifacts reuse the original sealed workers; add diagnostic sentinels and reports only. Existing isolated ablation worktree/branch is the authorized publication target.
**Tech Stack:** Existing Python3.11/PyTorch2.5.1 CUDA12.4, two4090D GPUs.
**Spec:** spec.md.

## Global Constraints
PAMAP2 0.15/8; seeds2023/2024/2025; original100+100+100 budget/batch128/losses/Adam/selectors/masks, fixed1:1; noStage2 or other experiments. A2 frozen at171f074157264df7f228a5b54de658e99c6af589.

## Review Focus
V leak: forbid sources/messages and poison V parameters. Public variable context: original module must execute and have nonzero gradients. Test barrier: all9checkpoints. Twenty masks: aggregate within trainingseed first. Prior observations: preserve all historical numeric columns and source reports.

### Task1: Source and reuse audit
Files tools/policy5.py,test_policy5.py,isolation5.py,audit5.py. Interfaces require_barrier(checkpoints), paired_training_summary(rows), forbid_variable(model), isolation(model,data,device,path).
- [ ] Write policy tests, verify RED missing implementation, adapt minimal policy, verify GREEN.
- [ ] Audit2023 original sources/checkpoints/library and all100epochs/phase; four realbatch128 isolation+poison checks, frontcrossvariable andH/T positive.
- [ ] Archive A2 published baseline and all source hashes.

### Task2: Fixed training/evaluation/export
Files run5.py,supervisor5.py,export5.py,verify5.py. Interfaces check_lock(),coarse(seed,device),calibration(seed,device),evaluate_seed(seed,device),build(final=False),verify().
- [ ] Use original workers with noV sentinels; paired initialization/order/masks; preserve original models and protocol.
- [ ] Verify96paired keys and earlier-table preservation; publish audited sources and protocol.
- [ ] Launch two GPUs; verify firstepoch actual inputs vsA0; automatically advance original100round phases, lockall, val/test20/strata, sameGPUcapacity, finalverify and upload. Stop on failure; no reruns or expansion.

### Task3: Combined evidence report
Files comparison5.py,test_comparison5.py, combined_source_comparison.csv/.md. Interface build_comparison(summary,per_seed).
- [ ] Tests RED→GREEN: reject missing seeds; preserve allA1/A2/A3/A4; variant−A0 sign.
- [ ] After final A3 evaluation compare four variants coarse/dual, val/test, full96/recent32, bothmetrics, threepaired differences and sampleSD; preserve source provenance; no new evaluations.
- [ ] Final gate verifies fullrowcoverage/baselineunchanged; upload report/commit; stop.
