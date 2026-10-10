# E23-v78 formal pre-run implementation review

Scope: focused extension of source-review.md after fixed-validation parent selection, formal configuration, CUDA preflight, helper adaptation and freeze. This review performed read-only inspection and hashing, with no GPU, training, optimizer steps or additional broad tests. Only this review file was written.

## Findings

Critical: none found.

Important: none found.

Minor — operational approval ordering is externally enforced. `launch.py:9` opens the run log and starts the controller without checking implementation-review-approval.json; `execute.py:86-97` checks passed preflight and frozen hashes, not the independent review record. Thus direct invocation could bypass the intended approval-before-launch order. The coordinator has explicitly committed to persisting the review approval, verifying all frozen paths, and launching in a separate subsequent tool call. That recorded orchestration is an adequate mitigation for this bounded run; modifying the already frozen launcher is not required for this review. Do not describe the launcher itself as a fail-closed approval gate.

## Parent and configuration

- The actual v77 MAE-selected manifest records fixed-validation MSE 0.9428018874862567 and MAE 0.6618061963122448. Its report audit records successful validation promotion and E23-v77 retained.
- Compared v78 YAML directly with the selected v77 cell configuration: the sole difference is historical_key_policy, from legacy_v1 to global_aligned_v2. The inherited global_level_mean_v2 path, original predictor settings, remaining repair recipe and training settings match.
- Plan binds v77 as the nonprovisional validation parent, repair parameters 424238 and 218 repair state entries. Auxiliary forecast_mse_v1/.05/warmup10/seed2022/discarded is explicitly retained, with 459665 total Stage1 parameters and 459662 trainable. The separate final predictor remains 35427 parameters with its original initialization SHA. No new historical parameters or completely hidden-patch eligibility is introduced.

## Preflight and frozen evidence

The recorded actual CUDA preflight is passed and includes captured global inference-query inputs, finite key/repair gradients, positive second-step inherited position/global-mean gradients, retained auxiliary gradients, original final predictor SHA, correct parameter counts, fully frozen Stage2 repair and no test scoring. Reviewed its source assertions and saved result without rerunning it. Existing preliminary source review independently ran six permitted non-training CPU behaviors; training-only EMA/checkpoint behavior remains supported by the recorded seven-test run and inspected code. The retained suite reports 1187 passed, nine disclosed inherited failures, one skip; five controller checks are recorded, and their cases were inspected.

Independent current hash checks all passed:

- 533 frozen source files, 4026 protected files and six controller/reporting helpers.
- Frozen plan and all 129 reference locks.
- Every one of the 533 source archive members.
- Source-review.md SHA: 50448d4902f7dc474335fe2f656644a7c4fc214b1a3270f3bc070e5428b90970.

At inspection time no console.log, status.json or implementation-review-approval.json existed. This is consistent with the stated pre-run state; the approval record is the next required orchestration artifact.

## Execution and reporting contracts

- Training commands preserve seed2023, inherited auxiliary selection, original 100+100 settings, batch128, Stage2 frozen repair, MSE+.2MAE/uniform, learning rate and zero-invalid padding. Stage1 completion checks include 6600 updates, auxiliary identity/discard policy, parent mean policy and aligned key/epoch EMA identity.
- Stage2 locks check configuration hash, seed, full epochs, zero trainable repair parameters, original final predictor SHA, validation-mask identity and historical policy/coordinate/loss/EMA metadata. Both MAE and MSE selections are locked before the old 20-mask development tests.
- Publisher loads v78/v77/v76/v72/v69/T1: six main rows and 120 draws. Five declared reference comparisons yield 100 paired rows; two same-trajectory selections remain separate. It selects the parent dynamically by plan.validation_parent, applies validation-only promotion, verifies paired mask and target-start identities, recomputes means/std, and accounts for complete measured stage resources without double counting selection rows.
- T1 historical policy/coordinate/EMA fields are explicitly not_applicable. Publisher checks auxiliary exclusion from final model state and equality of all 218 repair tensors across Stage1 and both Stage2 checkpoints.
- Long-gap diagnostic helper remains an explicitly labelled v72-versus-v78 historical comparison. It does not compare the immediate parent v77 and cannot isolate this iteration's H-coordinate change; the main forecast promotion/report comparison correctly uses v77. Treat that diagnostic according to its stated scope.

## Readiness and limits

Ready for the one bounded formal v78 run once the coordinator persists the independent review approval and verifies the frozen inventory before a separate launch call. There is no model or selection blocker. No formal result, performance benefit, convergence guarantee or T1 superiority is claimed. Future labels remain excluded specifically from retrieval-key loss/forward/bank; the declared separate auxiliary Stage1 forecast supervision still uses training future targets. Old-20 results remain development data and the reserved confirmation bank remains outside this run's scoring scope.
