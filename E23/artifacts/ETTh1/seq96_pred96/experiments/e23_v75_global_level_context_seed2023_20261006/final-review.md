# E23-v75 final result and record review

Scope: read-only review of completed v75 training, checkpoint selections, development evaluation, publication, long-gap diagnostic and autonomous-series record. No training, optimizer steps, GPU execution or scoring was performed. Only this requested review artifact was written.

## Findings

- Critical: none found.
- Important: none found.
- Minor: no new issue found. Both implementation-review metadata findings were resolved before formal outputs: cell plans take their description from the immutable plan, and the publisher promotion caption uses v75. The prior freeze definition and explicit revision chain remain available.

## Result and selection

Stage1 and Stage2 each completed 100 epochs. Stage1 records 6600 optimizer updates and selected epoch 76. Both Stage2 MAE and MSE selections selected epoch 90. Their forecast validation MSE/MAE are 0.9701405058101732 / 0.6702500351972861; their old-20 development MSE/MAE are 0.5915896540055741 / 0.5126202232443016. Both fixed-validation metrics worsened relative to retained v72. Retaining v72 is consistent across the report audit, report text and series ledger; neither validation nor development goal versus T1 was reached.

Both selection locks identify the scored checkpoint, match validation replay within 1e-6, and precede their test-command files. The controller log also records both validation locks before testing. MAE/MSE results are disclosed as two selections from one training run, not independent runs.

## Independently checked evidence

1. Tables contain exactly 4 main rows, 80 model-by-mask draws, 60 paired comparisons and 2 checkpoint selections. All four means and sample standard deviations were independently recomputed from draws. All 60 differences were recomputed and paired mask/group/window/target-element identities match. All 80 referenced per-window artifact hashes match.
2. Loaded selected checkpoints on CPU. Every one of the 215 Stage1 repair state tensors is bitwise identical in both Stage2 selections. Historical memory/key identities also match. No repair tensor changed during Stage2.
3. Every frozen hash matched: 528 source files, 3714 protected files, three controllers and the plan. All 528 files inside the frozen source archive also match. The retained pre-review freeze hash matches the explicit metadata revision, whose source map and plan hash remain unchanged. All 19 report input hashes match.
4. All four CSV report hashes match. The original report hash equals the diagnostic audit's pre-append hash; removing the appended diagnostic section reconstructs that original hash exactly. The current Markdown hash equals the diagnostic audit's final hash. Diagnostic raw JSON and CSV hashes match, and both diagnostic CSV rows exactly match raw JSON.
5. Long-gap diagnostics use the same fixed validation mask and coordinate counts: 1,387,503 occurrences in runs of at least 16 points for each model. Reported gap-at-least-16 MSE is 0.5036254354245573 for v72 and 0.5782144982852326 for v75. Reviewed diagnostic source restricts inference to validation, uses lookback reconstruction targets, preserves visible coordinates and verifies historical state. It does not consume forecast targets.
6. Recomputed complete resource costs directly from stage sidecars, including preparation/reporting and both training stages, with peak allocator memory as the stage maximum. v75: 842.5158342635259 seconds = 14.041930571058765 minutes; allocated/reserved 4.588778495788574 / 8.681640625 GiB. Reference totals also match: v72 14.00439596218057 minutes, v69 13.940992418552439 minutes, T1 18.80192616667288 minutes. The report discloses T1 batch/precision/update-policy differences, and does not treat the secondary selection as additional training cost.
7. The independent implementation review hash matches its approval record. Approval at 2026-10-06T02:25:17.417329+00:00 precedes formal launch at 02:25:27.384362+00:00. The reviewed implementation had no Critical/Important findings, and the two metadata resolutions plus four controller checks were recorded before launch. This is the recorded review-based execution approval, not a claim of separate human approval.
8. The v75 series result dictionary exactly matches the main CSV row; it records v72 retained, both T1 goals false, and v75 complete. The reserved confirmation manifest hash still matches and the ledger records it as unscored.

## Limits and disposition

No independent training/inference rerun was performed. The nine inherited full-suite failures remain disclosed; this review does not convert them into passing tests. Existing 20-mask results are repeatedly exposed development data, and this single training seed cannot establish general superiority, causal information usefulness, cross-dataset performance or controlled speed advantage. Hashes and recorded ordering provide local reproducibility evidence rather than external timestamp attestation.

Ready to archive as a completed negative experiment with v72 retained. No outstanding result/record correction was identified. Preserve all v75 frozen files and results; subsequent auxiliary Stage1 work belongs to a new recorded trial with the final head unchanged. This review is not approval of that subsequent implementation or its training launch.
