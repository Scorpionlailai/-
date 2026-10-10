# v77 focused final read-only review

## Assessment

Ready for delivery and retention by the fixed-validation rule. Critical0 / Important0 / Minor0; no outstanding findings. v77 improves retained parentv76 validation MSE and MAE, while both T1 goals remain false. This audit does not establish success against T1 or generalized performance.

Review scope follows the completed source and pre-run reviews; no source/full-suite repetition. No GPU use, optimizer steps, training, product/report/reference edits. Only this report is written.

## Independent checks and results

- Frozen inventory: all531 source hashes,3922 protected hashes,6 controller/helper hashes,129 reference-lock hashes and planSHA match. Cached model/checkpoint/evaluation/resource report hashes in report-audit.source_inventory also match. Thus prior baseline/reference artifacts are unchanged; no cached reference was retrained.
- Approval-before-launch chain: implementation-review-approval.saved_utc=2026-10-06T06:27:11.481706+00:00, launcher start=06:27:17.308113+00:00, first Stage1 start=06:27:17.430621+00:00. Approval reviewSHA matches current implementation-review.md, parentE23-v76 and auxiliaryforecast_mse_v1 match plan. Status is complete, exit_code0.
- Training completeness: Stage1 manifest records100 epochs/6600 optimizer updates/seed2023. Both Stage1 and Stage2 logs have exactlyone finalbatch66/66 entry for every epoch1..100, confirming100×66=6600 training batches each. Stage1 auxiliary metadata agrees with plan/payload,5940 active repair-gradient batches and844800 auxiliary windows; auxiliary state is serialized in Stage1 but absent from Stage2 model state.
- Fixed validation selection: scanned all100 Stage2 validation log entries; both minimumMAE and minimumMSE occur at epoch90, matching both manifests/checkpoint payloads. Validation MSE0.9428018874862567 and MAE0.6618061963122448. Both controller locks precede the first test subprocess, and contain no selected-checkpoint test metrics. Original predictor initialSHA is fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561. Primary/secondary entire model-state tensors are bitwise identical; their checkpoint fileSHAs differ due to selection metadata.
- Freeze: verified Stage1 checkpointSHA,218 repair state entries, and torch.equal for every Stage1 tensor against each Stage2 repair tensor. Memory/key and Stage1 checkpoint hashes agree through manifests and payloads; Stage2 trainable repair count is0.
- Mask support: recomputed bankSHA and all20 mask fileSHAs; missing fraction0.8193881149464994. All five model scores share group order, seed, maskSHA,2688 windows and1806336 target elements per draw. Stored per-window files match their fileSHAs; target_start arrays align for each paired comparison.
- Tables: main5 rows,draws100,paired80,selections2, with consistent CSV fields. Independently recomputed all five model means and sample standard deviations(ddof1); confirmed MAE primary/cached-role/checkpoint tags. v77 test MSE0.5511669487639619, MAE0.4930973239496451; stdMSE0.023582361584288725,stdMAE0.010415568089594582.
- Paired differences/wins independently match CSV/Markdown: againstv76 MSE-0.008268313072706302,MAE-0.0052799181062347985,wins20/20 and19/20; againstv72 -0.010399203127473933/-0.005685172005305219,wins20/20 and20/20; againstv69 -0.010449247835985397/-0.008132635709109484,wins20/20 and20/20; againstT1 +0.023807539096494915/+0.017042496036780395,wins0/20 and0/20.
- Resource audit: each model has unique sidecar paths and exactlyone repair plusone forecast stage, including T1 singleJSON containing both stages. Fullworkflow sum, per-stage time, preparation overhead, allocated/reserved maxima and recorded batch/precision agree with CSV. v77 full training14.50861583063379minutes; report rounds14.51. Secondary selection does not add training cost. Cached T1 retains its batch16/BF16 repair versus batch128/FP32 forecast disclosure.
- Promotion: independently evaluated gate using CSV validation scores. v77 improves parentv76 0.9449109349931989/0.6639194142521103, so retained_candidate=E23-v77 and validation_upgrade_vs_parent=True are correct. Both T1 goals evaluateFalse; T1 is0.9293841623665254/0.6575888862955868 on validation and0.5273594096674671/0.4760548279128649 on shared20 development tests. The report keeps validation promotion separate from T1 goals and labels original20 positions as repeatedly exposed development data.

## Report and long-gap hash chain

All four primary CSVs match report-audit.reports hashes. Main Markdown was subsequently augmented by the separately frozen long-gap diagnostic helper: its original prefix hash is db9eddfba5e7d929d0c2d9f9bd0b9764a39d6f0a95a399075518dc0669a2e42e, matching both report-audit and long-gap-diagnostic-audit.report_before_long_gap_sha256. Recovered this prefix from finalMarkdown and recomputed the hash. FinalMarkdownSHA=76d544b85b85bc07e722f8880f2402c17406d307ac9d9c84351cbdd1f73090e8 matches long-gap-diagnostic-audit.report_sha256. The older mainMD hash in report-audit is therefore the documented pre-append hash, not an unexplained report mutation.

Diagnostic rawJSON and CSV hashes match audit; their two rows agree field-for-field. Selected Stage1 checkpoint hashes and fixedvalidation maskSHA are correct. Models are labelledv72/v77, region coordinate and segment counts match, and counts are expressly occurrences across overlapping validation lookbacks. No-new-training/bank-state-unchanged flags and fixed_validation-only interpretation are consistent with the frozen diagnostic source. The table shows worse long-gap reconstruction forv77, and does not claim improved repair as the reason for forecast improvement. T1 was not included in additional repair inference.

## Declined to judge / limits

- Trained GPU inference and optimizer execution were not repeated. Accuracy was independently recomputed from saved20 draw metrics and paired window-support arrays, not raw forecasts; long-gap inference statistics were checked against saved rawJSON/CSV/hashes and frozen diagnostic code, not recomputed by repair inference.
- The previous full-suite author results remain1180passed/9inherited failures/1skip; no claim of an independently green full suite is made here. Frozen source equality means the earlier source review remains applicable.
- Single-seed, repeated development-test exposure, same future target interval and absent cross-dataset evidence remain limitations. T1 goals are unmet; no sealed confirmation was run or approved by this audit.

No deferred findings. No product, report, reference or training file changed during this review.
