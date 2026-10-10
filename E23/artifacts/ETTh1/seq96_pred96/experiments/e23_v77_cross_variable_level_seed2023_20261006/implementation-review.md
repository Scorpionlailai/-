# v77 focused final pre-run implementation review

Supplement to `source-review.md`; requesting-code-review procedure. No GPU, training, source edits, or repetition of the completed source/full-suite review. Only this review artifact is written.

## Assessment

Formal readiness: YES. Critical0 / Important0 / outstanding Minor0. The one report-freeze Minor below has been resolved. Fixed recipe/parent, preflight evidence, execution isolation, training locks and intended report logic are consistent. This is approval of run preparation, not a claim that v77 achieves validation or test goals.

## Verified preparation and conditioning

- Updated plan acceptance binds fixed-validation MSE lower and MAE no worse to retained E23-v76. Publisher lines145-157 obtain parent from `plan['validation_parent']`; acceptance uses validation values only. Existing v76 report-audit retains v76 by validation upgrade. v76 validation0.9449109349931989/0.6639194142521103 improves v72 0.9455473385159932/0.6641783692669165. Test metrics do not determine inherited auxiliary choice or promotion.
- Comparing actual v76/v77 YAML documents yields exactly one difference: `model.recent_mean_projection`: mean_v1 -> global_level_mean_v2. Stage1 auxiliary identities match v76 Stage1 manifest exactly: forecast_mse_v1, weight.05, warmup10, seed2022, discardedTrue. The command supplies this auxiliary mode; controller verifies its manifest metadata,5940 active repair-gradient batches and844800 total auxiliary training windows. Stage1 remains100 epochs/6600 updates, Stage2 is one100-epoch trajectory with primaryMAE and secondaryMSE checkpoints.
- Independent recomputation:531 source hashes,3921 protected hashes,129 reference hashes and all recorded controller hashes pass. Source archive has exactly531 entries whose contents match source hashes. The plan metadata correction retains sources, controller and protected hashes; before/after planSHA are recorded in `metadata_revision`. Current planSHA matches definition.
- Independently reran controller CPU suite:5 passed in0.04s. Verified fixed seed, auxiliary mode, full repair freeze, original forecast loss/padding, dual checkpoint flag, pairing rejection and v77 identity checks. Launcher/controller/publisher/diagnostic scripts parse with AST.

## Preflight evidence inspected, not rerun

`preflight.json` and matching frozen `preflight.py` record a real historical-memory CUDA path without test scoring. Global branch first-layer gradient is0 then0.2633994221687317 after second step; original position second gradient0.08161188662052155. Auxiliary warmup disconnects repair at epoch10 and enables hidden repair gradient at11. Formal head remains SHA fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561. Stage2 repair parameters all require_gradFalse, repair eval/predictor train modes correct, and repair hash remains identical after a forecast optimizer step.

Counts are consistent: inference repair424238, original formal head35427; Stage1 auxiliary total35427/trainable35424 owing to3 fixed attention scalars; Stage1 total459665/trainable459662. Publisher reports training and inference counts separately, auxiliary native padding, formal zero_invalid padding, and discards auxiliary from Stage2 inference state.

## Execution and report contract

- Commands run in isolated WORK with scenario manifest injected and PYTHONPATH=src:.; verify frozen content before/after each subprocess. Partial steps cannot silently restart. Stage1 run is new and passes its100-epoch/6600-update/identity assertions before new Stage2. Both primary and secondary checkpoint locks are written before either test command. Evaluator provides further validation replay, strict load/identity and repair-state checks inherited from the reviewed source.
- Publisher uses5 paths: newv77 and cachedv76/v72/v69/T1, giving5 main rows/100 draw rows. Four comparison names give80 paired rows; two validation selections give2 selection rows. Checked cached reference CSV availability, MAE role/checkpoint identities and numeric repair scores. Pairing checks bank/validation SHA, split/space, seeds/order/count; publisher compares future target_start arrays. Means and ddof1 standard deviations are recomputed from20 draws.
- Resource aggregation rejects duplicate record paths and requires exactlyone repair andone forecast stage; sums both workflows and overhead, verifies allocator peaks. T1 batch16/BF16 and forecast batch128/FP32, cached resource statuses, seed differences and joint policy remain disclosed. Secondary selection carries no extra training cost.
- Promotion statement uses parentv76 validation only. T1 fixed-validation and development-test goals are separate. The20 repeatedly exposed positions are labelled development, not blind confirmation; the separate reserved20 mask manifest and confirmation lock are within3921 protected files. No confirmation evaluation is launched by these helpers. Diagnostic labels name v72 plain parent and v77 correctly.

## Finding

### Minor1 — resolved: complete the hash boundary for report replay

`preflight.py` constructs `frozen-definition.controller` from execute.py,preflight.py,test_controller.py only. ROOT publish.py,launch.py,reporting/diagnose_long_gaps.py are not currently hashed there. Additionally publish.py lines75-76 reads ETTh1_E23_v69_T1_other_missing_seed2023_20261005.csv; this main CSV is outside both protected and reference-lock lists (the all_scenarios CSV is locked). Current content is correct and publisher crosschecks cached scores/checkpointSHA, so this does not block formal training. Adding these actual helper/input hashes to the frozen definition or a separately locked inventory would ensure the reviewed report logic/metadata is the one later used. This requires only a recorded freeze-metadata revision; no CUDA rerun or source changes.

## Declined to judge / limits

- Actual trained checkpoint locks,100-epoch outputs, achieved validation/test goals and resulting numeric report do not exist yet. Final artifacts require a post-run audit.
- CUDA preflight and previous full-suite results were inspected as author evidence, not rerun under the no-GPU/no-training scope.
- Sealed confirmation, multi-seed robustness and cross-dataset generalization are not claims approved here.

No Critical or Important source/helper finding. The prior source review remains valid.

## Reporting-lock resolution verified

Independently verified updated frozen definition:531 source entries unchanged,3922 protected files and6 controller/helper entries, all current hashes match. Exactly publish.py,launch.py,reporting/diagnose_long_gaps.py and the actual v69 main CSV were added to the hash boundary; all prior hashes and planSHA remain unchanged. Previous definition is preserved as frozen-definition-pre-reporting-locks.json, with reporting_lock_revision recording the additions. Minor1 is resolved; no outstanding findings. This metadata-only correction required no source change or CUDA rerun.
