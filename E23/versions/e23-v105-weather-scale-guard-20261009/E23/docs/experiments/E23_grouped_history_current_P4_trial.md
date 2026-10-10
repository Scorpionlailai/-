# E23 historical/current grouped experts: ETTh1 P4 validation trial

Date: 2026-09-30. This trial implements the user's two dimensions of recovery evidence: historical H plus a new historical-variable expert, and current-window T plus the existing current-variable expert. The historical-variable expert reads aligned training-only source-variable patches at valid H prototype starts. A two-level router fuses experts within each group and then chooses `none`, `historical_group` or `current_group`. The forecast mixer is disabled. The previous flat v3 checkpoint remains unchanged.

## Locked protocol and run

The ETTh1 P4 source, fixed E22 validation/test masks, 96/96 window, 16/8 patches, seven variables, train-only standardization, seed 2021 and batch 128 match v3. The new repair width is 64. Stage 1 ran all 100 epochs with hidden-coordinate SmoothL1 and the paired retrieval objective; earliest fixed-validation repair-MSE minimum selected epoch 50 (`105.376455652`, versus flat width-64 Stage 1 `107.139995715`). Stage 2 ran all 100 epochs with a fresh Adam optimizer, forecast MSE loss and earliest fixed-validation MAE selection; epoch 24 was selected. Stage 1 and Stage 2 training took 260.872 and 181.580 seconds on RTX 5090. An earlier exploratory Stage 1 run was interrupted after about 20 epochs during a scale check; the complete run used a fresh output directory and is the only selected Stage 1 for this trial.

The selected Stage 1 manifest is `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_grouped_history_current_width64_v5/stage1_p4_full/runs/e23-stage1.json` (SHA-256 `a5264bee1e22a5c4289d0b1efe29b0bb9c0fe4d7c3501ff501b3203a951d0516`), checkpoint SHA-256 `c058a29256fc7d8a1f5dba1ea643225e522bfa6d86e984e05c15122cf842c842`. The selected Stage 2 manifest is `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_grouped_history_current_width64_v5/stage2_p4/runs/e23-stage2.json` (SHA-256 `f3b0a2fa7335e0226df8c5b56927050f1bb39d2d3ab0cc5379b61d65cf4d60bd`), checkpoint SHA-256 `d16dd62336cd7dbe34248a1a1750e44a5441e12627469452ce39bec5cd0bf304`. The paired historical archive SHA-256 is `ae5ee3c28cf6884185c4c90a8273c63e007923db312382848d3a2a957dcc5811`; H bank/key hashes match both stages. Code revision at training was `045a2f7`.

## Predeclared validation decision

| Selected model | Validation MSE | Validation MAE |
|---|---:|---:|
| Flat E23 width 64 v3 | 0.764769533 | 0.590439666 |
| Grouped historical/current v5 | 0.790756002 | 0.600258510 |

The grouped variant is higher by `0.025986469` MSE and `0.009818845` MAE. It fails both predeclared requirements (MAE below v3, MSE no higher than v3), so **no v5 ETTh1 test evaluation or test-input speed profile was run**. The current selected ETTh1 test leader remains flat E23 v3 (`0.427264729` MSE, `0.434448576` MAE) versus SAITS (`0.440584382` MSE, `0.436957896` MAE); those are earlier test results, not v5 results.

## Validation-only evidence analysis

The read-only diagnostic script and output are under `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_grouped_history_current_width64_v5/benchmarks/`; the JSON SHA-256 is `f7f2d1bd983e1ac2495fbe7d17d6d1cedba0485edc7cbe2a253493eb73947a0d`. It rebuilt the selected model and paired H state, processed all 2,688 fixed-validation windows, reproduced the selected full metrics exactly, and confirmed that H state was unchanged.

| Inference condition on the selected checkpoint | Validation MSE | Validation MAE |
|---|---:|---:|
| Full grouped model | 0.790756 | 0.600259 |
| Historical group suppressed | 0.859846 | 0.628320 |
| Current group suppressed | 0.887177 | 0.643755 |
| Historical-variable leaf suppressed | 0.791896 | 0.608561 |

Across damaged validation patch targets, the historical group was available for 88.06% and had mean top-level weight 49.40%; the current group was available for 100% and had mean weight 32.30%; `none` had mean weight 18.30%. Effective leaf weights were H 30.81%, historical V 18.59%, T 14.53%, current V 17.77%. Both groups and the new historical V affect predictions: suppressing each raises error in this trained model. These inference ablations change the routing distribution and do not establish that the architecture would outperform a retrained alternative. Suppressing T or current V alone creates a particularly large routing shift, so those single-leaf numbers should not be read as a clean ranking of expert quality.

## Interpretation and verification

The two evidence sources are now genuinely separate and both are used, but this hierarchical routing trial did not improve forecast accuracy. Stage 1 repair selection improved while Stage 2 forecast selection worsened, so better hidden-coordinate repair MSE alone did not predict better forecasting. This trial changes expert count, router structure and random initialization together; it does not isolate which change caused the forecast regression. A controlled next variant would retain the v3 flat routing path and add the historical-variable expert with a zero-initialized residual gate, then use the same predeclared validation gate before any selected test.

All 62 E23-specific tests passed. The full repository suite passed 664 tests with one existing skip when `git` was on `PATH` and the old tests' expected local mask path was temporarily symlinked to the cataloged E22 artifact; that symlink was removed afterward. No SAITS/E22 model, mask or test artifact was changed. ETTh1 test had already informed architecture choices in this project; any later test improvement needs confirmation on independent held-out data before a generalization claim.

## Rollback after review

After seeing the failed validation gate, the user requested a return to the previous design. The grouped implementation, config and tests were reverted from the active code branch; the active E23 repair path is again the flat H/V/T v3 structure. This report and ignored v5 artifacts remain as the record of the completed negative trial. The optional forecast mixer from the earlier v4 trial remains disabled by default, as it was before this experiment.
