# E23 v11: MAE-Aligned Stage 2 Loss on ETTh1 P4

Date: 2026-09-30. This is the first of the user-selected v3 improvements (directions 1, 3 and 4). It reused the selected width-64 v3 Stage 1 checkpoint and flat H/V/T repair architecture, frozen H bank/key and PatchTST predictor width 16. A fresh Stage 2 optimizer trained for all 100 epochs with train-standardized forecast loss `MSE + 0.2 × MAE`; the fixed coefficient was chosen before the run. The earliest minimum fixed-validation forecast MAE selected epoch 87. The selected checkpoint was replayed over all 2,688 validation windows and reproduced its manifest metrics exactly.

| Model | Validation MSE ↓ | Validation MAE ↓ | Stage 2 loss |
| --- | ---: | ---: | --- |
| E23 v3 | 0.7647695325129157 | 0.5904396656148828 | MSE |
| **v11** | **0.7636515526544481** | **0.5890489781524589** | MSE + 0.2 MAE |

V11 lowers validation MAE by `0.0013906874624239` and MSE by `0.0011179798584676`, passing the predeclared gate. It becomes the *validation* incumbent for subsequent V and H trials. No v11 test or test-input speed profile was run; at most one final winner will be tested after all three directions are evaluated.

The selected Stage 2 manifest is `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v3_mixed_loss_v11/stage2_p4/runs/e23-stage2.json`, with selected checkpoint SHA-256 `d8bc9fef8e0f61c9729e476a40d1b9e88d1cfaf027d7d15fe1256524da7a0931`. The paired Stage 1 manifest SHA-256 is `4c3717d818c1d21ad17e29db2c3f16d7205400992d43b37fa4545e180c6ff937`; H memory/key SHA-256 remain `9ca39319c2847c0efb7d58d60a8e4a526e51bb8361197236e121144bc22a4b25` / `1ad11fea2978857761f6f75b3c3135df1744c6c34f082a8ec980301ceeaa84f6`. The `forecast_mse_plus_0p2_mae` identity is paired in manifest and checkpoint. Source/mask/config identity is unchanged from v3. Training took 159.17 seconds on an idle RTX 5090. The validation-only v3 error map is stored under the v11 `benchmarks/` root; it shows rising forecast MAE across four 24-step horizon bands (0.491, 0.575, 0.629, 0.667), which motivates later conditional evidence analysis but does not itself identify a causal mechanism.

This is one validation-selected run at seed 2021. The improvement is small and the ETTh1 test has informed earlier E23 decisions, so the final selected test and independent datasets/seeds remain necessary before a paper-level generalization claim.
