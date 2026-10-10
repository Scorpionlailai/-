# E23 v12 window-conditioned V: ETTh1 P4 validation trial

Date: 2026-09-30. The predeclared v12 trial adds window/patch-conditioned sparse source choice to V while retaining four source slots, the flat H/V/T router, explicit repair, and the width-16 PatchTST predictor. Both stages were trained fresh for 100 epochs at repair width 64 with seed 2021 and batch size 128. Stage 2 used the v11 winning `MSE + 0.2 MAE` objective and validation-MAE checkpoint selection. No test set was evaluated for this trial.

| Selection | Validation MSE | Validation MAE | Decision |
|---|---:|---:|---|
| Original v3 | 0.7647695325 | 0.5904396656 | Starting baseline |
| v11 mixed Stage 2 loss | 0.7636515527 | 0.5890489782 | Prior winner |
| **v12 window-conditioned V** | **0.7454477286** | **0.5882120695** | **Promoted: lower MAE and MSE within guard** |

Stage 1 selected epoch 51 with fixed-validation hidden-coordinate repair MSE 106.9069637043; Stage 2 selected epoch 53. The selected Stage 1 manifest/checkpoint SHA-256 values are `029aaab1521c9ddc81cf14a0c4c3ae53687283c2558ccdb7cb70a41f1ae82b84` / `c8c8ea04a5c60e03719bfd1a4a35c66fd38c486877192066cf8e03c147c88dff`. The Stage 2 checkpoint SHA-256 is `c3ae09195b793b144936f38ceba0909f377d7fb02cfea626689fb21e70af3c35`; its manifest is `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v3_window_v12/stage2/runs/e23-stage2.json`. Both manifests pair the `window_context_topk_v1` identity. The published H memory/key SHA-256 values `fe09f8c6742cd1477cb1dc44fed3e5bec88ab700c557cb516ea4949d43de2b28` / `db6b8c905b6b4074160815a370973a2d76ed3bff5e30fa61d22dabafcd2ca396` are unchanged across Stage 2.

Read-only replay of all 2,688 fixed-validation windows reproduced selected MSE/MAE exactly and confirmed H state immutability. V selected source counts across all target patches and four slots were `[135670, 181963, 110068, 123735, 129027, 156685, 66020]` for variables 0–6 (903,168 slots total). At least one of the four ordered sources changed between adjacent target patches in 35.54% of comparisons; damaged targets had 10.41 valid V source/offset candidates on average. These are descriptive checks of the mechanism, not causal evidence that each variable contributes equally. Replay output is in the ignored `benchmarks/v12_validation_replay.json` under the trial root. The one-batch P4 CUDA backward was finite; peak allocated memory was 0.478 GiB for repair and 0.514 GiB for forecasting.

This validation result makes v12 the input to the planned H reranking trial. Its forecast gain is still exploratory because the same ETTh1 validation set informed the design sequence; an independent dataset and seeds are needed for a paper-level generalization claim.
