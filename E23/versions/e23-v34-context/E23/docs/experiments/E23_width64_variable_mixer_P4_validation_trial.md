# E23 Width 64 Forecast Variable Mixer: ETTh1 P4 Validation Trial

Date: 2026-09-30. This is the patch-local cross-variable attention trial specified before training in `docs/superpowers/specs/2026-09-30-e23-forecast-variable-mixer-design.md`. The new Stage 2 reused the selected width 64 Stage 1 H/V/T checkpoint and paired H state. It added one zero-gated, four-head attention layer across seven variable tokens at each forecast patch; the PatchTST time encoder remained width 16. Stage 2 retained the fresh Adam optimizer, forecast-MSE-only loss, seed 2021, fixed P4 protocol and 100 epochs. Selection was by the earliest fixed-validation forecast MAE minimum.

## Predeclared gate and outcome

An ETTh1 selected test was permitted only if the new selected validation MAE was below v3 `0.5904396656148828` **and** validation MSE was no higher than v3 `0.7647695325129157`. The new run completed 100 epochs and selected epoch 87. It failed both parts of this gate, so **no v4 test evaluation or test-input speed profile was run**. The v3 selected test remains the E23 accuracy result against SAITS.

| Selected Stage 2 | Validation forecast MSE | Validation forecast MAE |
| --- | ---: | ---: |
| Width 64 v3, no mixer | **0.764769533** | **0.590439666** |
| Width 64 v4, patch variable mixer | 0.773467380 | 0.591542694 |

Relative to v3, the selected v4 checkpoint was worse by `0.008697847` MSE and `0.001103028` MAE. Its Stage 2 training took `159.117` seconds on GPU 0. The lowest validation MSE anywhere in v4's 100-epoch log was epoch 24 (`0.75909480`, MAE `0.59200506`), but it was not the MAE-selected checkpoint. No logged epoch met both predeclared validation thresholds; no epoch was reselected after reviewing the curve.

## Mixer-use diagnostic

At the selected epoch, the attention residual scale had learned `0.239347175` from its zero initialization. In a read-only validation inference pass over the same 2,688 windows, setting just that gate to zero changed MSE/MAE from `0.773467380`/`0.591542694` to `0.768358516`/`0.591918426`. Thus the learned attention path slightly improved MAE within this checkpoint but worsened MSE; the full v4 model still missed the v3 validation gate. This inference ablation is not a retraining comparison. The H state was unchanged.

## Provenance

- Stage 2 manifest: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_width64_variable_mixer_v4/stage2_p4/runs/e23-stage2.json`, SHA-256 `eb65a21d33277e9bcf6607eb8033010480fb9cccb3ad1d2513e17cd24b82fd5d`. Selected checkpoint SHA-256 `29423bc4efe55b36c5a893b70aad1d9d07a4b5b5554d8ea6624fb63884033b0d`.
- Read-only validation ablation: adjacent `benchmarks/e23_selected_validation_mixer_ablation.json`, SHA-256 `f9eb8f043297d1012f19f49b1483e9277fd6dbfb71bcae805b837ef6631e0077`.
- Stage 1 manifest SHA-256 `4c3717d818c1d21ad17e29db2c3f16d7205400992d43b37fa4545e180c6ff937`; H memory/key SHA-256 `9ca39319c2847c0efb7d58d60a8e4a526e51bb8361197236e121144bc22a4b25`/`1ad11fea2978857761f6f75b3c3135df1744c6c34f082a8ec980301ceeaa84f6`.
- Source CSV SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; fixed validation mask SHA-256 `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c`; E23 width 64 config SHA-256 `f6e11fe8a1256a672e07d97af958fc7091f7cd1ee728f571cc550a9060ef4ebf`. The model-specific mixer identity `patch_attention_v1` is paired in the Stage 2 manifest/checkpoint and defaults to `none` for old records.

This is a completed negative validation trial of one specific mixer design. It does not show that every forecast-side cross-variable mechanism would fail. No ETTh1 test result was produced for v4.
