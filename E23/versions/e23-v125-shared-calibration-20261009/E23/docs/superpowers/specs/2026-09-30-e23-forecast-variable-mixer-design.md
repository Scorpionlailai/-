# E23 Forecast-Side Variable Mixer: ETTh1 P4

Date: 2026-09-30. The user approved trying the patch-local cross-variable mixer described in chat. This is a new Stage 2 architecture trial after the selected width 64 validation-MAE E23 result. The prior v2/v3 checkpoints and tests remain immutable.

## Objective and evidence

The width 64 v3 selected test has MSE `0.4272647293` and MAE `0.4344485763`, beating SAITS on both metrics, but its MAE margin is only `0.0025093193`; E22 MAE remains lower by `0.0012003602`. Validation-only inference ablations of the v3 checkpoint raise both forecast errors when any H/V/T expert is disabled. The repair model has width 64 and 113,241 parameters; the downstream PatchTST predictor has width 16 and 35,427 parameters and processes variables independently after repair. This suggests testing a small forecast-side variable interaction while keeping the three useful repair experts unchanged. It is a hypothesis, not a guaranteed gain.

## Architecture

Reuse the selected width 64 Stage 1 manifest and its training-only H memory/frozen key. Keep P4 masks, source, normalization, 96/96 length, 16/8 patches, seven variables, seed 2021, batch 128, all H/V/T settings, Stage 2 fresh Adam optimizer, LR 0.0001, 100 epochs and forecast-MSE-only training unchanged. The only added operation is after the forecast patch projection and missingness projection and before the existing channel-independent PatchTST temporal encoder.

The forecast latent shape is `[B,C,N,D]` with `C=7`, `N=12`, `D=16`. For each patch index, reshape to `[B*N,C,D]`, apply one pre-normalized four-head self-attention layer across the seven variable tokens, reshape back, and add it through one trainable scalar residual gate initialized to zero. Attention dropout is zero. The zero gate makes the initial forecast path equal to the old path and lets training learn the strength of variable exchange. No future target or test label enters the mixer. Existing mask-derived latents remain its input, so the prediction path retains missingness information.

The implementation is optional (`forecast_mixer="none"` by default; `"patch_attention_v1"` for this trial) so old checkpoints and selected evaluations retain their exact model structure. A new Stage 2 manifest/checkpoint field records the mixer identity; old records lacking it mean `none`. Pair verification rejects a mismatch before profiling or evaluation. The CLI exposes only the reviewed mixer choice. The Stage 1 E23 config hash stays the same, avoiding a false Stage 1 incompatibility when only the predictor changes.

## Evaluation rule

Train the new Stage 2 for all 100 epochs in a fresh `e23_multi_expert_width64_variable_mixer_v4` artifact root. Select the earliest epoch minimizing fixed-validation forecast MAE, matching v3. Save both selected validation MSE and MAE. Before launching training, set the go/no-go gate for an ETTh1 selected test: selected validation MAE must be below v3 `0.5904396656148828` and selected validation MSE must be no higher than v3 `0.7647695325129157`. If the gate fails, report the validation result and do not run ETTh1 test. If it passes, profile fixed test inputs and run exactly one selected test, comparing SAITS/E22 and v3 on the same locked P4 protocol. No test result changes checkpoint selection or the gate.

The prior ETTh1 test has already informed this architecture choice, so even a v4 test win is exploratory reuse of that set. Independent held-out evaluation remains necessary for a paper-level generalization claim. No retraining, test, or configuration change is made to SAITS or E22.
