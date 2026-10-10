# E23 Width 64 Validation-MAE Selection: ETTh1 P4

Date: 2026-09-30. This is the separately recorded v3 follow-up specified in `docs/superpowers/specs/2026-09-30-e23-width64-mae-selection-design.md`. It reuses the selected width 64 Stage 1 checkpoint and changes only the Stage 2 checkpoint-selection rule from lowest fixed-validation forecast MSE to lowest fixed-validation forecast MAE. Stage 2 still trains from a fresh optimizer for 100 epochs with forecast MSE only. Its output directory, checkpoint, profile and sole selected test are separate from the width 64 MSE-selected v2 trial, but both trials use the same fixed test set.

## Accuracy

| Selected model | Test MSE | Test MAE | DRR-MSE | DRR-MAE |
| --- | ---: | ---: | ---: | ---: |
| E23 width 64, validation MAE selected (v3) | **0.427264729** | 0.434448576 | 72.144860% | 53.589746% |
| E23 width 64, validation MSE selected (v2) | 0.429693777 | 0.437349478 | 70.248381% | 48.798494% |
| SAITS selected P4 | 0.440584382 | 0.436957896 | — | — |
| E22 selected P4 | 0.443195461 | **0.433248216** | — | — |

The v3 selected checkpoint beats SAITS on both test forecast metrics: MSE lower by `0.013319652` (`3.02%`) and MAE lower by `0.002509319` (`0.57%`). It also improves over v2 by `0.002429047` MSE and `0.002900902` MAE. E22 retains the lowest MAE, ahead of v3 by `0.001200360`. All tests use the same train-standardized ETTh1 P4 metric space, fixed masks and 2,688 windows. The shared DRR references are the existing PatchTST Clean/Missing selected tests; no SAITS, E22 or reference selected test was rerun.

## Selection and diagnostics

The reused width 64 Stage 1 selected epoch 99. The fresh v3 Stage 2 completed 100 epochs in `158.222` seconds on GPU 0 and selected epoch 87 by validation forecast MAE `0.590439666`; its validation MSE was `0.764769533`. The v2 MSE policy had selected epoch 26. Seeing v2's test MAE deficit motivated the v3 selector change, so this is adaptive reuse of the same test set. Within v3, the epoch was selected by fixed validation before its test was run; v3 test labels did not choose its checkpoint. The selector is paired in the Stage 2 manifest and checkpoint; the evaluation reports `selection_metric=mae`.

The selected test's hidden-history repair MSE/MAE were `1.601452075`/`0.886625451`. On damaged patches, H/V/T availability was `0.880740/1.0/1.0`; mean none/H/V/T route weights were `0.062852/0.333128/0.286965/0.317054`. Mean H/V/T candidate counts were `3.361014/10.205778/4.746283`. The Stage 1 and Stage 2 H memory/key SHA-256 values match, and the selected test reports unchanged H state. This repair diagnostic is not the Stage 2 forecast-selection loss.

### Selected-checkpoint validation ablation

After the selected test, a read-only validation diagnostic masked each expert's availability before the router softmax, without changing weights or H state. Every variant used the same 2,688 fixed validation windows; the full-model row exactly reproduced the selected manifest's validation metrics. No test evaluation was repeated.

| Inference variant | Validation forecast MSE | Validation forecast MAE |
| --- | ---: | ---: |
| Full H/V/T model | **0.764770** | **0.590440** |
| Without H | 0.826515 | 0.613483 |
| Without V | 0.831872 | 0.626356 |
| Without T | 0.792099 | 0.603536 |
| Without all three | 0.935371 | 0.664033 |

Thus all three experts improve this *trained checkpoint's* validation forecast when available: removing H/V/T separately raises MSE by `0.061745`/`0.067103`/`0.027329`. This does not estimate how a model retrained without an expert would perform. The diagnostic is saved at `benchmarks/e23_v3_validation_expert_ablations.json` under the v3 root, SHA-256 `20ace2eb25724549f6da56a6cc3afb0a6b21ef20c5252565bddd7b63cfa06e7f`.

## Fixed-input speed and provenance

On idle RTX 5090 GPU 0, batch `128`, five warmup and ten timed fixed test-input batches, v3 processed `1,280` windows in `0.062385` seconds, or `20,518` windows/s end to end. Model-core time was `0.053700` seconds, cold first batch `0.165` seconds, model load `0.748` seconds, H index construction `0.0044` seconds and peak GPU allocation `217.1` MiB. The same timing boundary gave v2 about `20,616` and SAITS `25,339` windows/s. This profile did not compute test errors; speed is secondary to accuracy. PyTorch was `2.7.1+cu128`, CUDA `12.8`. Stage 1 was reused, so v3 Stage 2 time is not a full two-stage training comparison.

- Source CSV SHA-256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. Fixed validation/test mask SHA-256: `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c` / `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`.
- Width 64 config SHA-256: `f6e11fe8a1256a672e07d97af958fc7091f7cd1ee728f571cc550a9060ef4ebf`; reused Stage 1 manifest SHA-256: `4c3717d818c1d21ad17e29db2c3f16d7205400992d43b37fa4545e180c6ff937`.
- v3 Stage 2 manifest/checkpoint SHA-256: `18ac3d7a2a72cd1ea26866be3da0f602d54084b180c0b45323c24e77aa41a33c` / `1501fa96be1e43361d651d9a5889b3d7e3df147cfc18f918c8ad930aace576d9`. H memory/key SHA-256: `9ca39319c2847c0efb7d58d60a8e4a526e51bb8361197236e121144bc22a4b25` / `1ad11fea2978857761f6f75b3c3135df1744c6c34f082a8ec980301ceeaa84f6`.
- Artifacts under `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_width64_mae_v3/`: `stage2_p4/runs/e23-stage2.json`, `benchmarks/e23_selected_test_profile.json`, `evaluations/e23-width64-mae-test.json`, plus separate background job logs. The selected-test JSON SHA-256 is `5038ca9a8d86eff066008d88dc26460c2cb281cc2c218a24b15f3ba2190ef65f`.

This is one selected test with the locked seed 2021. The margin over SAITS MAE is small, and the selector was proposed after reviewing v2's result on the same test set. The result demonstrates a win in this fixed P4 evaluation, not independent held-out confirmation or measured variance across seeds and datasets.
