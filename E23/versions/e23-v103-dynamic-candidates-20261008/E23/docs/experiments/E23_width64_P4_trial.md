# E23 Width 64 Accuracy Trial: ETTh1 P4

Date: 2026-09-30. This is the independent width 64 variant specified in `docs/superpowers/specs/2026-09-30-e23-width64-accuracy-design.md`. The only architecture setting changed from E23 width 16 is the repair latent/key/message width `16→64`; the predictor remains the same width 16 PatchTST. Both stages ran 100 epochs with the same locked P4 source, masks, training objective, optimizer rates, seed and validation selection. One selected test was executed after Stage 2 selection.

| Selected model | Validation forecast MSE | Test MSE | Test MAE | DRR-MSE | DRR-MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| E23 width 64 | 0.760599101 | **0.429693777** | 0.437349478 | 70.2483807% | 48.7984938% |
| E23 width 16 | 0.782789142 | 0.448186815 | 0.439264753 | 55.8099399% | 45.6351459% |
| SAITS selected P4 | 0.790944151 | 0.440584382 | 0.436957896 | — | — |
| E22 selected P4 | 0.782638683 | 0.443195461 | **0.433248216** | — | — |

Width 64 beats SAITS test MSE by `0.010890605` (about `2.47%`) and improves over width 16 on both forecast metrics. It does **not** beat SAITS test MAE (`0.436957896`) or E22 test MAE (`0.433248216`); the original two-metric accuracy goal remains open. The DRRs use the unchanged cached Clean/Missing denominators and formula in the width 16 report. All test results cover 2,688 windows in train-standardized space; no baseline selected test was rerun.

## Training and diagnostics

Stage 1 selected epoch 99 by fixed-validation hidden-coordinate repair MSE `107.139995715`; its validation repair MAE was `0.780900948`, down from width 16's `0.917613758`. Stage 2 started a new optimizer and selected epoch 26 by fixed-validation forecast MSE `0.760599101`, with validation MAE `0.593048152`. Both stages completed all 100 epochs; elapsed times were `266.588` and `196.775` seconds. Stage 1 overlapped an E22 timing-only replay on the other GPU, so its wall time is descriptive and not a controlled SAITS speed verdict.

The selected width 64 test hidden-history repair MSE/MAE were `1.872836373`/`0.940488111`. Forecast improved despite this Stage 2 repair diagnostic being worse than width 16's `1.621225976`/`0.900597975`; future forecast optimization should track both without treating one as a substitute for the other.

On damaged test patches, H/V/T availability was `0.88074/1.0/1.0`; average none/H/V/T route weight was `0.06924/0.40878/0.24619/0.27579`, and mean H/V/T candidate counts were `3.3610/10.1895/4.7463`. The H memory/key hashes were identical across selected Stage 1/2 and unchanged after profiling and selected evaluation. The Stage 2 validation log's lowest MAE occurred at epoch 87 (`0.590439670`, MSE `0.764769530`), but the prespecified MSE selector chose epoch 26; epoch 87 was not tested in this trial.

## Selected input throughput and provenance

On an idle RTX 5090 GPU 0, fixed test inputs, batch `128`, warmup `5`, timed `10` batches, width 64 processed `20,616` windows/s end to end (`0.062089` seconds for 1,280 windows); model-core time was `0.053426` seconds. The cold first batch was `0.172` seconds, model load `0.739` seconds, H tensor-index construction `0.0043` seconds, and peak GPU allocation `217.1` MiB. Under the same boundary SAITS processed `25,339` windows/s and E22 `424` windows/s. PyTorch was `2.7.1+cu128`, CUDA `12.8`. Speed was measured without computing test metrics and is secondary to forecast accuracy.

- Protocol source CSV SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; fixed validation/test mask SHA-256 `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c`/`f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`.
- E23 width 64 config SHA-256 `f6e11fe8a1256a672e07d97af958fc7091f7cd1ee728f571cc550a9060ef4ebf`; Stage 1/2 checkpoint SHA-256 `d2f3a7c57449ea8ca84c41b8f96aef982552a92626a1c57dfaf77fe985e57a08`/`3ac529001c97f75828da56ec0d0169cb4c220117475c6392ed2afcab5c8264f4`.
- H memory/key SHA-256 `9ca39319c2847c0efb7d58d60a8e4a526e51bb8361197236e121144bc22a4b25`/`1ad11fea2978857761f6f75b3c3135df1744c6c34f082a8ec980301ceeaa84f6`.
- Manifests: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_width64_v2/stage1_p4/runs/e23-stage1.json` and `stage2_p4/runs/e23-stage2.json`. The sole selected test is `evaluations/e23-width64-test.json` under that root; the fixed-input profile is `benchmarks/e23_selected_test_profile.json`.

This single-seed result is exploratory. A separate validation-driven iteration is needed to improve MAE while retaining the MSE gain; the width 64 selected test and checkpoint remain immutable.
