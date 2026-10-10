# E23 Multi-Expert Explicit Recovery: ETTh1 P4 Width 16 Trial

Date: 2026-09-30. This records the corrected width 16 E23 run. An earlier Stage 1 engineering run was superseded after a final-patch retrieval-key padding fix and did not feed Stage 2 or selected testing. The selected test below was executed once.

## Locked protocol and outcome

All models use ETTh1 history/prediction `96/96`, patch `16/8`, seven variables, train-only standardization, dynamic training masks, the same fixed validation/test masks, batch `128` and seed `2021`. Forecast errors are in train-standardized space over 2,688 fixed test windows. The source CSV SHA-256 is `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; validation/test mask SHA-256 values are `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c` and `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`.

| Selected model | Test MSE | Test MAE |
| --- | ---: | ---: |
| E23 width 16 | 0.448186815 | 0.439264753 |
| SAITS, archived selected P4 | 0.440584382 | 0.436957896 |
| E22, archived selected P4 | 0.443195461 | 0.433248216 |

Width 16 did not meet the original accuracy thresholds of MSE below SAITS `0.4405843816917229` and MAE below E22 `0.4332482160624463`. The selected E23 test also reports hidden-history repair MSE/MAE `1.621225976`/`0.900597975`. The latter is a diagnostic, not the forecast-selection metric.

The shared DRR denominators are the existing PatchTST Clean/Missing cached tests: Clean MSE/MAE `0.391587319`/`0.406349116`; Missing MSE/MAE `0.519669281`/`0.466894917`. With `DRR=(Missing−E23)/(Missing−Clean)×100`, E23 width 16 has DRR-MSE `55.809940%` and DRR-MAE `45.635146%`. No Clean/Missing test was rerun.

## Selection and expert behavior

Stage 1 ran 100 epochs and selected epoch 17 by fixed-validation hidden-coordinate repair MSE `107.339907216`; its separately measured validation repair MAE was `0.917613758`. Stage 2 started a new optimizer, ran 100 epochs with forecast MSE only, and selected epoch 100 by validation MSE `0.782789142` (MAE `0.600139394`). Selected Stage 1/2 training times were `183.557` and `162.571` seconds, totaling `346.128` seconds. The published H memory/key hashes match across both stages and remained unchanged during profiling and test.

On the damaged test patches, availability rates were H `0.74436`, V `1.0`, T `1.0`; average none/H/V/T route weights were `0.24952/0.26619/0.28985/0.19445`. Mean H/V/T candidate counts were `2.7895/10.2020/4.7463`. A validation-only selected-checkpoint ablation gave forecast MSE `0.838075` without H, `0.803586` without T and `0.777363` without V, versus full-model `0.782789`; V removal slightly worsened validation MAE (`0.602468` versus `0.600139`). These diagnostics did not alter the selected test.

## Timing boundary

The selected models were loaded by their checked manifests and profiled on one idle NVIDIA GeForce RTX 5090 using the same fixed test input order, batch `128`, five warmup batches and ten timed batches (1,280 timed windows). PyTorch was `2.7.1+cu128`, CUDA `12.8`. Full-pipeline time includes batch preparation, host-to-device transfer, repair and forecast; model-core time starts after transfer. It does not compute test error. Model load and H index construction are reported separately.

| Model | Full pipeline windows/s | Model-core seconds / 1,280 windows | Cold first batch, s | Peak GPU allocation, MiB |
| --- | ---: | ---: | ---: | ---: |
| E23 width 16 | 22,984 | 0.047208 | 0.171 | 83.6 |
| SAITS | 25,339 | 0.041948 | 0.192 | 176.9 |
| E22 | 424 | 3.010129 | 0.574 | 678.7 |

E23 model load took `0.728` seconds and H tensor-index construction `0.0038` seconds. The E23 speed profile used the later output-equivalent batched T/router execution path; rechecking the selected checkpoint on validation changed MSE only at floating-point scale (`0.7827891406` versus stored `0.7827891417`). The original selected test metric was not rerun. Controlled full 100+100 SAITS timing on GPU 0 was `118.374+141.395=259.769` seconds, faster than width 16 E23; speed is now a reported measure rather than the primary acceptance gate. An E22 Stage 1 timing replay took `1440.141` seconds. Its Stage 2 timing-only replay was stopped at epoch 37 after concurrent work made the timing non-comparable; the archived E22 two-stage `1445.193+2367.523` seconds remain historical context, not a controlled same-run comparison.

## Provenance

- E23 Stage 1/2 manifests: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_v1/stage1_p4_v2/runs/e23-stage1.json` and `stage2_p4_v2/runs/e23-stage2.json`; selected checkpoint SHA-256 values `b398e810abfbcc32671de3f145e6cabae36e25e903e3292ff7380e2d6693dc3b` and `3587374055cc1ef49a2cf92f158271297190787f66be57736c6f8544ac377194`.
- E23 config SHA-256 `3e310d86f90ff28c0bf8f42834249558c75b542a2d1a892ba6baf525822ea217`; protocol config SHA-256 `8198daf84c6b583770f2d11c3c4fc2463aa24b4e713312374b07567d4a9797eb`; Stage 2 memory/key hashes `5dacd3c0c02fee37304fd20903210aa49216fa95352d02783b20271a04a2e7a5`/`99134cbc7fc8007752ecb1a6dc09a5072cec35cf77a6e556d9a76c347f51767c`.
- One-shot test: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_v1/evaluations/e23-test-v2.json`. Idle test-input speed records: adjacent `benchmarks/e23_selected_test_profile_idle_current_code.json`, `saits_selected_test_profile_idle.json`, and `e22_selected_test_profile_idle.json`.

This is a single-seed exploratory result. The next width 64 variant has separate config, checkpoints, profile and one-shot test records.
