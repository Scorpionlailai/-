# ETTh1 P4 additional baselines (2026-10-02)

Resource reporting update: the CSV now includes recorded loop time for **all** required training stages, plus full-workflow time and peak CUDA allocated/reserved fields for future measured runs. Historical full-workflow time and training memory were not recorded and remain blank. Scope, configuration differences from the papers, and reproduction limits are documented in [the resource and reproduction audit](2026-10-02_baseline_reproduction_and_resources_audit.md). Existing forecast scores have been preserved.

The six baselines were run in the agreed order: CRIB, CoIFNet, BiTGraph, ChannelTokenFormer, T1 plus the shared PatchTST head, and BRITS plus the shared PatchTST head. The complete numeric table, including E22, SAITS, S4M, and the selected E23 v12 reference, is in [ETTh1_P4_extended_baselines_2026-10-02.csv](ETTh1_P4_extended_baselines_2026-10-02.csv).

## Locked comparison

- ETTh1, seven variables, 96 observed steps to 96 forecast steps; P4 variable missing blocks with start rate 0.10 and block length 8.
- Train, validation, and test time boundaries: `[0,8640)`, `[8640,11520)`, `[11520,14400)`. Each evaluation uses 2,688 windows and 1,806,336 forecast targets. Train-only variable standardization; seed 2021.
- The source CSV SHA-256 is `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. The locked validation and test mask SHA-256 values are `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c` and `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`.
- Each new baseline completed 100 forecast epochs; T1 and BRITS also completed 100 preceding imputation epochs. A checkpoint was selected by the earliest minimum fixed-validation forecast MAE. The selected validation score was replayed before one test evaluation. All reported forecast metrics are in train-standardized units; lower is better.

| Model | Validation MSE | Validation MAE | Test MSE | Test MAE | Selected forecast epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| **E23 v12** | **0.745448** | **0.588212** | **0.425613** | 0.431006 | 53 |
| CRIB | 0.814480 | 0.614301 | 0.612933 | 0.552574 | 80 |
| CoIFNet | 0.827019 | 0.604723 | 0.457392 | **0.430951** | 97 |
| BiTGraph | 0.844372 | 0.629620 | 0.536847 | 0.513945 | 12 |
| ChannelTokenFormer | 0.795896 | 0.621776 | 0.550205 | 0.524615 | 90 |
| T1 + shared PatchTST | 0.775441 | 0.595749 | 0.430288 | 0.434663 | 11 |
| BRITS + shared PatchTST | 0.842885 | 0.627185 | 0.556675 | 0.539466 | 8 |

E23 v12 retains the lowest validation MAE and MSE and the lowest test MSE among these runs. CoIFNet has a test MAE advantage of `0.0000551185` over v12, but a test MSE disadvantage of `0.0317791022`, and both of its validation metrics are worse. This tiny MAE difference from one seed does not establish a stable ordering. T1 plus the shared head is close on test MSE (`+0.004676` versus v12) while trailing on MAE (`+0.003657`).

## Adaptation details

| Model | Source revision | P4 adapter and training objective |
| --- | --- | --- |
| [CRIB](https://github.com/Muyiiiii/CRIB) | `a457672c7b0152f74c929858dba2a9c886405519` | Official forecast model; its MAE, consistency MSE, and KL terms. |
| [CoIFNet](https://github.com/KaiTang-eng/CoIFNet) | `114c964945cbd1a1b1791ca403d08f827e3c9fe4` | Official forecast model; `0.1 × forecast MAE + 0.9 × observed reconstruction MAE`. |
| [BiTGraph](https://github.com/chenxiaodanhit/BiTGraph) | `4f1d05bcc20bb3f5084bd1facc995478f84a40ed` | Official forecast model; forecast MAE loss. |
| [ChannelTokenFormer](https://github.com/jinkwan1115/ChannelTokenFormer) | `b1c100effb9c27cd25de07e91de7f9005d18d5e7` | Official `ChannelTokenFormer_missing` model, forecast MSE loss. Its patch-removal code reads only sample 0's empty-patch pattern, so the adapter groups samples by identical full-patch visibility before each forward pass. This preserves each sample's actual P4 mask without editing upstream source. |
| [T1](https://github.com/Oppenheimerdinger/T1) | `823b76aab57a2e60eb5029caab06fdfec8b19256` | Official ETTh1 96-step imputer settings; hidden-coordinate MSE in train-standardized units for Stage 1; shared P4 PatchTST head with joint forecast MSE fine-tuning in Stage 2. Stage 1 selected hidden-coordinate MAE at epoch 99 (`0.260810`); Stage 2 selected epoch 11. |
| BRITS | SAITS repository revision `a3cd8de756c4de8a16bebb9d228deef7d71eb718` | The SAITS repository's BRITS implementation, hidden size 128, MIT + reconstruction + consistency training on train-standardized inputs. Stage 1 selected hidden-coordinate MAE at epoch 37 (`0.419150`); the shared P4 PatchTST head was jointly fine-tuned with forecast MSE and selected epoch 8. This is not a direct run of the original authors' BRITS repository. |

The T1 and BRITS adapters clamp originally observed values exactly before passing repaired windows to the shared PatchTST predictor. Their imputation targets are used only for Stage 1 supervision, never as model inputs. The initially attempted BRITS run with per-window normalized inputs was stopped before selection: rare near-zero window standard deviations inflated hidden targets to a maximum absolute value of about 1,242. The completed BRITS run uses train-only global standardization, matching its source repository's data preparation. The aborted log is retained separately and is not part of the comparison.

## Reproduction and limits

The adapters are in `integrations/forecast_compare/*_p4.py`; runners are `scripts/run_{crib,coifnet,bitgraph,ctf,t1,brits}_p4.py`. Each runner accepts `--smoke` or stage-specific smoke options, training options, and `--evaluate-selected`. The upstream checkouts are under ignored `artifacts/upstream/`; BRITS uses `../SAITS`. Selection JSON, selected checkpoint, training log, and test JSON are under `artifacts/<model>_etth1_p4/`. Selection records contain upstream and source hashes, dataset and mask hashes, checkpoint hash, model settings, and the full 100-epoch completion count.

These are single-seed adaptations to **missing-lookback forecasting** under one ETTh1 P4 setting. Native forecasting models retain their source-specific losses and architectures; T1 and BRITS are imputation encoders with the common PatchTST predictor. They should be described as task adaptations, not the originally reported benchmark numbers. The small CoIFNet test MAE difference in particular needs independent seeds and datasets before any ranking claim.
