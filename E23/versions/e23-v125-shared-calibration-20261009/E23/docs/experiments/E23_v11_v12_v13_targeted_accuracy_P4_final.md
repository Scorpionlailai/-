# E23 targeted accuracy iterations: ETTh1 P4 final record

Date: 2026-09-30. The user selected directions 1, 3 and 4 in sequence: Stage 2 mixed MSE/MAE loss (v11), window-conditioned V source selection (v12), then two-group causal H shortlist and visible-error rerank (v13). All variants retain the flat H/V/T repair, 64-wide repair latent, and original width-16 PatchTST predictor. The ETTh1 P4 protocol is 96→96, seven variables, patch length 16/stride 8, fixed E22 masks, train-only standardization, seed 2021, batch 128, and 100 epochs per trained stage. Stage 2 checkpoints were selected by earliest minimum fixed-validation forecast MAE. The predeclared promotion gate required lower validation MAE than the incumbent and validation MSE no higher than original v3 `0.7647695325`.

## Sequential validation decisions

| Trial | Stage 1 | Stage 2 loss | Selected Stage 2 epoch | Validation MSE ↓ | Validation MAE ↓ | Decision |
|---|---|---|---:|---:|---:|---|
| Original E23 v3 | Original | MSE | 87 | 0.764769533 | 0.590439666 | Starting incumbent |
| v11: mixed loss | Reused v3 | MSE + 0.2 MAE | 87 | 0.763651553 | 0.589048978 | Promoted |
| **v12: contextual V** | Fresh, epoch 51 | MSE + 0.2 MAE | 53 | **0.745447729** | **0.588212069** | **Promoted** |
| v13: two-group H on v12 design | Fresh, epoch 99 | MSE + 0.2 MAE | 87 | 0.777758755 | 0.593380181 | Rejected; v12 retained |

Both v12 and v13 completed all 100 epochs in each newly trained stage; each log contains 100 validation records per stage. V12 Stage 1 selected hidden-coordinate repair MSE `106.9069637043`, and v13 Stage 1 selected `107.0926960600`. Selected-checkpoint replays over all 2,688 fixed-validation windows reproduced both v12 and v13 manifest forecast metrics exactly. Stage 2 kept each selected H memory/key state unchanged. V13 improves neither forecast metric over v12 and also exceeds the original v3 MSE guard. The H rerank mechanism therefore remains an experimental branch; the retained design uses v12's original one-group H query.

V12's four-source V choice varies by window and patch. Across validation target patches, the seven source-variable slot counts were `[135670, 181963, 110068, 123735, 129027, 156685, 66020]` (903,168 slots); 35.54% of adjacent target-patch comparisons changed at least one ordered source. Damaged targets had 10.41 valid V source/offset candidates on average. V13's final H candidate-count histogram on damaged validation targets was `[24295, 0, 2781, 0, 176639]` for 0–4 candidates, with H available on 88.07%. These diagnostics describe the selected models; they do not isolate causal effects of each expert.

## One final selected test: v12

Only v12 received a new selected test and a fixed test-input profile after all validation decisions. The test covered the same 2,688 fixed ETTh1 P4 windows and train-standardized metric space as the existing baselines; v3, E22, SAITS, S4M, and the PatchTST references were not rerun.

| Existing selected result | Test MSE ↓ | Test MAE ↓ |
|---|---:|---:|
| **E23 v12** | **0.425612804** | **0.431006141** |
| E23 v3 | 0.427264729 | 0.434448576 |
| SAITS | 0.440584382 | 0.436957896 |
| E22 | 0.443195461 | 0.433248216 |

V12 improves on SAITS by `0.014971578` MSE and `0.005951755` MAE, and on E22 by `0.017582657` MSE and `0.002242075` MAE. Relative to v3, it lowers MSE by `0.001651925` and MAE by `0.003442436`. Its hidden-history repair MSE/MAE were `1.515454333` / `0.853963607`. On damaged patches, H/V/T availability was `0.880711/1.0/1.0`, mean route weight `0.369555/0.284923/0.270242` (none `0.075279`), and mean H/V/T candidate counts `3.346013/10.416935/4.746283`. Shared PatchTST Clean/Missing DRR was `73.4346%` for MSE and `59.2754%` for MAE. The selected test reports H state unchanged.

The fixed-input profile used batch 128, five warmups and ten timed test-input batches on an RTX 5090. It processed 1,280 windows in `0.064512` seconds, or `19,841` windows/s through the same end-to-end timing boundary as prior E23 profiles. Model-core time was `0.055734` seconds, peak allocated GPU memory `227,643,392` bytes, and cold first batch `0.168809` seconds. This profile measured no test error; the selected test above is the only new accuracy evaluation.

## Reproducibility and limits

- V12 selected Stage 1 manifest/checkpoint SHA-256: `029aaab1521c9ddc81cf14a0c4c3ae53687283c2558ccdb7cb70a41f1ae82b84` / `c8c8ea04a5c60e03719bfd1a4a35c66fd38c486877192066cf8e03c147c88dff`. Selected Stage 2 manifest/checkpoint SHA-256: `899838f6e1e4d0753e15d84c221019d34c65f95e12870552ef164d3b86a04a68` / `c3ae09195b793b144936f38ceba0909f377d7fb02cfea626689fb21e70af3c35`. Test JSON SHA-256: `cd8d14472a855b7d8401b8409c1343c026019aa4fea279fab4a2815e96dd9c59`; profile JSON SHA-256: `d8ba007efe3344952986ccfe450f9abefa2ea9c2bf0baa98034b507ebb6b1fe6`.
- V13 selected Stage 1 manifest/checkpoint SHA-256: `2e999221e27f7a8095fe4ded57b35e3f48908fb496790bd94fb5328d667952df` / `aa75be351d37ff462790e7e8d7a6bd093894bad64ae22065f6f7ae0e83c4fb8e`. Selected Stage 2 manifest/checkpoint SHA-256: `79dce9aee6d4c54256ed9ec6e1b41fcb0dba3bd0007a7d590d8ec872cc7a284f` / `7024bfbd3dea2c284f1f1d5a43cf45a4fb6be89b6c301c47dc5dee0c99721edf`. V13 received validation replay only; no test.
- Source CSV SHA-256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; fixed validation/test mask SHA-256: `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c` / `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`. V12/V13 config SHA-256: `a9a2ab31fd22bf5bf55d449b7f3cd0d60821d54a5d82f69002b6aa4ae14a8e1b` / `e98ea7217fbe6363182388472f6d8a08462e6a80ece534dcde26333c9b9fc929`.
- Ignored artifact roots: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v3_window_v12/` and `e23_v3_two_group_h_v13/`. The v12 selected test is `evaluations/e23-v12-selected-test.json`; the test-input profile and read-only replay are in `benchmarks/`.

This result comes from a single seed and a validation set repeatedly used during E23 design. Earlier E23 variants also informed choices after seeing ETTh1 test performance. The v12 selected test is therefore encouraging but exploratory; independent datasets and seeds are needed before claiming a general advantage over SAITS or E22.

Follow-up validation-only trials [v14](E23_v14_late_horizon_loss_P4_trial.md) and [v15](E23_v15_stronger_context_V_P4_trial.md) both failed to improve v12; neither received a new selected test. V12 remains the current best completed E23 version.
