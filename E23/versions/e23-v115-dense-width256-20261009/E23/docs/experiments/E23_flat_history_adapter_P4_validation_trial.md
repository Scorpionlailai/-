# E23 flat historical-variable adapter: ETTh1 P4 validation trial

Date: 2026-09-30. This v6 trial kept the selected width-64 flat H/V/T E23 v3 model frozen and trained only a zero-gated historical-variable adapter. Its evidence came from cross-variable patches aligned to H candidates in the training archive, before the current lookback window.

## Locked run and selection

The ETTh1 P4 96/96 protocol, fixed masks, seed 2021, batch size 128, forecast-MSE-only objective and selected v3 parent were retained. Adapter-only Adam ran all 100 epochs at learning rate 0.001; the earliest fixed-validation MAE minimum over epochs 0..100 was selected. Epoch 0 exactly reproduced the parent: validation MSE `0.7647695325129157`, MAE `0.5904396656148828`. **Epoch 0 remained selected after all 100 epochs.** Training took 155.750 seconds, with 24,071 trainable adapter parameters. H bank/key and every parent weight remained frozen and hash-checked during training.

| Selected model | Validation MSE | Validation MAE |
|---|---:|---:|
| Flat E23 v3 parent | 0.764769533 | 0.590439666 |
| v6 selected, epoch 0 | 0.764769533 | 0.590439666 |
| v6 best trained epoch by MSE, epoch 11 | 0.762804240 | 0.592124290 |

Epoch 11 slightly reduced MSE but increased MAE; selecting it would violate the locked MAE rule. No trained epoch beat the parent MAE. The predeclared gate required strictly lower MAE and no higher MSE, so **no v6 ETTh1 test evaluation or test-input profile was run**. The previous v3 selected test remains MSE `0.427264729`, MAE `0.434448576`, versus SAITS `0.440584382`/`0.436957896`; these are prior results, not v6 results.

## Validation-only input diagnostic

A read-only inspection of the selected model's historical-variable inputs found a major train/validation scale difference. Across 21 training batches, the maximum absolute normalized archive source was `62.329746`; none exceeded 100, and the minimum query-window standard deviation was `0.06117337`. Across all 21 validation batches, the maximum was `2559.73535`, `0.7177%` of valid source scalars exceeded 100, and the minimum standard deviation was `0.0031622776`. Historical-variable evidence was available for `80.94%` of training patch targets and `79.46%` of validation patch targets. Thus the failure cannot be explained simply by absence of candidates. The scale mismatch motivates one controlled bounded-input variant; this diagnostic alone does not establish the cause of the validation error.

## Provenance

The ignored selected manifest is `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_flat_history_adapter_width64_v6/stage2_p4/runs/e23-stage2-history-adapter.json` (SHA-256 `e23c502bf84a195e705409d8b53ff0fc4df414d498b84400827b567361bc3d6d`), checkpoint SHA-256 `0395e7205b2f79e948fe71267e7c6ac1f852780317a11fe6f3a68ff0fa90c6cd`. Parent v3 Stage 2 manifest SHA-256 `18ac3d7a2a72cd1ea26866be3da0f602d54084b180c0b45323c24e77aa41a33c`; parent checkpoint SHA-256 `1501fa96be1e43361d651d9a5889b3d7e3df147cfc18f918c8ad930aace576d9`. H memory/key SHA-256 are `9ca39319c2847c0efb7d58d60a8e4a526e51bb8361197236e121144bc22a4b25` and `1ad11fea2978857761f6f75b3c3135df1744c6c34f082a8ec980301ceeaa84f6`; train archive SHA-256 `ae5ee3c28cf6884185c4c90a8273c63e007923db312382848d3a2a957dcc5811`. Code revision during training was `1f4a4608558edc3b8129ec4554a32bb76d07cd72`.

The full repository suite passed 668 tests with one existing skip after the cataloged E22 mask fixture was temporarily linked at the legacy local path; the link was removed afterward. This negative result leaves flat v3 as the active selected model.
