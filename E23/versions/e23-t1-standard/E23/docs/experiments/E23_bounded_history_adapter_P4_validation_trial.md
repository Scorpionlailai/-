# E23 bounded historical-variable adapter: ETTh1 P4 validation trial

Date: 2026-09-30. This v7 trial applied the fixed `8*tanh(x/8)` transform to normalized historical source and target patch scalars in the optional flat-v3 historical-variable adapter. The H/V/T parent, H archive and retrieval state, masks, optimizer objective and selection policy match v6. The unbounded `aligned_v1` implementation and the default flat v3 path remain available.

The trial trained all 100 adapter-only epochs from the selected v3 Stage 2 checkpoint, using seed 2021, batch 128 and Adam learning rate 0.001. The earliest lowest fixed-validation MAE over epochs 0..100 was selected. Training took `151.971` seconds and all parent weights and H state remained unchanged.

| Candidate | Validation MSE | Validation MAE |
|---|---:|---:|
| Flat v3 parent / v6 selected epoch 0 | 0.764769533 | 0.590439666 |
| v7 best trained MAE, epoch 1 | 0.764079350 | 0.590516540 |
| v7 lowest trained MSE, epoch 2 | 0.763098530 | 0.591600720 |
| **v7 selected, epoch 0** | **0.764769533** | **0.590439666** |

Although epoch 1 reduced MSE, its MAE was higher than v3 by `0.000076874`; no trained epoch beat v3 MAE. **The predeclared dual validation gate failed. No v7 ETTh1 test evaluation or test-input speed profile was run.** The E23 test comparison therefore continues to use the previously tested v3 result, MSE `0.427264729`, MAE `0.434448576`.

A read-only selected-checkpoint validation pass reproduced the stored MSE/MAE exactly over the fixed validation split. The historical-variable adapter was available on `179,420` of `203,715` damaged patch targets (`88.0740%`). Its maximum absolute bounded source scalar was `8.0` in float32 across `43,821,696` valid source scalars. All seven selected gates are zero, as expected for epoch 0, and H state was unchanged. The transform fixed the measured input range, but this trial did not establish an accuracy gain.

Ignored artifacts are under `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_flat_history_adapter_width64_v7/`. Selected manifest SHA-256 `ef5944f80f320d7c91a544010989f4d5869cdca23426145e3675011d2feb741e`; selected checkpoint SHA-256 `4bb19be5998c2f6f9ec1ac60b87277894ccfba2a76e6b1ce39a2abd0f0e2bc7e`; validation diagnostic JSON SHA-256 `9dd5809a1a274daf10bc3601c8869ec66495f211b8c2a6e2ed84a167c0c53957`. Parent v3 manifest/checkpoint SHA-256 values are `18ac3d7a2a72cd1ea26866be3da0f602d54084b180c0b45323c24e77aa41a33c` / `1501fa96be1e43361d651d9a5889b3d7e3df147cfc18f918c8ad930aace576d9`. The v7 training code revision was `7ffcb44bf3816ebf7568061418fd1bf50f06de7d`.

Before training, the full repository suite passed `676` tests with one existing skip using a temporary cataloged E22 mask fixture link; the link was removed. The selected v6 `aligned_v1` artifact also loaded successfully after the v7 code change. The flat v3 checkpoint remains the active accuracy result. Its prior test was used in earlier model decisions, so any future test improvement requires independent confirmation.
