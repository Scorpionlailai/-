# E23 v25: eight full-width candidate heads with learned head selection

Date: 2026-10-01. Status: complete; v12 remains selected under the fixed two-metric gate.

## Design and protocol

Each of eight heads reads the selected H/V/T candidates and produces a 64-dimensional residual proposal. The head-level gate scores the proposals and a no-change option for each damaged patch; their weighted sum returns a 64-dimensional repair latent. H/V/T candidate selection runs once, and the explicit repair interface and original PatchTST predictor are unchanged. This tests the user's proposal to preserve a full 64 dimensions per head while letting multiple heads choose useful information.

The locked ETTh1 P4 96→96 protocol, E22 masks, seed 2021, batch 128, v12 H/V/T candidate policies, and 100 epochs per stage were retained. Stage 1 selected minimum validation hidden-coordinate repair MSE. Stage 2 used MSE + 0.2 MAE and selected minimum validation forecast MAE. The selected checkpoint was replayed on validation; H key state remained unchanged.

| Model | Head design | Stage 1 best epoch / repair MSE | Stage 2 best epoch | Validation MSE ↓ | Validation MAE ↓ |
|---|---|---:|---:|---:|---:|
| v12 incumbent | one 64-dimensional H/V/T router | 51 / 106.906964 | 53 | **0.745448** | 0.588212 |
| v23 | eight 16-dimensional heads, concatenated to 128 | 99 / 107.063878 | 25 | 0.753421 | 0.590573 |
| v24 | eight 8-dimensional heads, concatenated to 64 | 3 / **106.862496** | 98 | 0.771615 | 0.596324 |
| v25 | eight 64-dimensional heads, gated to 64 | 99 / 107.082403 | 32 | 0.751190 | **0.587814** |

v25 lowers validation MAE relative to v12 by 0.000398, while increasing MSE by 0.005742. It improves both metrics relative to v23 and v24, but fails the predeclared promotion rule: MAE must be strictly below 0.5882120694703256 **and** MSE no higher than 0.7454477286392869. No v25 test evaluation was performed. Its small MAE improvement on one repeatedly used validation set is exploratory, not evidence of a generalizable advantage.

The two v25 training stages took 426.4 and 357.8 seconds, compared with 184.8 and 160.2 seconds for v12 on the same machine. The added head capacity therefore carries a substantial training cost in this implementation.

## Head-level diagnosis

On damaged patches, the mean gate weights were 0.005 for no change and 0.156, 0.132, 0.130, 0.039, 0.046, 0.094, 0.260, 0.137 for heads 0–7. Per-head H/V/T routes also differed. Removing any one head contribution post hoc increased both validation MAE and MSE, with MAE increases from 0.002708 to 0.016481. The gate therefore uses multiple heads and no head is clearly harmful on this checkpoint. These ablations are diagnostics on the same validation set and were not used to select a model.

The result supports the narrower hypothesis that full-width heads and learned head selection recover more useful information than concatenating eight narrow heads under the same 64-dimensional output. It does not explain the remaining MSE gap to v12. A future trial should target large forecast errors while preserving this MAE gain, with its criterion fixed before training; the present result must not be promoted by changing the rule after seeing it.

## Artifacts

- Config: `configs/e23_multi_expert_p4_width64_window_fullhead_v25.yaml`.
- Stage 1/2 manifests: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v25_fullhead_64/stage1/runs/e23-stage1.json` and `stage2/runs/e23-stage2.json`.
- Selected validation replay: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v25_fullhead_64/validation_replay.json`.
- Head gate and ablation diagnosis: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v25_fullhead_64/head_validation_diagnostic.json`.

Verification: 134 E23 unit/integration tests passed; v12/v23/v24 old selected checkpoints loaded strictly after the router change. The user-edited E22 results CSV was not modified.
