# E23 v27: bounded channel-wise V candidate aggregation

Date: 2026-10-02. Status: complete; v12 remains selected under the fixed two-metric gate.

## Hypothesis and one controlled change

v26 let each of the 64 V channels add an unrestricted learned score to each selected source/offset candidate. It lowered validation MAE by 0.000416 relative to v12 while worsening MSE by 0.009144. v27 keeps the same four source variables, three offsets, candidate message projector, scalar quality, H/T experts, H/V/T router, explicit repair and PatchTST predictor. It changes only the channel score offset to `0.5 * tanh(linear(candidate_message))`. This bounds each offset to ±0.5; with fixed base scores, a valid candidate's channel-specific weight cannot differ from its scalar weight by a factor outside `[exp(-1), exp(1)]`. Zero initialization still starts at the v12 scalar aggregation.

## Locked evaluation

Use the cataloged E22 masks under ETTh1 P4 96→96, seed 2021, batch size 128 and 100 epochs for each stage. Stage 1 selects minimum fixed-validation hidden-coordinate repair MSE. Stage 2 uses forecast MSE + 0.2 MAE and selects minimum fixed-validation forecast MAE. Replay that selected checkpoint over all validation windows and verify H read-only state. Promote v27 and run held-out test only if selected validation MAE is strictly below v12's **0.5882120694703256** and validation MSE is no higher than v12's **0.7454477286392869**. Otherwise retain v12. No cap search or alternate checkpoint selection is permitted after viewing results.

Config: `configs/e23_multi_expert_p4_width64_bounded_channelwise_v27.yaml`. Trial root: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v27_bounded_channelwise_V/`.

## Result

| Model | Stage 1 best epoch / repair MSE | Stage 2 best epoch | Validation MSE ↓ | Validation MAE ↓ |
|---|---:|---:|---:|---:|
| v12 incumbent | 51 / 106.906964 | 53 | **0.745448** | 0.588212 |
| v26 unrestricted channel V | 68 / 106.953485 | 86 | 0.754592 | **0.587796** |
| v27 bounded channel V | 99 / 107.202249 | 87 | 0.756037 | 0.588205 |

v27 selected MAE improves on v12 by only 0.0000067363, while MSE worsens by 0.0105891174. Relative to v26, both selected metrics worsen. Across all 100 validation epochs, v27's minimum MSE was 0.75224479 at epoch 49, so no epoch met the v12 MSE ceiling. **v27 fails the predeclared two-metric gate; v12 remains the selected E23 version.** The v27 test set was not evaluated.

The selected checkpoint replayed all 2,688 fixed-validation windows and 1,806,336 forecast coordinates exactly: MSE 0.7560368460051867 and MAE 0.5882053331993605. The H key hash was identical before and after replay. On the first validation batch, the mean absolute channel-versus-scalar weight difference was 0.02425, versus 0.08081 in v26. This confirms that the cap reduced weight deviation, but that reduction did not recover v12's MSE. These first-batch diagnostics are mechanism checks, not accuracy results.

The Stage 1/2 manifests are under `stage1/runs/e23-stage1.json` and `stage2/runs/e23-stage2.json`. Replay and mechanism diagnostics are `validation_replay.json` and `channelwise_diagnostic.json` in the trial root. Verification: 139 E23 unit/integration tests passed; v12 and v26 selected checkpoints loaded strictly after the change. Repeated use of one ETTh1 validation set means the tiny MAE difference should not be interpreted as a generalization gain.
