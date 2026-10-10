# E23 v26: channel-wise aggregation inside V

Date: 2026-10-02. Status: complete; v12 remains selected under the fixed two-metric gate.

## Hypothesis and controlled change

The v12 V expert selects four source variables per target patch and reads their three nearby offsets. Its candidate scores are shared by all 64 output channels. v26 retains these candidates, their validity checks and scalar quality summary, but adds a zero-initialized linear map from each visible candidate message to 64 channel-specific score offsets. A separate softmax across the 12 candidate slots for each channel forms the V message. H, T, the H/V/T router, explicit repair interface and PatchTST head are unchanged. The candidate evidence interface retains the scalar candidate weights. This is a controlled adaptation of feature-dependent cross-variable aggregation; it is not a reproduction of T1's entire architecture.

## Locked evaluation

Use ETTh1 P4 96→96, the cataloged E22 fixed masks, seed 2021, batch size 128, 100 epochs each stage and the v12 learning rates. Stage 1 selects lowest validation hidden-coordinate repair MSE. Stage 2 uses forecast MSE + 0.2 MAE and selects lowest fixed-validation forecast MAE. Replay the selected validation checkpoint and verify read-only H state. Promote v26 and run the held-out test only if validation MAE is strictly below **0.5882120694703256** and validation MSE is no higher than **0.7454477286392869**. Otherwise retain v12. A single seed and repeated use of ETTh1 validation make any gain exploratory.

Artifacts: `configs/e23_multi_expert_p4_width64_channelwise_v26.yaml` and `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v26_channelwise_V/`.

## Result

| Model | Stage 1 best epoch / repair MSE | Stage 2 best epoch | Validation MSE ↓ | Validation MAE ↓ |
|---|---:|---:|---:|---:|
| v12 incumbent | 51 / 106.906964 | 53 | **0.745448** | 0.588212 |
| v26 channel-wise V | 68 / 106.953485 | 86 | 0.754592 | **0.587796** |

v26 lowers selected validation MAE by 0.000416 but increases MSE by 0.009144. Its minimum MSE across the 100 validation epochs was 0.753362 at epoch 58, still above the v12 MSE threshold. No epoch simultaneously satisfied both thresholds, and the predeclared rule does not permit picking a different checkpoint after inspecting these results. **v12 remains the selected E23 version; v26 was not evaluated on the test set.** Given the repeated use of this validation set, the tiny MAE gain should not be treated as a generalization claim.

The selected v26 checkpoint replayed all 2,688 fixed-validation windows and 1,806,336 forecast coordinates exactly: MSE 0.75459166935512, MAE 0.5877961117664432. The H key hash before and after replay was identical. A first-batch mechanism check found nonzero trained channel projection norm (11.3459) and mean absolute difference 0.08081 between channel-specific and scalar candidate weights over active targets. Thus the learned channel weights are used, but this particular version trades larger squared errors for the small MAE improvement. These first-batch values describe the mechanism and are not an accuracy metric.

Stage 1/2 manifests are under `stage1/runs/e23-stage1.json` and `stage2/runs/e23-stage2.json`; replay and mechanism diagnostics are `validation_replay.json` and `channelwise_diagnostic.json` in the trial root. The config changes only V aggregation relative to v12. The original v12 selected checkpoint loaded strictly after the code change. Verification: 137 E23 unit/integration tests passed. The unrelated user-edited E22 CSV was left untouched.
