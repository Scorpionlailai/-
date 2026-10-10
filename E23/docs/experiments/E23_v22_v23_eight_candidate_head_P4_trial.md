# E23 v22/v23: candidate-reading eight-head trial on ETTh1 P4

Date: 2026-10-01. Status: complete; v12 remains selected.

## Design

The comparison uses the locked ETTh1 P4 96→96 protocol, fixed E22 masks, seed 2021, batch 128, and 100 epochs in each stage. Stage 1 selects the earliest minimum validation hidden-coordinate repair MSE. Stage 2 uses forecast MSE + 0.2 MAE and selects the earliest minimum validation forecast MAE. Both trials use v12 H top-4, window-conditioned V top-4 with offsets −1/0/+1, and ten T bands. The original PatchTST prediction head remains unchanged.

- v22: width-128 single H/V/T router, matched capacity control.
- v23: eight 16-dimensional heads, each attending independently to selected H/V/T candidates, routing none/H/V/T independently, and concatenating its output to width 128.

All listed validation metrics were replayed from the selected checkpoints; H key state stayed unchanged. The prespecified promotion rule requires validation MAE below v12's 0.5882120694703256 and validation MSE no higher than 0.7454477286392869. Test evaluation was reserved for a passing trial.

## Results

| Model | Stage 1 best epoch | Stage 1 repair MSE | Stage 2 best epoch | Validation MSE ↓ | Validation MAE ↓ | Promotion |
|---|---:|---:|---:|---:|---:|---|
| v12 selected, width 64 | 51 | 106.906964 | 53 | **0.745448** | **0.588212** | Incumbent |
| v22 single-head width 128 | 95 | 107.078476 | 23 | 0.759926 | 0.594541 | Fail |
| v23 eight-head width 128 | 99 | 107.063878 | 25 | 0.753421 | 0.590573 | Fail |

Relative to its matched v22 control, v23 lowers validation MSE by 0.006506 and MAE by 0.003968. Relative to v12, v23 remains worse by 0.007973 MSE and 0.002361 MAE. Thus the candidate-reading heads add useful capacity compared with a single width-128 router, but do not improve the selected E23 model. No v22/v23 test metrics were produced.

## Head diagnosis on selected v23 validation checkpoint

Eight heads learned different mean none/H/V/T routes on truly damaged patches; the most T-oriented head (head 3) assigned T weight 0.518, whereas heads 4 and 5 assigned V weight 0.501 and 0.497. Mean pairwise candidate-attention L1 distances were H 0.265, V 0.090, and T 0.103. Seven of eight head-suppression ablations worsened both forecast metrics. Suppressing head 6 improved validation MAE by 0.002449 and MSE by 0.004498, indicating a harmful head on this validation set. That post-hoc ablation is diagnostic only: its MAE 0.588124 narrowly crosses the v12 MAE threshold, but MSE 0.748922 still fails the gate, and it was chosen after inspecting validation results. It must not be reported as a selected model or evaluated on test.

The result suggests that a future head-level reliability gate could be explored prospectively, with its structure fixed before training. It does not justify manually removing head 6 from this checkpoint. Since v12 has been compared repeatedly on the same validation set, a final paper claim will also need an independent dataset or seed.

## Artifacts

- Configs: `configs/e23_multi_expert_p4_width128_window_singlehead_v22.yaml`, `configs/e23_multi_expert_p4_width128_window_eighthead_v23.yaml`.
- Selected validation summary: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v22_v23_validation_selection.json`.
- v23 head diagnosis: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v23_eighthead_128/head_validation_diagnostic_v2.json`.
- Individual Stage 1/2 manifests and validation replays are under the respective `e23_v22_singlehead_128` and `e23_v23_eighthead_128` experiment directories.

Verification: 128 E23 unit/integration tests passed; old v12 selected checkpoint loaded strictly after the code change; `git diff --check` passed for modified implementation files. The separate user-edited E22 result CSV was left untouched.
