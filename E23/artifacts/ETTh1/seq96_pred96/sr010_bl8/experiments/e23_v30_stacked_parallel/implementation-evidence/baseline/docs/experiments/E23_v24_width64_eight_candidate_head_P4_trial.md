# E23 v24: eight candidate-reading heads with 64-dimensional output

Date: 2026-10-01. Status: complete; v12 remains selected.

## Locked change

v24 changes v23's eight heads from 16 dimensions each to 8 dimensions each. The concatenated repair latent and explicit repair decoder input are therefore 64-dimensional. H/V/T candidate pools, candidate-level attention, per-head none/H/V/T routing, original PatchTST forecast head, ETTh1 P4 masks, seed 2021, batch size 128, and training objectives are unchanged. The v12 width-64 single router is the direct capacity-matched reference.

Both stages ran 100 epochs. Stage 1 selected the earliest minimum fixed-validation hidden-coordinate repair MSE. Stage 2 used MSE + 0.2 MAE and selected the earliest minimum fixed-validation forecast MAE. The selected checkpoint was replayed; the H key hash was unchanged.

| Model | Heads × width | Stage 1 best epoch / repair MSE | Stage 2 best epoch | Validation MSE ↓ | Validation MAE ↓ |
|---|---:|---:|---:|---:|---:|
| v12 selected | 1 × 64 | 51 / 106.906964 | 53 | **0.745448** | **0.588212** |
| v23 | 8 × 16 | 99 / 107.063878 | 25 | 0.753421 | 0.590573 |
| v24 | 8 × 8 | 3 / **106.862496** | 98 | 0.771615 | 0.596324 |

v24 improves selected Stage 1 repair MSE over v12 by 0.044467, but its validation forecast MSE is higher by 0.026167 and MAE by 0.008112. The prespecified promotion gate (MAE < 0.588212 and MSE ≤ 0.745448) fails, so no v24 test evaluation was run. The validation result does not support shrinking each of eight heads to eight dimensions for this task.

## Head diagnosis

All eight post-hoc head-suppression ablations increased validation MAE, so no head is clearly redundant by this check. Routes are diverse, although V dominates heads 0–2 (mean V weights 0.682, 0.652, 0.770 on damaged patches). Mean pairwise candidate-attention L1 distances were H 0.233, V 0.093, T 0.093. These observations establish neither a causal reason for the forecast gap nor generalization beyond this validation set. The Stage 1 best epoch was 3; the better repair MSE and poorer forecast together reinforce that repair MSE alone is insufficient for selecting the forecast model.

## Artifacts and verification

- Config: `configs/e23_multi_expert_p4_width64_window_eighthead_v24.yaml`.
- Stage 1/2 manifests: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v24_eighthead_64/stage1/runs/e23-stage1.json` and `stage2/runs/e23-stage2.json`.
- Replay: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v24_eighthead_64/validation_replay.json`.
- Head diagnosis: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v24_eighthead_64/head_validation_diagnostic.json`.

The new width-64 router and config tests passed. The existing E22 dataset CSV was not changed.
