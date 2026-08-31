# Missing-Aware Latent Recovery

Stage-1 PyTorch implementation for multivariate forecasting when the input history contains variable-wise random missing blocks. The missing-aware front-end recovers damaged patch latents, then reuses PatchTST's positional encoding, channel-independent Transformer encoder, and flatten forecast head.

The locked first version supports ETTh1 and Weather, `seq_len=pred_len=96`, PatchTST `patch_len=16/stride=8`, S4M-style `variable_missing`, direct hard Top-K relations, and a train-only Matrix Profile motif bank. The forecasting pipeline follows the official PatchTST preprocessing and dataset-specific scripts except for the explicitly agreed `seq_len=96` override: train-only per-variable `StandardScaler`, sample-level RevIN, end replication, Adam with the `type3` schedule, and standardized-space MSE/MAE. Weather sentinel values such as `-9999` are preserved, matching the official loader.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

Place the source CSVs at `data/raw/ETTh1.csv` and `data/raw/weather.csv`, or edit only their paths in `configs/stage1.yaml`. Both files must contain a `date` column followed by numeric variables.

## Clean reproduction first

Run the following from the repository root. Clean training and evaluation do not require generated missing masks.

```powershell
malr validate-data --config configs/stage1.yaml --dataset ETTh1 --artifacts artifacts/clean_repro_v1
malr train-clean --config configs/stage1.yaml --dataset ETTh1 --artifacts artifacts/clean_repro_v1 --device cuda
malr evaluate --config configs/stage1.yaml --dataset ETTh1 --artifacts artifacts/clean_repro_v1 --run-id <printed-clean-run-id> --device cuda

malr validate-data --config configs/stage1.yaml --dataset Weather --artifacts artifacts/clean_repro_v1
malr train-clean --config configs/stage1.yaml --dataset Weather --artifacts artifacts/clean_repro_v1 --device cuda
malr evaluate --config configs/stage1.yaml --dataset Weather --artifacts artifacts/clean_repro_v1 --run-id <printed-clean-run-id> --device cuda
```

Replace `cuda` with `cpu` when CUDA is unavailable. The separate `artifacts/clean_repro_v1` root keeps these checkpoints and results isolated from earlier raw-scale runs. The reported MSE and MAE are in the train-standardized space only. Because the official PatchTST publication scripts use `seq_len=336`, compare these runs with the official implementation rerun at `seq_len=96`, not directly with the paper's `336 -> 96` table entry.

## Full stage-1 experiment sequence

Run the following from the repository root. The same sequence applies to Weather by replacing `ETTh1` with `Weather`.

```powershell
malr validate-data --config configs/stage1.yaml --dataset ETTh1
malr make-eval-masks --config configs/stage1.yaml --dataset ETTh1
malr train-clean --config configs/stage1.yaml --dataset ETTh1
malr build-bank --config configs/stage1.yaml --dataset ETTh1
malr train --config configs/stage1.yaml --dataset ETTh1 --variant missing
malr train --config configs/stage1.yaml --dataset ETTh1 --variant missing_pretrained
malr train --config configs/stage1.yaml --dataset ETTh1 --variant ours
malr evaluate --config configs/stage1.yaml --dataset ETTh1 --run-id <finalized-run-id>
```

Each training command prints a `run_id`. Copy the exact `run_id` printed by the final `ours` training command into the single final evaluation command. That command first evaluates the matching selected clean and naive checkpoints when their records do not yet exist, then evaluates `ours` and calculates MSE, MAE, DRR-MSE, and DRR-MAE. A finalized run ID cannot access the test split a second time.

The four agreed ablations use the booleans under `ablations` in `configs/stage1.yaml`; change only one switch per run. Generated masks, prototype banks, checkpoints, run manifests, and result records live under `artifacts/` and are excluded from Git. Evaluation writes immutable JSON provenance plus `summary.csv` and `summary.md`; unavailable DRR values are shown as `N/A`.

## Verification

```powershell
python -m pytest -v
```

The suite covers split boundaries, dynamic/fixed masks, no-leak normalization, patch states, motif construction, hard Top-K selection, frozen teacher gradients, all model variants, checkpoint selection, metrics, and final-test access guards.

## Official S4M fair baseline

The official S4M repository can be evaluated against PatchTST with the exact same ETTh1 split, dynamic training mask, fixed validation/test masks, and standardized metrics. The integration preserves the official S4M/Memory/S4 core and applies only a pinned data-and-lifecycle patch. See [`integrations/s4m/README.md`](integrations/s4m/README.md) for environment setup, validation-only tuning, guarded evaluation, and comparison commands.
