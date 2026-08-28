# Missing-Aware Latent Recovery

Stage-1 PyTorch implementation for multivariate forecasting when the input history contains variable-wise random missing blocks. The missing-aware front-end recovers damaged patch latents, then reuses PatchTST's positional encoding, channel-independent Transformer encoder, and flatten forecast head.

The locked first version supports ETTh1 and Weather, `seq_len=pred_len=96`, PatchTST `patch_len=16/stride=8`, S4M-style `variable_missing`, direct hard Top-K relations, and a train-only Matrix Profile motif bank.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

Place the source CSVs at `data/raw/ETTh1.csv` and `data/raw/weather.csv`, or edit only their paths in `configs/stage1.yaml`. Both files must contain a `date` column followed by numeric variables.

## First experiment sequence

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
