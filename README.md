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

Each training command prints a `run_id`. Copy the exact `run_id` printed by the final `ours` training command into the single final evaluation command. After the validation and provenance gate passes, that command evaluates the selected `missing` baseline once and then evaluates `ours` once. It always calculates MSE and MAE. DRR-MSE and DRR-MAE are calculated only when compatible clean and missing evaluation records already exist; otherwise they are reported as unavailable. A finalized run ID cannot access the test split a second time.

The four agreed ablations use the booleans under `ablations` in `configs/stage1.yaml`; change only one switch per run. Generated masks, prototype banks, checkpoints, run manifests, and result records live under `artifacts/` and are excluded from Git. Evaluation writes immutable JSON provenance plus `summary.csv` and `summary.md`; unavailable DRR values are shown as `N/A`.

## ETTh1 residual-MLP comparison

This controlled comparison uses the following protocol:

```text
Dataset: ETTh1
Lookback / horizon: 96 / 96
Mask: fixed validation/test variable missing mask
start_rate: 0.10
Compared variants: missing and ours
ours initialization: scratch
Recovery loss: disabled
Metric space: train-standardized
Seed: 2021
```

Run the following from the repository root in the clean `malr` environment:

```powershell
conda activate malr
$CFG="configs/stage1_sr010_residual_no_recloss.yaml"
$ART="artifacts/etth1_sr010_residual_no_recloss"

malr make-eval-masks --config $CFG --dataset ETTh1 --artifacts $ART
malr build-bank --config $CFG --dataset ETTh1 --artifacts $ART
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant missing --device cuda
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant ours --device cuda

$oursRun=(Get-ChildItem "$ART/runs/ETTh1-ours-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).BaseName
malr evaluate --config $CFG --dataset ETTh1 --artifacts $ART --run-id $oursRun --device cuda
```

`malr build-bank` scans standardized training-only data under the leakage guard; it does not consume clean checkpoint representations. The residual `ours` model starts from scratch.

Before inspecting results, use these acceptance criteria:

```text
Primary: ours test MSE < missing test MSE on the identical mask SHA-256.
Secondary: ours test MAE <= missing test MAE.
Validation gate: ours best validation MSE is strictly lower than missing best validation MSE.
Run provenance: both run manifests have identical gated metadata fields.
Diagnostic: residual applied norms are finite and nonzero after training; observed patches remain exact bypasses.
```

Every checkpoint and run manifest records `metadata.dataset`, `metadata.config_hash`, `metadata.source_csv_sha256`, `metadata.seed`, `metadata.validation_mask_sha256`, `metadata.validation_final_rate`, `metadata.train_start`, `metadata.train_end`, `metadata.validation_end`, `metadata.test_end`, `metadata.seq_len`, `metadata.pred_len`, and `metadata.metric_space`. Before either model accesses the test mask or test batches, the gated `ours` evaluation command requires every one of these values to match the latest `missing` run and requires the `ours` best validation MSE to be strictly lower. Missing provenance fields also block test evaluation. After both gated evaluations, compare their evaluation-record `mask_sha256` values to verify the fixed test mask is identical; the test-mask SHA-256 is intentionally an evaluation-record field because it is read only after the pre-test gate passes.

If residual `ours` improves validation and test MSE, repeat with Weather before adding complexity. If validation MSE improves but test MSE does not, inspect overfitting and repeat across additional fixed seeds before changing architecture. If validation MSE does not improve, keep the residual MLP fixed and run relation-source ablations in this order: temporal only, variable only, temporal plus variable, then add historical. Do not increase MLP depth first. If residual applied norms stay zero, inspect gradient flow and output-gate learning rate before judging the relation messages. If residual applied norms explode, reduce residual learning rate or add an explicit residual scale; do not reintroduce direct prior addition.

## 缺失感知边筛选：仅验证集优先

这一轮只比较同一套遮挡下的原始 PatchTST 缺失基线 `missing` 与新筛选模型 `ours`。新模型从头联合训练，采用替换式恢复 MLP；三类候选边先按置信度阈值过滤，再分别限制最多保留 6、4、4 条边。第一轮关闭恢复损失和全局重处理层，避免同时引入过多变量。

在仓库根目录的 `malr` 环境中执行：

```powershell
$CFG="configs/stage1_sr010_replacement_filter_no_recloss.yaml"
$ART="artifacts/etth1_sr010_replacement_filter_no_recloss"

malr make-eval-masks --config $CFG --dataset ETTh1 --artifacts $ART
malr train-clean --config $CFG --dataset ETTh1 --artifacts $ART --device cuda
malr build-bank --config $CFG --dataset ETTh1 --artifacts $ART
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant missing --device cuda
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant ours --device cuda
```

先比较最后两条训练结果中的 `best_val_mse` 与 `best_val_mae`。只有当 `ours` 的两项指标都严格低于 `missing` 时，才允许使用 `ours` 的 `run_id` 执行 `malr evaluate`；否则不读取测试集，先依据 checkpoint 中的三类边保留率、零边率和置信度统计继续调整模型。

## Verification

```powershell
python -m pytest -v
```

The suite covers split boundaries, dynamic/fixed masks, no-leak normalization, patch states, motif construction, hard Top-K selection, frozen teacher gradients, all model variants, checkpoint selection, metrics, and final-test access guards.

## Official S4M fair baseline

The official S4M repository can be evaluated against PatchTST with the exact same ETTh1 split, dynamic training mask, fixed validation/test masks, and standardized metrics. The integration preserves the official S4M/Memory/S4 core and applies only a pinned data-and-lifecycle patch. See [`integrations/s4m/README.md`](integrations/s4m/README.md) for environment setup, validation-only tuning, guarded evaluation, and comparison commands.
