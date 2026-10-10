# Official S4M fair comparison

This integration runs the official S4M model, memory bank, and S4 implementation from the pinned upstream checkout. The patch changes only the data-provider dispatch, fair training/evaluation lifecycle, and command-line arguments. It does not edit `model/S4M.py`, `model/Bank.py`, or the protected S4 core.

The first experiment is ETTh1 with the current MALR protocol:

- train `[0,8640)`, validation `[8640,11520)`, test `[11520,14400)`;
- `seq_len=96`, `pred_len=96`, seed `2021`;
- `start_rate=0.10`, `block_len=5`, overlapping blocks allowed;
- dynamic training mask and the existing fixed validation/test masks;
- complete future target supervision and metrics in `train_standardized` space.

Run every command below from the MALR repository root in PowerShell.

## 1. Create an isolated Python environment

Clone the already working `malr` environment so its PyTorch CUDA runtime stays unchanged:

```powershell
conda create -n s4m-fair --clone malr
conda activate s4m-fair
python -m pip install -r integrations/s4m/requirements-s4m-fair.txt
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

These commands do not uninstall, replace, or update the system CUDA driver. The requirements file intentionally contains no `torch`, `nvidia-*`, `cudatoolkit`, or CUDA driver package.

## 2. Prepare and verify the official checkout

```powershell
powershell -ExecutionPolicy Bypass -File integrations/s4m/bootstrap.ps1
python integrations/s4m/verify_upstream.py --root external/S4M --allow-patched
$env:S4M_OFFICIAL_ROOT=(Resolve-Path external/S4M).Path
python -m pytest tests/s4m/test_s4m_patch_contract.py -q
```

The bootstrap pins commit `a718823addd3606e763dfc261174e0135b2535f4`, copies the MALR bridge, applies the minimal patch, and verifies the protected core hashes. It is safe to rerun on the correctly patched checkout.

## 3. Confirm the fixed masks

Use the same MALR artifact directory that produced the PatchTST missing result:

```powershell
$CFG="configs/stage1_sr010_no_recloss.yaml"
$MALR_ART="artifacts/etth1_sr010_no_recloss_scratch"
$S4M_ART="artifacts/s4m_fair_etth1_sr010"

Get-FileHash "$MALR_ART/masks/ETTh1_validation.npz" -Algorithm SHA256
Get-FileHash "$MALR_ART/masks/ETTh1_test.npz" -Algorithm SHA256
```

Do not regenerate these masks after beginning the comparison. The S4M loader validates their dataset, split, seed, start rate, block length, support interval, shape, checksum, and final missing rate before training or evaluation.

## 4. Run a small GPU smoke test

```powershell
$env:WANDB_MODE="disabled"
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --smoke --device cuda
```

Expected final line: `FAIR_SMOKE_OK loss=...`. This runs only one forward/backward batch and does not create a formal test result.

## 5. Run the official-profile validation search

```powershell
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --profile official --device cuda
```

This runs `e_layers in {2,4,8}` with learning rate `0.005`, dropout `0.1`, Adam, and the public S4M memory settings. It selects the lowest fixed-validation MSE, then trains one S4M clean reference with that configuration. It creates:

```text
artifacts/s4m_fair_etth1_sr010/selection-official.json
artifacts/s4m_fair_etth1_sr010/selection-official-clean.json
```

Finalized validation manifests are reused when the command is resumed; they are never overwritten.

To run only the eight-layer model for a full 100 epochs without early termination,
use a fresh output directory and set patience equal to the epoch count:

```powershell
$S4M_ART_8L100="artifacts/s4m_fair_etth1_sr010_el8_e100"
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART_8L100 --profile official --e-layers 8 --train-epochs 100 --patience 100 --device cuda
```

Validation is still measured after every epoch, and the best validation checkpoint
across all 100 epochs is retained. The command trains the missing-data run followed
by its clean reference using the same eight-layer configuration.

## 6. Run the tuned validation search

```powershell
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --profile tuned --device cuda
```

The first stage searches:

```text
e_layers       = 2, 4, 8
learning_rate  = 0.001, 0.003, 0.005
dropout        = 0.1
```

The official-profile candidates are reused. The two best validation candidates are then refined with dropout `0.0` and `0.2`. No test loader is constructed during either search. The winner is written to `selection-tuned.json`.

To inspect all commands without training:

```powershell
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --profile tuned --device cuda --dry-run
```

## 7. Evaluate the three selected S4M runs

Each command is authorized by its selection file before the test loader is constructed:

```powershell
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --evaluate-selected --selection "$S4M_ART/selection-official.json" --device cuda
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --evaluate-selected --selection "$S4M_ART/selection-official-clean.json" --device cuda
python integrations/s4m/search.py --config $CFG --artifacts $MALR_ART --s4m-root external/S4M --output $S4M_ART --evaluate-selected --selection "$S4M_ART/selection-tuned.json" --device cuda
```

Evaluation records are immutable. Do not rerun a finalized evaluation using the same run ID.

## 8. Generate the PatchTST versus S4M table

Set the two PatchTST run IDs to the clean and missing runs in `$MALR_ART`:

```powershell
$PATCH_CLEAN_RUN="ETTh1-clean-<run-id>"
$PATCH_MISSING_RUN="ETTh1-missing-<run-id>"

$officialSelection=Get-Content "$S4M_ART/selection-official.json" | ConvertFrom-Json
$cleanSelection=Get-Content "$S4M_ART/selection-official-clean.json" | ConvertFrom-Json
$tunedSelection=Get-Content "$S4M_ART/selection-tuned.json" | ConvertFrom-Json

python integrations/s4m/compare_results.py `
  --patchtst-manifest "$MALR_ART/runs/$PATCH_MISSING_RUN.json" `
  --patchtst-clean "$MALR_ART/evaluations/$PATCH_CLEAN_RUN.json" `
  --patchtst-missing "$MALR_ART/evaluations/$PATCH_MISSING_RUN.json" `
  --s4m-clean "$S4M_ART/evaluations/$($cleanSelection.selected_run_id).json" `
  --s4m-official "$S4M_ART/evaluations/$($officialSelection.selected_run_id).json" `
  --s4m-tuned "$S4M_ART/evaluations/$($tunedSelection.selected_run_id).json" `
  --tuned-selection "$S4M_ART/selection-tuned.json" `
  --output "$S4M_ART/comparison"
```

The command refuses to write a table unless the source CSV, configuration, test mask SHA-256, final missing rate, split boundaries, metric space, test sample count, and window-index checksum agree. The final table is `artifacts/s4m_fair_etth1_sr010/comparison/comparison.md`.

## Result interpretation

Use `S4M official missing` versus `PatchTST missing` as the published-profile comparison. Use `S4M tuned missing` to show how far S4M can be optimized under this fixed validation protocol. Report the raw standardized MSE/MAE first; clean-to-missing degradation is a secondary diagnostic and is calculated separately for each backbone.
