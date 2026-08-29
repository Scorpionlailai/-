# S4M Official Fair Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the official S4M model and memory implementation against the exact MALR ETTh1 split and masks, then report both an official-profile result and a validation-tuned result in the same standardized metric space as PatchTST.

**Architecture:** Keep the official repository as an ignored, pinned checkout under `external/S4M`, preserve its model, memory bank, and S4 core unchanged, and apply a small reproducible patch to its data-provider and experiment entry points. A tracked `integrations/s4m` overlay supplies the MALR-compatible dataset, provenance records, search orchestration, and comparison checks without adding S4M to `Stage1Model` or `ModelVariant`.

**Tech Stack:** Python 3.10, PyTorch, NumPy 1.26, pandas, scikit-learn, PyYAML, official S4M commit `a718823addd3606e763dfc261174e0135b2535f4`, pytest, PowerShell, Conda.

**Spec:** `docs/superpowers/specs/2026-08-29-s4m-fair-baseline-design.md`

## Global Constraints

- The official checkout must resolve to commit `a718823addd3606e763dfc261174e0135b2535f4`.
- Do not copy the complete upstream source into this repository; track only the overlay, minimal patch, source metadata, and scripts.
- Do not modify `external/S4M/s4m/model/S4M.py`, `model/Bank.py`, or `model/S4/s4/models/s4/s4.py`.
- Use ETTh1 bounds `[0,8640)`, `[8640,11520)`, and `[11520,14400)` with `seq_len=96` and `pred_len=96`.
- Use `start_rate=0.10`, `block_len=5`, overlap enabled, and seed `2021` from `configs/stage1_sr010_no_recloss.yaml`.
- Training masks are `EpochMaskManager.mask_for_epoch(epoch)`; validation and test masks are the existing MALR `.npz` files.
- Fit standardization on the complete training range only and report metrics in `train_standardized` space.
- Use clean future targets for training, validation, and testing; masked future values never become model inputs.
- Validation chooses all hyperparameters and checkpoints. Test evaluation is an explicit, separate command and runs only for the selected configuration.
- Do not change or reinstall system CUDA. Create `s4m-fair` by cloning the working `malr` Conda environment and add only Python dependencies.
- Do not run the expensive ETTh1 search during implementation; provide tested commands for the user to run on GPU.

## File Map

| Path | Responsibility |
|---|---|
| `.gitignore` | Ignore the official checkout and S4M run artifacts. |
| `integrations/s4m/upstream.json` | Pin repository, commit, and protected official core blob hashes. |
| `integrations/s4m/requirements-s4m-fair.txt` | Add only dependencies missing from the cloned `malr` environment. |
| `integrations/s4m/bootstrap.ps1` | Clone/verify upstream, copy overlay, and apply the minimal patch reproducibly. |
| `integrations/s4m/verify_upstream.py` | Reject wrong commits and modified protected core files. |
| `integrations/s4m/overlay/malr_fair/data.py` | Build official S4M batch tuples from MALR splits and masks. |
| `integrations/s4m/overlay/malr_fair/artifacts.py` | Atomic manifests, checksums, global MSE/MAE, and selection guards. |
| `integrations/s4m/overlay/malr_fair/__init__.py` | Export the overlay interfaces used by patched official files. |
| `integrations/s4m/patches/s4m-malr-fair.patch` | Minimal changes to official `run.py`, `data_factory.py`, and `exp_pretrain1.py`. |
| `integrations/s4m/search.py` | Generate official-profile and tuned validation runs and select the winner. |
| `integrations/s4m/compare_results.py` | Verify result identity and render the PatchTST/S4M comparison. |
| `integrations/s4m/README.md` | Exact Windows/Conda/GPU commands for the user. |
| `tests/s4m/test_s4m_provenance.py` | Pin and protected-file verification. |
| `tests/s4m/test_s4m_fair_data.py` | Split, mask, auxiliary-tensor, and leakage behavior. |
| `tests/s4m/test_s4m_artifacts.py` | Metrics, manifest, and test-selection guard behavior. |
| `tests/s4m/test_s4m_search.py` | Search candidates, deduplication, and validation-only selection. |
| `tests/s4m/test_s4m_compare.py` | Cross-model identity checks and degradation calculations. |

---

### Task 1: Pin and bootstrap the official checkout

**Files:**
- Modify: `.gitignore`
- Create: `integrations/s4m/upstream.json`
- Create: `integrations/s4m/requirements-s4m-fair.txt`
- Create: `integrations/s4m/verify_upstream.py`
- Create: `integrations/s4m/bootstrap.ps1`
- Test: `tests/s4m/test_s4m_provenance.py`

**Interfaces:**
- Produces: `verify_checkout(root: Path, metadata_path: Path, allow_patched: bool) -> dict[str, object]`.
- Produces: a bootstrapped checkout whose Python root is `external/S4M/s4m`.
- Consumes: no earlier task interfaces.

- [ ] **Step 1: Write the failing provenance tests**

```python
from pathlib import Path
import json
import subprocess

import pytest

from integrations.s4m.verify_upstream import verify_checkout


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_verify_checkout_rejects_wrong_commit(tmp_path: Path):
    repo = tmp_path / "S4M"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "sentinel.txt").write_text("wrong", encoding="utf-8")
    _git(repo, "add", "sentinel.txt")
    _git(repo, "commit", "-m", "wrong")
    metadata = tmp_path / "upstream.json"
    metadata.write_text(json.dumps({"commit": "0" * 40, "protected_blobs": {}}))
    with pytest.raises(ValueError, match="commit"):
        verify_checkout(repo, metadata, allow_patched=False)
```

- [ ] **Step 2: Run the provenance test to verify it fails**

Run: `pytest tests/s4m/test_s4m_provenance.py -v`

Expected: FAIL during import because `integrations.s4m.verify_upstream` does not exist.

- [ ] **Step 3: Add exact upstream metadata and verification**

`upstream.json` must contain these protected Git blob identities:

```json
{
  "repository": "https://github.com/WINTERWEEL/S4M.git",
  "commit": "a718823addd3606e763dfc261174e0135b2535f4",
  "protected_blobs": {
    "s4m/model/S4M.py": "9af60e3b5ab5ffd181010124d482bf5a073586ba",
    "s4m/model/Bank.py": "5de63a12e8b7c7423f54599bfa99d71c1afe2106",
    "s4m/model/S4/s4/models/s4/s4.py": "d1a62425886529f40f3d654b6d9531e2104ad6bf"
  },
  "patch_base_blobs": {
    "s4m/run.py": "8512b16f3e0fe491a8ff19877e8ccc7259fd6d55",
    "s4m/data_provider/data_factory.py": "95c57daa8823755df5384c36d6b3f3a3e59b3022",
    "s4m/experiments/exp_pretrain1.py": "43ea309f0a3eabf437b99afcdbb8bf8562a4a3c0"
  }
}
```

The implementation must resolve `git rev-parse HEAD`, run `git hash-object` on protected files, and reject any mismatch. With `allow_patched=True`, dirty files are allowed only under the three patch targets and `s4m/malr_fair/`.

- [ ] **Step 4: Add the Windows bootstrap script and ignore rules**

`bootstrap.ps1` must perform these operations with checked exit codes:

```powershell
git clone https://github.com/WINTERWEEL/S4M.git external/S4M
git -C external/S4M checkout --detach a718823addd3606e763dfc261174e0135b2535f4
python integrations/s4m/verify_upstream.py --root external/S4M
Copy-Item integrations/s4m/overlay/malr_fair external/S4M/s4m/malr_fair -Recurse -Force
git -C external/S4M apply --check ../../integrations/s4m/patches/s4m-malr-fair.patch
git -C external/S4M apply ../../integrations/s4m/patches/s4m-malr-fair.patch
python integrations/s4m/verify_upstream.py --root external/S4M --allow-patched
```

Make the script idempotent: an existing correct patched checkout verifies and exits successfully; an incorrect or partly patched checkout stops with a recovery message and does not reset or delete it.

- [ ] **Step 5: Define environment additions without PyTorch/CUDA packages**

`requirements-s4m-fair.txt` contains pinned Python-only additions:

```text
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.0
einops==0.8.0
opt-einsum==3.3.0
matplotlib==3.9.0
wandb==0.17.0
thop==0.1.1.post2209072238
torchstat==0.0.7
rich==13.8.0
```

Document that the environment is created with `conda create -n s4m-fair --clone malr`; do not include `torch`, `nvidia-*`, `cudatoolkit`, or CUDA drivers in this requirements file.

- [ ] **Step 6: Run the provenance tests**

Run: `pytest tests/s4m/test_s4m_provenance.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the bootstrap unit**

```powershell
git add .gitignore integrations/s4m/upstream.json integrations/s4m/requirements-s4m-fair.txt integrations/s4m/verify_upstream.py integrations/s4m/bootstrap.ps1 tests/s4m/test_s4m_provenance.py
git commit -m "build: pin official S4M checkout"
```

---

### Task 2: Build the exact MALR split and mask data provider

**Files:**
- Create: `integrations/__init__.py`
- Create: `integrations/s4m/__init__.py`
- Create: `integrations/s4m/overlay/malr_fair/__init__.py`
- Create: `integrations/s4m/overlay/malr_fair/data.py`
- Test: `tests/s4m/test_s4m_fair_data.py`

**Interfaces:**
- Consumes: `Stage1Config`, `ForecastDataset`, `EpochMaskManager`, `MaskExpectation`, `load_fixed_mask`, `build_window_index` from MALR.
- Produces: `FairS4MDataset(args: Namespace, flag: Literal["train", "val", "test"])`.
- Produces: `fair_data_provider(args: Namespace, flag: str) -> tuple[FairS4MDataset, DataLoader]`.
- Produces dataset methods: `set_epoch(epoch: int) -> None` and `identity() -> dict[str, object]`.

- [ ] **Step 1: Write failing tests for split and mask identity**

```python
def test_validation_uses_exact_fixed_mask(s4m_args, fixed_masks):
    dataset = FairS4MDataset(s4m_args, "val")
    assert dataset.refs[0].input_start == 8640 - 96
    assert dataset.refs[0].target_start == 8640
    assert dataset.refs[-1].target_end <= 11520
    assert dataset.identity()["mask_sha256"] == fixed_masks["validation_sha256"]


def test_training_mask_matches_malr_epoch_manager(s4m_args):
    dataset = FairS4MDataset(s4m_args, "train")
    dataset.set_epoch(7)
    expected = EpochMaskManager(8640, 7, 0.10, 5, 2021).mask_for_epoch(7)
    assert np.array_equal(dataset.observed_timeline, expected)
```

Use the real constructor keyword names from `EpochMaskManager` in the committed test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/s4m/test_s4m_fair_data.py -v`

Expected: FAIL because `FairS4MDataset` is not defined.

- [ ] **Step 3: Implement train-only standardization and window construction**

The provider must use this data flow:

```python
config = load_config(Path(args.malr_config))
series = load_clean_csv(dataset_cfg.path, dataset_cfg.n_vars, ...)
protocol = protocol_for(args.malr_dataset, len(series.values))
stats = fit_train_variable_stats(
    torch.from_numpy(series.values[protocol.train.start:protocol.train.end]),
    eps=0.0,
)
standardized = standardize_values(torch.from_numpy(series.values), stats).numpy()
refs = build_window_index(protocol, split_name, config.data.seq_len, config.data.pred_len)
```

For validation/test, preserve the first `floor(len(refs) / 128) * 128` references because the existing PatchTST evaluation used `drop_last=True`, batch size 128, and non-shuffled refs. S4M may evaluate those canonical refs in smaller batches without dropping more samples.

- [ ] **Step 4: Implement dynamic and fixed mask selection**

Training uses:

```python
self.mask_manager = EpochMaskManager(
    length=protocol.train.end,
    n_vars=series.values.shape[1],
    start_rate=config.missing.start_rate,
    block_len=config.missing.block_len,
    base_seed=config.training.seed,
)
self.observed_timeline = self.mask_manager.mask_for_epoch(epoch)
```

Validation/test must build `MaskExpectation` over `[bounds.start - seq_len, bounds.end)`, load `<artifacts>/masks/ETTh1_validation.npz` or `ETTh1_test.npz`, and retain both the stored metadata checksum and the whole-file SHA-256 used by MALR result records. `fair_variant=clean` replaces the observed timeline with all `True` while keeping the same refs.

- [ ] **Step 5: Implement the official 10-tensor sample contract**

For each `WindowRef`, return NumPy arrays in this exact order:

```python
seq_x_clean = standardized[ref.input_start:ref.input_end].astype(np.float32)
seq_x_mask = observed_slice.astype(np.float32)
seq_x = np.where(seq_x_mask == 1.0, seq_x_clean, 0.0).astype(np.float32)
seq_true_y = standardized[ref.target_start:ref.target_end].astype(np.float32)
idx_grid = np.arange(seq_x.shape[0])[:, None]
max_pos = np.argmax(seq_x, axis=0)
min_pos = np.argmin(seq_x, axis=0)
max_idx = np.abs(max_pos[None, :] - idx_grid).astype(np.float32)
min_idx = np.abs(min_pos[None, :] - idx_grid).astype(np.float32)
max_value = np.broadcast_to(np.max(seq_x, axis=0), seq_x.shape).astype(np.float32)
min_value = np.broadcast_to(np.min(seq_x, axis=0), seq_x.shape).astype(np.float32)
return (
    seq_x, seq_x_mask, seq_true_y, np.ones_like(seq_true_y),
    seq_x_clean, seq_true_y, max_idx, min_idx, max_value, min_value,
)
```

This preserves the official `Dataset_Custom4` auxiliary formulas while preventing masked clean values from entering them.

- [ ] **Step 6: Add leakage and deterministic behavior tests**

Tests must mutate clean values only at masked input coordinates and assert that `seq_x`, `max_idx`, `min_idx`, `max_value`, and `min_value` do not change. A second test must assert that `set_epoch(3)` is repeatable and differs from `set_epoch(4)`.

- [ ] **Step 7: Run the data-provider tests**

Run: `pytest tests/s4m/test_s4m_fair_data.py tests/unit/test_masks.py tests/unit/test_splits_and_windows.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the fair data provider**

```powershell
git add integrations/__init__.py integrations/s4m/__init__.py integrations/s4m/overlay/malr_fair tests/s4m/test_s4m_fair_data.py
git commit -m "feat: feed MALR masks to official S4M"
```

---

### Task 3: Add traceable S4M artifacts and selection guards

**Files:**
- Create: `integrations/s4m/overlay/malr_fair/artifacts.py`
- Modify: `integrations/s4m/overlay/malr_fair/__init__.py`
- Test: `tests/s4m/test_s4m_artifacts.py`

**Interfaces:**
- Consumes: `FairS4MDataset.identity()`.
- Produces: `MetricAccumulator.update(prediction: Tensor, target: Tensor) -> None`.
- Produces: `MetricAccumulator.compute() -> dict[str, float]`.
- Produces: `write_training_manifest(path: Path, payload: dict[str, object]) -> None`.
- Produces: `authorize_test(selection_path: Path, run_id: str, checkpoint: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing global-metric and guard tests**

```python
def test_metrics_are_global_not_mean_of_batch_means():
    acc = MetricAccumulator()
    acc.update(torch.tensor([0.0]), torch.tensor([2.0]))
    acc.update(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0]))
    assert acc.compute() == {"mse": 1.75, "mae": 1.25}


def test_test_guard_rejects_non_selected_run(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected_run_id": "best"}))
    with pytest.raises(ValueError, match="selected"):
        authorize_test(selection, "not-best", tmp_path / "checkpoint.pth")
```

- [ ] **Step 2: Run the artifact tests to verify they fail**

Run: `pytest tests/s4m/test_s4m_artifacts.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Implement element-weighted MSE/MAE**

Accumulate `sum_squared_error`, `sum_absolute_error`, and `element_count` as Python float/int values after detaching tensors. Reject non-finite tensors, shape mismatches, and zero elements.

- [ ] **Step 4: Implement atomic manifests and the test guard**

Training manifests must include `status="validation_finalized"`, validation MSE/MAE, best epoch, checkpoint SHA-256, dataset identity, source CSV hash, config hash, mask hashes, upstream commit, protected core hashes, environment versions, all S4M arguments, parameter count, and training seconds. `authorize_test` must verify the selected run ID, checkpoint path, checkpoint SHA-256, and `status` before test data is constructed.

- [ ] **Step 5: Run the artifact tests**

Run: `pytest tests/s4m/test_s4m_artifacts.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the artifact protocol**

```powershell
git add integrations/s4m/overlay/malr_fair tests/s4m/test_s4m_artifacts.py
git commit -m "feat: record guarded S4M experiments"
```

---

### Task 4: Patch the official data and experiment entry points

**Files:**
- Create: `integrations/s4m/patches/s4m-malr-fair.patch`
- Modify: `integrations/s4m/bootstrap.ps1`
- Test: `tests/s4m/test_s4m_official_smoke.py`
- External patch target: `external/S4M/s4m/data_provider/data_factory.py`
- External patch target: `external/S4M/s4m/experiments/exp_pretrain1.py`
- External patch target: `external/S4M/s4m/run.py`

**Interfaces:**
- Consumes: `fair_data_provider`, `MetricAccumulator`, manifest writers, and test guard from Tasks 2–3.
- Produces: official commands with `--fair_protocol 1 --fair_mode train|evaluate`.
- Preserves: official `S4M.Model.forward(seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value)`.

- [ ] **Step 1: Write an opt-in external smoke test**

```python
@pytest.mark.skipif(not os.getenv("S4M_OFFICIAL_ROOT"), reason="official checkout not bootstrapped")
def test_official_s4m_fair_forward(tmp_path):
    command = [
        sys.executable, str(Path(os.environ["S4M_OFFICIAL_ROOT"]) / "s4m" / "run.py"),
        "--model", "S4M", "--fair_protocol", "1", "--fair_mode", "smoke",
        "--malr_config", "configs/stage1_sr010_no_recloss.yaml",
        "--malr_dataset", "ETTh1", "--malr_artifacts", str(tmp_path),
        "--seq_len", "96", "--pred_len", "96", "--batch_size", "2",
        "--e_layers", "2", "--d_model", "32", "--d_ff", "32",
        "--fair_device", "cpu", "--wandb_mode", "disabled",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert "FAIR_SMOKE_OK" in completed.stdout
```

- [ ] **Step 2: Patch `data_factory.py` only at provider dispatch**

Add an early branch to the official function:

```python
def data_provider(args, flag):
    if getattr(args, "fair_protocol", 0):
        from malr_fair import fair_data_provider
        return fair_data_provider(args, flag)
    # all original official lines continue unchanged below
```

- [ ] **Step 3: Patch `exp_pretrain1.py` for the fair loss and lifecycle**

Under `fair_protocol` only:

- `_select_optimizer` returns Adam or official SGD according to `--fair_optimizer`.
- `train` does not construct `test_data`/`test_loader`.
- At the start of each epoch it calls `train_data.set_epoch(epoch)`.
- Training and validation loss compare `outputs[:, -pred_len:, :]` with `seq_true_y[:, -pred_len:, :]` across every element.
- Validation uses `MetricAccumulator` instead of averaging batch means.
- Early stopping uses `delta=0.0` and saves the official model plus memory tensors.
- Training writes a validation-finalized manifest and never invokes test.
- Evaluation calls `authorize_test` before `_get_data(flag="test")`, loads one explicit checkpoint, computes global standardized MSE/MAE, and writes one immutable test result.

Leave the non-fair branches byte-for-byte equivalent to upstream behavior.

- [ ] **Step 4: Patch `run.py` with fair arguments and explicit modes**

Add arguments with these exact defaults:

```python
parser.add_argument("--fair_protocol", type=int, default=0)
parser.add_argument("--fair_mode", choices=["smoke", "train", "evaluate"], default="train")
parser.add_argument("--fair_variant", choices=["clean", "missing"], default="missing")
parser.add_argument("--fair_optimizer", choices=["adam", "sgd"], default="adam")
parser.add_argument("--fair_device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--malr_config", type=str, default="")
parser.add_argument("--malr_dataset", choices=["ETTh1", "Weather"], default="ETTh1")
parser.add_argument("--malr_artifacts", type=str, default="")
parser.add_argument("--fair_run_dir", type=str, default="artifacts/s4m_fair")
parser.add_argument("--fair_run_id", type=str, default="")
parser.add_argument("--fair_selection", type=str, default="")
parser.add_argument("--fair_checkpoint", type=str, default="")
parser.add_argument("--fair_eval_reference_batch", type=int, default=128)
parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
```

The fair control flow must call only one of `smoke`, `train`, or `evaluate`; it must bypass the official unconditional post-training `exp.test(...)` call.
It must derive `args.use_gpu` and `args.device` from `--fair_device` before constructing the experiment, avoiding the upstream `type=bool` parsing behavior for `--use_gpu`.

- [ ] **Step 5: Capture the minimal patch and verify protected files**

The patch may contain only:

```text
s4m/data_provider/data_factory.py
s4m/experiments/exp_pretrain1.py
s4m/run.py
```

Run: `python integrations/s4m/verify_upstream.py --root external/S4M --allow-patched`

Expected: PASS and all three protected model blob hashes equal `upstream.json`.

- [ ] **Step 6: Run the official CPU smoke test**

```powershell
$env:S4M_OFFICIAL_ROOT=(Resolve-Path external/S4M)
pytest tests/s4m/test_s4m_official_smoke.py -v
```

Expected: PASS with `FAIR_SMOKE_OK`; no ETTh1 training is run.

- [ ] **Step 7: Commit the official integration patch**

```powershell
git add integrations/s4m/patches/s4m-malr-fair.patch integrations/s4m/bootstrap.ps1 tests/s4m/test_s4m_official_smoke.py
git commit -m "feat: patch official S4M for fair evaluation"
```

---

### Task 5: Implement validation-only S4M optimization

**Files:**
- Create: `integrations/s4m/search.py`
- Test: `tests/s4m/test_s4m_search.py`

**Interfaces:**
- Consumes: bootstrapped official `run.py` and validation manifests.
- Produces: `SearchCandidate(run_id: str, e_layers: int, learning_rate: float, dropout: float, profile: str)`.
- Produces: `official_candidates() -> list[SearchCandidate]`.
- Produces: `tuned_stage1_candidates() -> list[SearchCandidate]`.
- Produces: `select_best(manifests: Sequence[Path], output: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing candidate and selection tests**

```python
def test_official_profile_searches_paper_layer_choices():
    candidates = official_candidates()
    assert {(c.e_layers, c.learning_rate, c.dropout) for c in candidates} == {
        (2, 0.005, 0.1), (4, 0.005, 0.1), (8, 0.005, 0.1)
    }


def test_selection_uses_validation_mse_only(tmp_path):
    manifests = write_manifests(tmp_path, [
        {"run_id": "a", "validation_mse": 0.8, "test_mse": 0.1},
        {"run_id": "b", "validation_mse": 0.7, "test_mse": 9.0},
    ])
    assert select_best(manifests, tmp_path / "selection.json")["selected_run_id"] == "b"
```

- [ ] **Step 2: Run the search tests to verify they fail**

Run: `pytest tests/s4m/test_s4m_search.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Implement the official and first-stage tuned grids**

Generate unique candidates for:

```python
official = product(e_layers=[2, 4, 8], learning_rate=[0.005], dropout=[0.1])
tuned_stage1 = product(
    e_layers=[2, 4, 8],
    learning_rate=[0.001, 0.003, 0.005],
    dropout=[0.1],
)
```

All commands fix `d_model=256`, `d_ff=256`, `batch_size=16`, `train_epochs=15`, `patience=3`, seed 2021, Adam, and the memory parameters in the design spec. `--dry-run` prints commands without launching subprocesses.

- [ ] **Step 4: Implement the dropout refinement and final selection**

After stage 1, take the two smallest validation-MSE candidates and generate only the unseen combinations with `dropout in [0.0, 0.1, 0.2]`. Sort final manifests by `(validation_mse, validation_mae, run_id)`. Write `selection.json` with every candidate's validation metrics, the selected run/checkpoint hashes, search space, and `test_metrics_seen=false`.

- [ ] **Step 5: Add a clean S4M command**

Generate one `fair_variant=clean` run using the winning official-profile architecture and optimizer settings. This clean run is reported as a backbone reference and is not part of missing-configuration selection.

- [ ] **Step 6: Run search unit tests and inspect dry-run commands**

```powershell
pytest tests/s4m/test_s4m_search.py -v
python integrations/s4m/search.py --config configs/stage1_sr010_no_recloss.yaml --artifacts artifacts/etth1_sr010_no_recloss_scratch --s4m-root external/S4M --output artifacts/s4m_fair_etth1_sr010 --dry-run
```

Expected: tests PASS; dry-run prints three official candidates, nine stage-1 candidates with overlap deduplicated, and no evaluate command.

- [ ] **Step 7: Commit the search orchestration**

```powershell
git add integrations/s4m/search.py tests/s4m/test_s4m_search.py
git commit -m "feat: tune S4M on fixed validation masks"
```

---

### Task 6: Produce a guarded cross-model comparison

**Files:**
- Create: `integrations/s4m/compare_results.py`
- Test: `tests/s4m/test_s4m_compare.py`

**Interfaces:**
- Consumes: PatchTST clean/missing evaluation JSON, S4M clean/official/tuned evaluation JSON, and `selection.json`.
- Produces: `compare_results(...) -> dict[str, object]`.
- Produces: `comparison.json`, `comparison.csv`, and `comparison.md`.

- [ ] **Step 1: Write failing identity and degradation tests**

```python
def test_comparison_rejects_different_test_mask(records):
    records["s4m_missing"]["mask_sha256"] = "different"
    with pytest.raises(ValueError, match="mask_sha256"):
        compare_results(**records)


def test_absolute_and_relative_degradation():
    result = degradation(clean=0.4, missing=0.5)
    assert result == {"absolute": pytest.approx(0.1), "relative_percent": pytest.approx(25.0)}
```

- [ ] **Step 2: Run comparison tests to verify they fail**

Run: `pytest tests/s4m/test_s4m_compare.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Enforce comparison identity**

Before writing output, require equality of dataset, source CSV SHA-256, split boundaries, `seq_len`, `pred_len`, `metric_space`, test mask SHA-256, mask `final_rate`, canonical window indices checksum, and canonical sample count. Require that the tuned S4M run ID and checkpoint SHA-256 equal `selection.json`.

- [ ] **Step 4: Render results without DRR ambiguity**

Write raw MSE/MAE plus clean-to-missing absolute and relative degradation. Do not reuse MALR's DRR for cross-backbone comparison. Label rows `PatchTST clean`, `PatchTST missing`, `S4M official clean`, `S4M official missing`, and `S4M tuned missing`.

- [ ] **Step 5: Run comparison tests**

Run: `pytest tests/s4m/test_s4m_compare.py -v`

Expected: PASS.

- [ ] **Step 6: Commit comparison tooling**

```powershell
git add integrations/s4m/compare_results.py tests/s4m/test_s4m_compare.py
git commit -m "feat: compare S4M and PatchTST fairly"
```

---

### Task 7: Document the exact user-run workflow

**Files:**
- Create: `integrations/s4m/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: bootstrap, search, official patched CLI, and comparison tooling.
- Produces: copy-paste PowerShell commands; no new Python interface.

- [ ] **Step 1: Document environment setup without CUDA changes**

Include exactly:

```powershell
conda create -n s4m-fair --clone malr
conda activate s4m-fair
python -m pip install -r integrations/s4m/requirements-s4m-fair.txt
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

State that these commands clone the existing working PyTorch environment and do not install or alter the system CUDA driver.

- [ ] **Step 2: Document bootstrap and mask preflight**

```powershell
powershell -ExecutionPolicy Bypass -File integrations/s4m/bootstrap.ps1
malr make-eval-masks --config configs/stage1_sr010_no_recloss.yaml --dataset ETTh1 --artifacts artifacts/etth1_sr010_no_recloss_scratch
python integrations/s4m/verify_upstream.py --root external/S4M --allow-patched
```

Explain that an existing fixed mask is validated and must not be regenerated after PatchTST/S4M comparison runs begin.

- [ ] **Step 3: Document smoke, official-profile, and tuned runs**

```powershell
$env:WANDB_MODE="disabled"
python integrations/s4m/search.py --config configs/stage1_sr010_no_recloss.yaml --artifacts artifacts/etth1_sr010_no_recloss_scratch --s4m-root external/S4M --output artifacts/s4m_fair_etth1_sr010 --smoke --device cuda
python integrations/s4m/search.py --config configs/stage1_sr010_no_recloss.yaml --artifacts artifacts/etth1_sr010_no_recloss_scratch --s4m-root external/S4M --output artifacts/s4m_fair_etth1_sr010 --profile official --device cuda
python integrations/s4m/search.py --config configs/stage1_sr010_no_recloss.yaml --artifacts artifacts/etth1_sr010_no_recloss_scratch --s4m-root external/S4M --output artifacts/s4m_fair_etth1_sr010 --profile tuned --device cuda
```

The README must explain expected run counts and how to resume without overwriting finalized manifests.

- [ ] **Step 4: Document explicit final evaluation**

```powershell
python integrations/s4m/search.py --config configs/stage1_sr010_no_recloss.yaml --artifacts artifacts/etth1_sr010_no_recloss_scratch --s4m-root external/S4M --output artifacts/s4m_fair_etth1_sr010 --evaluate-selected --device cuda
```

State that this is the first command allowed to construct the S4M test loader.

- [ ] **Step 5: Document comparison generation**

Provide one command with explicit paths to the existing PatchTST clean/missing results and the three S4M results. The script must print the generated `.md` path.

- [ ] **Step 6: Commit documentation**

```powershell
git add integrations/s4m/README.md README.md
git commit -m "docs: add official S4M comparison workflow"
```

---

### Task 8: Verify the integration without running the expensive search

**Files:**
- Modify only files required to fix failures found by the commands below.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: a verified handoff for user-run GPU experiments.

- [ ] **Step 1: Run all local tests**

Run: `pytest -q`

Expected: all existing MALR tests and all non-external S4M tests PASS; the official smoke test is SKIPPED if `S4M_OFFICIAL_ROOT` is unset.

- [ ] **Step 2: Verify patch reproducibility in the real checkout**

```powershell
powershell -ExecutionPolicy Bypass -File integrations/s4m/bootstrap.ps1
$env:S4M_OFFICIAL_ROOT=(Resolve-Path external/S4M)
pytest tests/s4m/test_s4m_official_smoke.py -v
```

Expected: bootstrap verifies the pinned commit and patch; smoke test PASS.

- [ ] **Step 3: Verify protected official model files remain unchanged**

Run: `python integrations/s4m/verify_upstream.py --root external/S4M --allow-patched`

Expected: the three protected blob hashes match `upstream.json`; dirty paths are limited to allowed patch targets and `s4m/malr_fair/`.

- [ ] **Step 4: Validate dry-run search and no test access**

Run: `python integrations/s4m/search.py --config configs/stage1_sr010_no_recloss.yaml --artifacts artifacts/etth1_sr010_no_recloss_scratch --s4m-root external/S4M --output artifacts/s4m_fair_etth1_sr010 --profile tuned --dry-run`

Expected: commands contain `--fair_mode train` only; no `--fair_mode evaluate` and no test result is created.

- [ ] **Step 5: Inspect the final diff**

Run: `git diff --check`

Expected: no whitespace errors. Confirm no existing PatchTST/MALR model file is included in the S4M integration commits.

- [ ] **Step 6: Commit verification-only fixes if needed**

```powershell
git add integrations/s4m tests/s4m README.md .gitignore
git commit -m "test: verify official S4M fair baseline"
```
