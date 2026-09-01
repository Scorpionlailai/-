# Post-Recovery Global Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add replacement latent fusion and an optional identity-initialized global recovery refiner before the unchanged PatchTST encoder.

**Architecture:** `replacement_mlp` emits an absolute recovered latent and blends it with the masked-patch embedding by observation ratio. An optional one-layer global attention refiner jointly reads every variable-patch node while applying its gated residual only to damaged nodes.

**Tech Stack:** Python 3.10, PyTorch, dataclasses, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-01-post-recovery-global-refinement-design.md`

## Global Constraints

- Preserve `additive` and `residual_mlp` behavior and checkpoint compatibility.
- Leave PatchTST encoder and forecast head unchanged.
- Keep ETTh1 `seq_len=96`, `pred_len=96`, `start_rate=0.10`, seed 2021, and existing fixed-mask protocol.
- Keep recovery loss disabled in both new experiment configs.
- Never use recovered nodes as first-pass relation sources.
- Do not merge the feature branch.

---

### Task 1: Replacement fusion contract

**Files:**
- Modify: `tests/unit/test_recovery.py`
- Modify: `src/malr/models/recovery.py`

**Interfaces:**
- Consumes: existing `MissingAwareRecovery.forward(...)` tensors and relation messages.
- Produces: `fusion="replacement_mlp"` and `z_mix = rho * visible + (1 - rho) * candidate`.

- [ ] **Step 1: Write failing replacement tests**

```python
def test_replacement_mlp_interpolates_base_and_candidate_by_observation_ratio():
    module = MissingAwareRecovery(..., fusion="replacement_mlp")
    output = module.fuse(base, prior, candidate, state, rho)
    torch.testing.assert_close(output, rho.unsqueeze(-1) * base + (1-rho).unsqueeze(-1) * candidate)

def test_fully_missing_replacement_does_not_retain_base_latent():
    ...
    torch.testing.assert_close(first, second)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/test_recovery.py -k replacement -q`

Expected: FAIL because `replacement_mlp` is rejected.

- [ ] **Step 3: Implement minimal replacement fusion**

Add the fusion option, create a five-input-width replacement MLP, and implement
the observation-ratio blend while preserving the existing branches unchanged.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/unit/test_recovery.py -q`

Expected: all recovery tests pass.

- [ ] **Step 5: Commit checkpoint**

```powershell
git add src/malr/models/recovery.py tests/unit/test_recovery.py
git commit -m "feat: add replacement latent fusion"
```

### Task 2: Global recovery refiner

**Files:**
- Create: `src/malr/models/refinement.py`
- Modify: `src/malr/models/recovery.py`
- Modify: `src/malr/models/__init__.py`
- Create: `tests/unit/test_refinement.py`

**Interfaces:**
- Produces: `GlobalLatentRefiner(d_model, n_heads, ff_multiplier, dropout, output_init)`.
- Consumes: `latents`, `prior`, and `rho`, each with recovery node geometry.
- Returns: refined latents plus detached delta diagnostics.

- [ ] **Step 1: Write failing refiner tests**

```python
def test_global_refiner_is_identity_at_zero_initialization(): ...
def test_global_refiner_never_changes_fully_observed_nodes(): ...
def test_global_refiner_jointly_uses_other_nodes_after_gate_opens(): ...
def test_global_refiner_output_gate_receives_first_step_gradient(): ...
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/test_refinement.py -q`

Expected: collection fails because `malr.models.refinement` does not exist.

- [ ] **Step 3: Implement the refiner**

Implement one pre-norm `nn.MultiheadAttention(batch_first=True)` block, one
pre-norm FFN, a zero-initialized output projection, flatten/unflatten of the
variable-patch axes, and `(1-rho)` output gating.

- [ ] **Step 4: Integrate after replacement fusion**

Construct the refiner only when enabled, call it after `fuse`, and expose
detached global delta and applied-delta norms through recovery diagnostics.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/unit/test_refinement.py tests/unit/test_recovery.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit checkpoint**

```powershell
git add src/malr/models/refinement.py src/malr/models/recovery.py src/malr/models/__init__.py tests/unit/test_refinement.py
git commit -m "feat: add post-recovery global refiner"
```

### Task 3: Configuration and model construction

**Files:**
- Modify: `src/malr/config.py`
- Modify: `src/malr/cli.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/integration/test_gradient_flow.py`
- Create: `configs/stage1_sr010_replacement_no_recloss.yaml`
- Create: `configs/stage1_sr010_replacement_global_no_recloss.yaml`

**Interfaces:**
- Extends `RecoveryConfig` with disabled-by-default global refiner settings.
- Passes refiner settings from `build_experiment_model` into `MissingAwareRecovery`.

- [ ] **Step 1: Write failing config and construction tests**

```python
def test_replacement_experiment_configs_are_explicit(): ...
def test_invalid_refiner_heads_are_rejected(): ...
def test_ours_builds_enabled_global_refiner(): ...
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/unit/test_config.py tests/integration/test_gradient_flow.py -q`

Expected: FAIL because the new fields, fusion mode, and configs are absent.

- [ ] **Step 3: Implement config parsing and validation**

Add `refiner_enabled=False`, `refiner_heads=4`, `refiner_ff_multiplier=2`,
`refiner_dropout=0.0`, and `refiner_output_init="zero"`. Validate positive
non-boolean integer dimensions, dropout range, zero initialization, and
`d_model % refiner_heads == 0` when enabled.

- [ ] **Step 4: Wire model construction and add configs**

Pass all new fields through `build_experiment_model`. Derive both configs from
the existing residual ETTh1 config while changing only fusion/refiner fields.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/unit/test_config.py tests/integration/test_gradient_flow.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit checkpoint**

```powershell
git add src/malr/config.py src/malr/cli.py tests/unit/test_config.py tests/integration/test_gradient_flow.py configs/stage1_sr010_replacement_no_recloss.yaml configs/stage1_sr010_replacement_global_no_recloss.yaml
git commit -m "feat: configure replacement refinement experiments"
```

### Task 4: Full regression and user protocol

**Files:**
- Modify if needed: `README.md`

**Interfaces:**
- Produces reproducible PowerShell commands for the two ETTh1 validation-only experiments.

- [ ] **Step 1: Run focused model tests**

Run: `python -m pytest tests/unit/test_recovery.py tests/unit/test_refinement.py tests/unit/test_config.py tests/integration/test_gradient_flow.py tests/integration/test_training_smoke.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Run complete regression suite**

Run: `python -m pytest -q`

Expected: all MALR tests pass; the optional unbootstrapped S4M contract may skip.

- [ ] **Step 3: Inspect the diff**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors and only planned files plus pre-existing untracked configs.

- [ ] **Step 4: Provide the experiment commands**

Give separate artifact directories for replacement-only and
replacement-plus-refiner. Reuse one fixed mask set by copying or explicitly
pointing both runs to the same generated mask artifacts before training. Do not
evaluate the official test split until validation selects a winner.
