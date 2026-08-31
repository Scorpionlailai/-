# Residual MLP Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstable direct-add recovery path in `ours` with a zero-initialized residual MLP that starts exactly as the PatchTST missing-data baseline and learns bounded corrections according to each patch's missing fraction.

**Architecture:** Keep PatchTST, hard top-k temporal/variable/historical relations, masking, prediction head, and joint training unchanged. Concatenate the three relation messages and the structural prior, pass them through `LayerNorm -> Linear -> GELU -> Dropout -> zero-initialized Linear`, and add the result to the damaged PatchTST patch latent as `(1 - rho) * delta_z`. Preserve the old additive fusion behind a configuration switch for controlled ablations and backward compatibility.

**Tech Stack:** Python 3.10, PyTorch, PyYAML, NumPy, pytest, editable `malr` package, PowerShell experiment commands.

**Spec:** `docs/superpowers/specs/2026-08-31-residual-mlp-recovery-design.md`

## Global Constraints

- Preserve all unrelated dirty-worktree changes. Because several target files were already modified before this feature, use interactive hunk staging and inspect the cached diff before every commit.
- Do not modify or run the S4M integration.
- Keep `ours` initialized from scratch. Only `missing_pretrained` may load the clean PatchTST checkpoint.
- Keep `seq_len=96`, `pred_len=96`, PatchTST hyperparameters, fixed validation/test masks, `start_rate=0.10`, seed `2021`, and standardized-space MSE/MAE unchanged.
- Keep `use_temporal`, `use_variable`, and `use_historical` enabled in the first residual experiment.
- Keep `use_recovery_loss=false` in the first residual experiment so the first comparison isolates the recovery architecture.
- Preserve the old additive recovery behavior whenever the `recovery` block is absent or `fusion: additive` is selected.
- Do not train the 100-epoch experiment during implementation verification; the user will run it on the configured GPU after the code and tests pass.

---

## File Map

| File | Responsibility | Planned change |
|---|---|---|
| `src/malr/config.py` | Typed stage-one configuration and validation | Add `RecoveryConfig`, backward-compatible defaults, parsing, and validation |
| `configs/stage1_sr010_residual_no_recloss.yaml` | First residual-MLP experiment | Add explicit residual settings while retaining ETTh1 `start_rate=0.10` and disabled recovery loss |
| `src/malr/models/recovery.py` | Relation aggregation and latent recovery | Add zero-initialized residual MLP, missing-fraction scaling, exact observed bypass, and diagnostics |
| `src/malr/cli.py` | Model construction from configuration | Pass recovery settings into `MissingAwareRecovery` |
| `tests/unit/test_config.py` | Configuration contract | Cover fallback additive mode, residual parsing, and invalid values |
| `tests/unit/test_recovery.py` | Recovery math and initialization | Cover identity-at-initialization, damage scaling, gradient flow, and detached diagnostics |
| `tests/unit/test_teacher_and_losses.py` | Tiny model helpers and loss behavior | Allow tests to request residual fusion without changing existing additive tests |
| `tests/integration/test_gradient_flow.py` | Joint training gradient contract | Verify forecast loss reaches the residual output layer and, once opened, the earlier MLP layer |
| `tests/integration/test_synthetic_end_to_end.py` | Training/checkpoint acceptance | Exercise residual mode through a complete synthetic training path |
| `tests/integration/test_training_smoke.py` | CLI/model wiring | Verify the new YAML builds the requested residual recovery module |
| `README.md` | Reproducible experiment instructions | Add the exact PatchTST `missing` versus residual-MLP `ours` command sequence |

---

### Task 1: Add the recovery configuration contract

**Files:**

- Modify: `src/malr/config.py`
- Modify: `tests/unit/test_config.py`
- Create: `configs/stage1_sr010_residual_no_recloss.yaml`

**Interfaces:**

```python
@dataclass(frozen=True)
class RecoveryConfig:
    fusion: str = "additive"
    hidden_multiplier: int = 2
    dropout: float = 0.0
    output_init: str = "zero"
    damage_scaling: str = "missing_fraction"


@dataclass(frozen=True)
class Stage1Config:
    version: int
    data: DataConfig
    patch: PatchConfig
    missing: MissingConfig
    prototype: PrototypeConfig
    relations: RelationConfig
    recovery: RecoveryConfig
    model: ModelConfig
    training: TrainingConfig
    ablations: AblationConfig
```

`load_config(path: Path) -> Stage1Config` must use `RecoveryConfig()` when the YAML has no `recovery` mapping. This keeps all existing configurations on the additive path.

- [ ] Add failing tests to `tests/unit/test_config.py`:

```python
def test_legacy_config_defaults_to_additive_recovery() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))
    assert cfg.recovery == RecoveryConfig()


def test_residual_experiment_config_is_explicit() -> None:
    cfg = load_config(Path("configs/stage1_sr010_residual_no_recloss.yaml"))
    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.recovery.fusion == "residual_mlp"
    assert cfg.recovery.hidden_multiplier == 2
    assert cfg.recovery.dropout == pytest.approx(0.0)
    assert cfg.recovery.output_init == "zero"
    assert cfg.recovery.damage_scaling == "missing_fraction"
    assert cfg.ablations.use_recovery_loss is False
    assert cfg.ablations.use_temporal is True
    assert cfg.ablations.use_variable is True
    assert cfg.ablations.use_historical is True
```

Add this parametrized validation test, with imports for `asdict`, `yaml`, and `RecoveryConfig`:

```python
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fusion", "unknown"),
        ("hidden_multiplier", 0),
        ("dropout", -0.1),
        ("dropout", 1.0),
        ("output_init", "default"),
        ("damage_scaling", "none"),
    ],
)
def test_invalid_recovery_config_is_rejected(
    tmp_path: Path, key: str, value: object
) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["recovery"] = asdict(RecoveryConfig())
    raw["recovery"][key] = value
    bad = tmp_path / f"bad_recovery_{key}.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="recovery"):
        load_config(bad)
```

- [ ] Run the focused test and confirm it fails because `RecoveryConfig` and the new YAML do not exist:

```powershell
conda activate malr
pytest tests/unit/test_config.py -q
```

Expected failure: import or attribute failure for `RecoveryConfig`/`cfg.recovery`, followed by a missing-file failure for the new configuration.

- [ ] Add `RecoveryConfig` immediately after `RelationConfig` in `src/malr/config.py`, add the field to `Stage1Config`, and extend `Stage1Config.validate()` with:

```python
if self.recovery.fusion not in {"additive", "residual_mlp"}:
    raise ValueError("recovery.fusion must be additive or residual_mlp")
if self.recovery.hidden_multiplier <= 0:
    raise ValueError("recovery.hidden_multiplier must be positive")
if not 0.0 <= self.recovery.dropout < 1.0:
    raise ValueError("recovery.dropout must be in [0, 1)")
if self.recovery.output_init != "zero":
    raise ValueError("recovery.output_init must be zero")
if self.recovery.damage_scaling != "missing_fraction":
    raise ValueError("recovery.damage_scaling must be missing_fraction")
```

- [ ] Parse the optional block in `load_config` without mutating the raw YAML mapping:

```python
recovery_raw = raw.get("recovery", {})
recovery = RecoveryConfig(**_require_mapping(recovery_raw, "recovery"))
```

Pass `recovery=recovery` when constructing `Stage1Config`.

- [ ] Preserve hashes for legacy additive configurations so existing run manifests remain evaluable. Change `config_sha256` to remove the default recovery block before serialization:

```python
payload_mapping = asdict(config)
if config.recovery == RecoveryConfig():
    payload_mapping.pop("recovery")
payload = json.dumps(
    payload_mapping,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
```

Add a test that computes the same digest from `asdict(cfg)` after removing `recovery`, then asserts equality with `config_sha256(cfg)`. Also assert the residual configuration's digest differs from the legacy additive digest.

- [ ] Create `configs/stage1_sr010_residual_no_recloss.yaml` as a complete copy of `configs/stage1_sr010_no_recloss.yaml` and insert this top-level block after `relations`:

```yaml
recovery:
  fusion: residual_mlp
  hidden_multiplier: 2
  dropout: 0.0
  output_init: zero
  damage_scaling: missing_fraction
```

- [ ] Run the focused tests and confirm they pass:

```powershell
pytest tests/unit/test_config.py -q
```

- [ ] Confirm the complete configuration is readable and prints the intended contract:

```powershell
python -c "from dataclasses import asdict; from pathlib import Path; from malr.config import load_config; c=load_config(Path('configs/stage1_sr010_residual_no_recloss.yaml')); print(asdict(c.recovery), c.missing.start_rate, asdict(c.ablations))"
```

Expected output includes `residual_mlp`, `2`, `0.0`, `zero`, `missing_fraction`, `0.1`, and `use_recovery_loss: False`.

- [ ] Commit only the configuration contract files:

```powershell
git add -p src/malr/config.py tests/unit/test_config.py
git add configs/stage1_sr010_residual_no_recloss.yaml
git diff --cached --check
git diff --cached
git commit -m "feat: configure residual MLP recovery"
```

---

### Task 2: Implement identity-preserving residual recovery

**Files:**

- Modify: `src/malr/models/recovery.py`
- Modify: `tests/unit/test_recovery.py`

**Interfaces:**

```python
class MissingAwareRecovery(nn.Module):
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        d_model: int,
        *,
        temporal_topk: int = 6,
        variable_topk: int = 4,
        historical_relation: HistoricalRelation | None = None,
        use_temporal: bool = True,
        use_variable: bool = True,
        use_historical: bool = True,
        fusion: str = "additive",
        residual_hidden_multiplier: int = 2,
        residual_dropout: float = 0.0,
        residual_output_init: str = "zero",
        damage_scaling: str = "missing_fraction",
    ) -> None

    def fuse(
        self,
        visible: torch.Tensor,
        prior: torch.Tensor,
        recovered: torch.Tensor,
        state: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor
```

For `fusion="residual_mlp"`, `RecoveryOutput.recovered` is the unscaled `delta_z`. For `fusion="additive"`, it keeps its existing meaning.

- [ ] Update existing direct `fuse` calls in `tests/unit/test_recovery.py` to pass a `[batch, variables, patches]` `rho` tensor while retaining the existing additive expected values.

- [ ] Add a failing identity-at-initialization test:

```python
def test_residual_mlp_is_exact_identity_at_initialization() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
        fusion="residual_mlp",
    )
    base = torch.randn(2, 2, 3, 8)
    rho = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.75, 1.0]]]).expand(2, -1, -1)
    state = torch.tensor(
        [[[PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
          [PATCH_MISSING, PATCH_PARTIAL, PATCH_OBSERVED]]]
    ).expand(2, -1, -1)
    values = torch.zeros(2, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    out = module(values, mask, rho, state, base_latents=base)
    assert torch.equal(out.latents, base)
    assert torch.count_nonzero(module.residual_mlp[-1].weight) == 0
    assert torch.count_nonzero(module.residual_mlp[-1].bias) == 0
```

- [ ] Add a failing exact damage-scaling test. Call `fuse` with `recovered=torch.full_like(base, 2.0)` and assert:

```python
expected = base + (1.0 - rho).unsqueeze(-1) * 2.0
assert torch.equal(output[rho == 1.0], base[rho == 1.0])
assert torch.allclose(output, expected)
```

This covers observed patches (`rho=1`), partially missing patches (`0<rho<1`), and fully missing patches (`rho=0`) in one test.

- [ ] Add a failing initialization-gradient test using the same `values`, `mask`, `rho`, `state`, and `base` tensors as the identity test:

```python
loss = module(values, mask, rho, state, base_latents=base).latents.square().mean()
loss.backward()
assert module.residual_mlp[-1].weight.grad is not None
assert module.residual_mlp[-1].weight.grad.abs().sum() > 0
assert module.residual_mlp[1].weight.grad is not None
assert torch.count_nonzero(module.residual_mlp[1].weight.grad) == 0
```

Then set the final projection weight to a small nonzero value, clear gradients, run a second backward pass, and assert `module.residual_mlp[1].weight.grad.abs().sum() > 0`. This proves the intended zero-initialized output gate opens first and then allows learning in the earlier MLP layer.

- [ ] Add a failing diagnostics test that checks the following keys exist in residual mode and every value has `requires_grad is False`:

```python
{
    "residual_delta_norm",
    "residual_applied_norm",
    "residual_partial_mean_norm",
    "residual_missing_mean_norm",
}
```

- [ ] Run the recovery tests and confirm they fail because the constructor has no residual options and `fuse` does not accept `rho`:

```powershell
pytest tests/unit/test_recovery.py -q
```

- [ ] In `MissingAwareRecovery.__init__`, retain the current `self.fusion = nn.Linear(3 * d_model, d_model)` for additive mode and add:

```python
self.fusion_mode = fusion
self.damage_scaling = damage_scaling
self.residual_mlp: nn.Sequential | None = None
if fusion == "residual_mlp":
    hidden_dim = residual_hidden_multiplier * d_model
    self.residual_mlp = nn.Sequential(
        nn.LayerNorm(4 * d_model),
        nn.Linear(4 * d_model, hidden_dim),
        nn.GELU(),
        nn.Dropout(residual_dropout),
        nn.Linear(hidden_dim, d_model),
    )
    nn.init.zeros_(self.residual_mlp[-1].weight)
    nn.init.zeros_(self.residual_mlp[-1].bias)
```

Reject unsupported constructor values with the same rules as configuration validation. Keep constructor validation because tests and downstream callers can instantiate the module without `Stage1Config`.

- [ ] In `forward`, keep relation selection unchanged and branch only at fusion:

```python
relation_messages = torch.cat(
    [temporal_message, variable_message, historical_message], dim=-1
)
if self.fusion_mode == "residual_mlp":
    assert self.residual_mlp is not None
    recovered = self.residual_mlp(torch.cat([relation_messages, prior], dim=-1))
else:
    recovered = self.fusion(relation_messages)

latents = self.fuse(visible, prior, recovered, state, rho)
```

- [ ] Implement the two fusion equations exactly:

```python
if self.fusion_mode == "additive":
    partial = (state == PATCH_PARTIAL).unsqueeze(-1).to(visible.dtype)
    damaged_output = prior + recovered + partial * visible
    return torch.where(
        (state == PATCH_OBSERVED).unsqueeze(-1), visible, damaged_output
    )

damage_strength = (1.0 - rho).unsqueeze(-1).to(visible.dtype)
residual_output = visible + damage_strength * recovered
return torch.where(
    (state == PATCH_OBSERVED).unsqueeze(-1), visible, residual_output
)
```

Validate `rho.shape == state.shape`, finite values, and `0 <= rho <= 1`. The final `torch.where` is required even though `rho=1` should already produce identity; it protects the observed-patch invariant from numerical or caller errors.

- [ ] Add detached residual diagnostics. Use zero scalars when a batch has no partial or fully missing patches:

```python
delta_norm = recovered.norm(dim=-1)
applied_norm = ((1.0 - rho).unsqueeze(-1) * recovered).norm(dim=-1)
partial_mask = state == PATCH_PARTIAL
missing_mask = state == PATCH_MISSING
zero = applied_norm.new_zeros(())
diagnostics["residual_delta_norm"] = delta_norm.detach()
diagnostics["residual_applied_norm"] = applied_norm.detach()
diagnostics["residual_partial_mean_norm"] = (
    applied_norm[partial_mask].mean() if partial_mask.any() else zero
).detach()
diagnostics["residual_missing_mean_norm"] = (
    applied_norm[missing_mask].mean() if missing_mask.any() else zero
).detach()
```

- [ ] Run the focused tests and confirm additive backward compatibility and residual behavior both pass:

```powershell
pytest tests/unit/test_recovery.py -q
```

- [ ] Run relation tests to prove relation-source semantics did not change:

```powershell
pytest tests/unit/test_relations.py tests/unit/test_recovery.py -q
```

- [ ] Commit only the recovery module and its unit tests:

```powershell
git add -p src/malr/models/recovery.py tests/unit/test_recovery.py
git diff --cached --check
git diff --cached
git commit -m "feat: add identity-preserving residual recovery"
```

---

### Task 3: Wire residual recovery into joint training

**Files:**

- Modify: `src/malr/cli.py`
- Modify: `tests/unit/test_teacher_and_losses.py`
- Modify: `tests/integration/test_gradient_flow.py`
- Modify: `tests/integration/test_synthetic_end_to_end.py`
- Modify: `tests/integration/test_training_smoke.py`

**Interfaces:**

`build_experiment_model(config, dataset, variant, *, clean_checkpoint=None, bank=None) -> Stage1Model` must map:

```python
fusion=config.recovery.fusion
residual_hidden_multiplier=config.recovery.hidden_multiplier
residual_dropout=config.recovery.dropout
residual_output_init=config.recovery.output_init
damage_scaling=config.recovery.damage_scaling
```

The `ours` training branch must continue to leave `clean_checkpoint` unloaded. The `missing_pretrained` branch remains unchanged.

- [ ] Change the `_tiny_ours` helper in `tests/unit/test_teacher_and_losses.py` to accept `fusion: str = "additive"` and pass it to `MissingAwareRecovery`. Keep all existing callers additive unless a test explicitly requests residual mode.

- [ ] Add a failing CLI wiring test to `tests/integration/test_training_smoke.py`:

```python
def test_residual_config_builds_residual_recovery_without_clean_initialization() -> None:
    config = load_config(Path("configs/stage1_sr010_residual_no_recloss.yaml"))
    config = replace(
        config,
        ablations=replace(config.ablations, use_historical=False),
    )
    model = build_experiment_model(config, "ETTh1", "ours")
    assert model.recovery.fusion_mode == "residual_mlp"
    assert model.recovery.residual_mlp is not None
    assert torch.count_nonzero(model.recovery.residual_mlp[-1].weight) == 0
```

- [ ] Add a failing integration gradient test using `_tiny_ours(fusion="residual_mlp")`. With `use_recovery_loss=False`, compute only forecast MSE and assert the residual output projection receives a finite, nonzero gradient. Then set the final weight to `0.01`, run a fresh backward pass, and assert the earlier residual linear layer receives a finite, nonzero gradient.

- [ ] Change the synthetic end-to-end test's tiny `ours` recovery to `fusion="residual_mlp"`. Keep the same synthetic data, checkpoint acceptance, and one-epoch limit so this tests integration rather than forecast quality.

- [ ] Run the focused integration tests and confirm failure because the CLI does not yet pass recovery configuration:

```powershell
pytest tests/integration/test_training_smoke.py tests/integration/test_gradient_flow.py tests/integration/test_synthetic_end_to_end.py tests/unit/test_teacher_and_losses.py -q
```

- [ ] Pass all recovery fields in the existing `MissingAwareRecovery` construction inside `build_experiment_model`:

```python
recovery = MissingAwareRecovery(
    c_in=core.c_in,
    patch_num=core.patch_num,
    patch_len=core.patch_len,
    d_model=core.d_model,
    temporal_topk=config.relations.temporal_topk,
    variable_topk=config.relations.variable_topk,
    historical_relation=historical,
    use_temporal=config.ablations.use_temporal,
    use_variable=config.ablations.use_variable,
    use_historical=config.ablations.use_historical,
    fusion=config.recovery.fusion,
    residual_hidden_multiplier=config.recovery.hidden_multiplier,
    residual_dropout=config.recovery.dropout,
    residual_output_init=config.recovery.output_init,
    damage_scaling=config.recovery.damage_scaling,
)
```

Keep the already-existing `historical` local and do not alter model construction outside the recovery arguments.

- [ ] Audit the `ours` initialization branch in `src/malr/cli.py` and retain this invariant:

```python
if variant is ModelVariant.MISSING_PRETRAINED:
    load clean PatchTST weights
elif variant is ModelVariant.OURS:
    do not load clean PatchTST weights
```

Add or retain a test assertion that `ours` reports `clean_initialized is False` in its checkpoint/run metadata if that field is present. Do not add a clean checkpoint dependency to residual `ours` beyond the existing optional prototype-bank construction workflow.

- [ ] Run the focused tests and confirm they pass:

```powershell
pytest tests/integration/test_training_smoke.py tests/integration/test_gradient_flow.py tests/integration/test_synthetic_end_to_end.py tests/unit/test_teacher_and_losses.py -q
```

- [ ] Run checkpoint and leakage guards because model wiring affects experiment dependencies:

```powershell
pytest tests/integration/test_checkpoint_protocol.py tests/integration/test_leakage_guards.py -q
```

- [ ] Commit only the wiring and integration-test files:

```powershell
git add -p src/malr/cli.py tests/unit/test_teacher_and_losses.py tests/integration/test_gradient_flow.py tests/integration/test_synthetic_end_to_end.py tests/integration/test_training_smoke.py
git diff --cached --check
git diff --cached
git commit -m "feat: train residual recovery jointly from scratch"
```

---

### Task 4: Document and verify the first PatchTST comparison

**Files:**

- Modify: `README.md`

- [ ] Add a section named `ETTh1 residual-MLP comparison` that states the controlled comparison:

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

- [ ] Add the exact PowerShell run sequence:

```powershell
conda activate malr
$CFG="configs/stage1_sr010_residual_no_recloss.yaml"
$ART="artifacts/etth1_sr010_residual_no_recloss"

malr make-eval-masks --config $CFG --dataset ETTh1 --artifacts $ART
malr train-clean --config $CFG --dataset ETTh1 --artifacts $ART --device cuda
malr build-bank --config $CFG --dataset ETTh1 --artifacts $ART
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant missing --device cuda
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant ours --device cuda

$missingRun=(Get-ChildItem "$ART/runs/ETTh1-missing-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).BaseName
$oursRun=(Get-ChildItem "$ART/runs/ETTh1-ours-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).BaseName
malr evaluate --config $CFG --dataset ETTh1 --artifacts $ART --run-id $missingRun --device cuda
malr evaluate --config $CFG --dataset ETTh1 --artifacts $ART --run-id $oursRun --device cuda
```

Explain that `train-clean` exists only to produce clean training representations for the historical prototype bank. The residual `ours` model itself starts from scratch.

- [ ] Document the acceptance criteria before results are inspected:

```text
Primary: ours test MSE < missing test MSE on the identical mask SHA-256.
Secondary: ours test MAE <= missing test MAE.
Sanity: both run manifests have the same config hash, mask hash, source CSV hash, seed, final_rate, and metric_space.
Diagnostic: residual applied norms are finite and nonzero after training; observed patches remain exact bypasses.
```

- [ ] Run the complete local suite without launching training:

```powershell
pytest -q
```

Expected result: all MALR tests pass; environment-dependent tests may skip only when their existing skip condition is satisfied.

- [ ] Inspect the diff for accidental changes and confirm the S4M tree is untouched by this implementation:

```powershell
git diff --check
git diff --stat HEAD
git status --short
```

- [ ] Commit only the documentation change:

```powershell
git add -p README.md
git diff --cached --check
git diff --cached
git commit -m "docs: add residual recovery experiment protocol"
```

---

## Post-Training Decision Rule

After the user runs the commands above, compare `missing` and `ours` only when their mask SHA-256, source CSV SHA-256, seed, final rate, metric space, and configuration hash match.

- If residual `ours` improves validation and test MSE, repeat with Weather before adding complexity.
- If validation MSE improves but test MSE does not, inspect overfitting and repeat across additional fixed seeds before changing architecture.
- If validation MSE does not improve, keep the residual MLP fixed and run relation-source ablations in this order: temporal only, variable only, temporal plus variable, then add historical. Do not increase MLP depth first.
- If residual applied norms stay zero, inspect gradient flow and output-gate learning rate before judging the relation messages.
- If residual applied norms explode, reduce residual learning rate or add an explicit residual scale; do not reintroduce direct prior addition.
