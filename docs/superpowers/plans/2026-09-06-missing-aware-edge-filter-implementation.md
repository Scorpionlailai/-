# Missing-Aware Edge Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace forced fixed hard Top-K recovery edges with learned, bilateral-context-aware hard filtering while leaving PatchTST unchanged.

**Architecture:** A new filtering module builds one target query from visible patch evidence and bilateral same-variable context, scores temporal, variable, and historical candidates with relation-specific MLPs, and applies thresholded hard gates with straight-through gradients. Existing relation modules still generate candidate keys, values, base scores, and eligibility; replacement fusion consumes only filtered messages and falls back safely when every edge is rejected.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, NumPy, PyYAML, pytest, editable `malr` package.

**Spec:** `docs/superpowers/specs/2026-09-06-missing-aware-edge-filter-design.md`

## Global Constraints

- Keep ETTh1 `seq_len=96`, `pred_len=96`, patch length `16`, stride `8`, and end-replication padding.
- Keep variable missingness with `start_rate=0.10`, `block_len=5`, overlapping blocks, and seed `2021`.
- Keep `replacement_mlp`, disable the global refiner, and keep `use_recovery_loss: false` in the new experiment.
- Train `ours` and PatchTST jointly from scratch; never initialize `ours` from the clean checkpoint.
- Preserve the existing clean, missing, missing-pretrained, and legacy hard-Top-K paths.
- Recovered nodes cannot become relation sources.
- Use training-only prototypes and masked look-back information; never expose a masked clean value to the filter query.
- Forward and evaluation gates are binary. Straight-through probabilities exist only to supply training gradients.
- Top-K values `6/4/4` are upper bounds, so every relation may retain zero edges.
- Do not inspect the official test result until the new configuration is frozen from validation MSE and MAE.
- Do not modify `src/malr/models/patchtst.py` or vendored PatchTST files.

---

## File Structure

- Create `src/malr/models/filtering.py`: bilateral context construction, deterministic hard threshold/cap selection, straight-through edge confidence heads, aggregation weights, and filter result dataclasses.
- Modify `src/malr/config.py`: filter configuration contract, validation, nested parsing, and legacy hash compatibility.
- Modify `src/malr/models/relations.py`: expose candidate tensors and route temporal, variable, and historical candidates through an optional edge filter while preserving legacy behavior.
- Modify `src/malr/models/recovery.py`: build the filter query, collect relation filter losses and diagnostics, and preserve replacement fallback.
- Modify `src/malr/models/stage1.py`: pass filter retention loss into the stage-1 objective.
- Modify `src/malr/models/__init__.py`: export the new public filtering types used by tests and the composition root.
- Modify `src/malr/training/losses.py`: add the weighted filter retention term.
- Modify `src/malr/training/trainer.py`: carry `lambda_keep`, aggregate validation filter diagnostics, and store diagnostics for the selected checkpoint.
- Modify `src/malr/cli.py`: construct the filter from config and persist its settings and diagnostics.
- Create `configs/stage1_sr010_replacement_filter_no_recloss.yaml`: approved ETTh1 experiment configuration.
- Create `tests/unit/test_filtering.py`: focused context, hard-gate, fallback, initialization, and gradient tests.
- Modify `tests/unit/test_config.py`, `tests/unit/test_relations.py`, `tests/unit/test_recovery.py`, and `tests/unit/test_teacher_and_losses.py`: component integration tests.
- Modify `tests/integration/test_gradient_flow.py`, `tests/integration/test_training_provenance.py`, and `tests/integration/test_training_smoke.py`: end-to-end gradient, diagnostics, and checkpoint coverage.
- Modify `tests/unit/test_masks.py`: cross-variant dynamic-mask equality guard.
- Modify `README.md`: exact validation-first experiment commands and decision rule.

---

### Task 1: Add the Backward-Compatible Filter Configuration Contract

**Files:**
- Modify: `src/malr/config.py`
- Create: `configs/stage1_sr010_replacement_filter_no_recloss.yaml`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: existing `Stage1Config`, `RelationConfig`, `load_config(Path)`, and `config_sha256(Stage1Config)`.
- Produces: `EdgeFilterConfig`, `RelationConfig.filter`, and the new `missing_aware_filter` selection value used by model construction in Task 6.

- [ ] **Step 1: Write failing tests for parsing, validation, and legacy hashes**

Add imports and tests equivalent to:

```python
from malr.config import EdgeFilterConfig, RecoveryConfig, config_sha256, load_config


def test_missing_aware_filter_experiment_config_is_explicit() -> None:
    cfg = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )

    assert cfg.relations.selection == "missing_aware_filter"
    assert cfg.relations.filter == EdgeFilterConfig(
        temporal_threshold=0.5,
        variable_threshold=0.5,
        historical_threshold=0.5,
        historical_candidate_pool=16,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
        lambda_keep=0.0001,
        straight_through=True,
    )
    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.recovery.fusion == "replacement_mlp"
    assert cfg.recovery.refiner_enabled is False
    assert cfg.ablations.use_recovery_loss is False
    assert cfg.ablations.use_temporal is True
    assert cfg.ablations.use_variable is True
    assert cfg.ablations.use_historical is True


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("temporal_threshold", 0.0, "threshold"),
        ("variable_threshold", 1.0, "threshold"),
        ("historical_candidate_pool", 0, "candidate_pool"),
        ("hidden_multiplier", True, "hidden_multiplier"),
        ("dropout", 1.0, "dropout"),
        ("temperature", 0.0, "temperature"),
        ("initial_keep_probability", 1.0, "initial_keep_probability"),
        ("lambda_keep", -0.1, "lambda_keep"),
        ("straight_through", False, "straight_through"),
    ],
)
def test_invalid_missing_aware_filter_config_is_rejected(
    tmp_path: Path, key: str, value: object, match: str
) -> None:
    raw = yaml.safe_load(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["relations"]["filter"][key] = value
    bad = tmp_path / f"bad_filter_{key}.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(bad)


def test_legacy_hard_topk_hash_ignores_default_filter_settings() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))
    payload_mapping = asdict(cfg)
    payload_mapping.pop("recovery")
    payload_mapping["relations"].pop("filter")
    payload = json.dumps(
        payload_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert config_sha256(cfg) == sha256(payload).hexdigest()
```

- [ ] **Step 2: Run the focused tests and confirm they fail for missing types/config**

Run:

```powershell
python -m pytest tests/unit/test_config.py -q
```

Expected: FAIL because `EdgeFilterConfig`, nested parsing, and the new YAML file do not exist, and `missing_aware_filter` is rejected.

- [ ] **Step 3: Implement the dataclass, nested parser, validation, and legacy hash rule**

Add this immutable configuration before `RelationConfig`:

```python
@dataclass(frozen=True)
class EdgeFilterConfig:
    temporal_threshold: float = 0.5
    variable_threshold: float = 0.5
    historical_threshold: float = 0.5
    historical_candidate_pool: int = 16
    hidden_multiplier: int = 2
    dropout: float = 0.0
    temperature: float = 1.0
    initial_keep_probability: float = 0.9
    lambda_keep: float = 0.0001
    straight_through: bool = True
```

Import `field` from `dataclasses` and add, as the final `RelationConfig` field,
`filter: EdgeFilterConfig = field(default_factory=EdgeFilterConfig)` to
`RelationConfig`. Parse
`relations.filter` explicitly rather than passing the nested dictionary into
the dataclass:

```python
relations_raw = dict(_require_mapping(raw["relations"], "relations"))
filter_raw = relations_raw.pop("filter", {})
relations = RelationConfig(
    **relations_raw,
    filter=EdgeFilterConfig(
        **_require_mapping(filter_raw, "relations.filter")
    ),
)
```

Accept only `hard_topk` and `missing_aware_filter`. Validate every constraint
from the test table and require `straight_through is True` when selection is
`missing_aware_filter`. In `config_sha256`, remove the default nested filter
mapping when `selection == "hard_topk"` so existing config hashes remain
unchanged.

Create the new YAML by copying the approved replacement-only configuration and
changing only:

```yaml
relations:
  selection: missing_aware_filter
  temporal_topk: 6
  variable_topk: 4
  historical_topk: 4
  recovered_nodes_can_be_sources: false
  num_message_passing_layers: 1
  filter:
    temporal_threshold: 0.5
    variable_threshold: 0.5
    historical_threshold: 0.5
    historical_candidate_pool: 16
    hidden_multiplier: 2
    dropout: 0.0
    temperature: 1.0
    initial_keep_probability: 0.9
    lambda_keep: 0.0001
    straight_through: true
```

- [ ] **Step 4: Run configuration tests**

Run:

```powershell
python -m pytest tests/unit/test_config.py -q
```

Expected: all configuration tests PASS, including unchanged legacy hashes.

- [ ] **Step 5: Commit the configuration contract**

```powershell
git add src/malr/config.py tests/unit/test_config.py configs/stage1_sr010_replacement_filter_no_recloss.yaml
git commit -m "feat: configure missing-aware edge filters"
```

---

### Task 2: Build the Bilateral Visible-Context Encoder

**Files:**
- Create: `src/malr/models/filtering.py`
- Create: `tests/unit/test_filtering.py`
- Modify: `src/malr/models/__init__.py`

**Interfaces:**
- Consumes: `visible`, `mask_embedding`, and `prior` tensors shaped `[B,C,N,D]`, plus `rho` and `target_eligible` shaped `[B,C,N]`.
- Produces: `BilateralContextOutput(query, left, right, left_available, right_available)` and `BilateralContextEncoder.forward(visible, mask_embedding, prior, rho, target_eligible)` for Task 6.

- [ ] **Step 1: Write failing tests for bilateral summaries and leakage-safe inputs**

Create `tests/unit/test_filtering.py` with focused tests including:

```python
import pytest
import torch

from malr.models.filtering import BilateralContextEncoder


def test_bilateral_summary_uses_visible_sources_on_each_side() -> None:
    module = BilateralContextEncoder(d_model=1, hidden_multiplier=2)
    visible = torch.tensor([[[[2.0], [100.0], [8.0], [14.0]]]])
    mask_embedding = torch.zeros_like(visible)
    prior = torch.zeros_like(visible)
    rho = torch.tensor([[[1.0, 0.0, 0.5, 1.0]]])
    target = torch.tensor([[[False, True, False, False]]])

    output = module(visible, mask_embedding, prior, rho, target)

    torch.testing.assert_close(output.left[0, 0, 1], torch.tensor([2.0]))
    expected_right = ((0.5 / 2.0) * 8.0 + (1.0 / 3.0) * 14.0) / (
        0.5 / 2.0 + 1.0 / 3.0
    )
    torch.testing.assert_close(
        output.right[0, 0, 1], torch.tensor([expected_right])
    )
    assert output.left_available[0, 0, 1]
    assert output.right_available[0, 0, 1]


def test_bilateral_summary_marks_an_absent_side() -> None:
    module = BilateralContextEncoder(d_model=2, hidden_multiplier=2)
    visible = torch.randn(1, 1, 3, 2)
    rho = torch.tensor([[[0.0, 0.0, 1.0]]])
    target = torch.tensor([[[True, False, False]]])

    output = module(
        visible,
        torch.zeros_like(visible),
        torch.zeros_like(visible),
        rho,
        target,
    )

    torch.testing.assert_close(output.left[0, 0, 0], torch.zeros(2))
    assert not output.left_available[0, 0, 0]
    assert output.right_available[0, 0, 0]


def test_context_encoder_zeroes_queries_for_non_target_nodes() -> None:
    torch.manual_seed(3)
    module = BilateralContextEncoder(d_model=4, hidden_multiplier=2)
    visible = torch.randn(1, 1, 3, 4)
    rho = torch.tensor([[[1.0, 0.0, 1.0]]])
    target = torch.tensor([[[False, True, False]]])
    output = module(
        visible,
        torch.zeros_like(visible),
        torch.zeros_like(visible),
        rho,
        target,
    )

    torch.testing.assert_close(output.query[:, :, 0], torch.zeros(1, 1, 4))
    torch.testing.assert_close(output.query[:, :, 2], torch.zeros(1, 1, 4))
```

- [ ] **Step 2: Run the new test file and confirm the import fails**

Run:

```powershell
python -m pytest tests/unit/test_filtering.py -q
```

Expected: FAIL because `malr.models.filtering` is absent.

- [ ] **Step 3: Implement vectorized bilateral summaries and the context MLP**

Create these public types:

```python
@dataclass(frozen=True)
class BilateralContextOutput:
    query: Tensor
    left: Tensor
    right: Tensor
    left_available: Tensor
    right_available: Tensor


class BilateralContextEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_multiplier: int = 2) -> None:
        super().__init__()
        hidden = hidden_multiplier * d_model
        self.network = nn.Sequential(
            nn.LayerNorm(5 * d_model + 3),
            nn.Linear(5 * d_model + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        visible: Tensor,
        mask_embedding: Tensor,
        prior: Tensor,
        rho: Tensor,
        target_eligible: Tensor,
    ) -> BilateralContextOutput:
        expected_latent_shape = visible.shape
        if visible.ndim != 4:
            raise ValueError("visible must have shape [B,C,N,D]")
        if mask_embedding.shape != expected_latent_shape:
            raise ValueError("mask_embedding must match visible")
        if prior.shape != expected_latent_shape:
            raise ValueError("prior must match visible")
        if rho.shape != visible.shape[:-1]:
            raise ValueError("rho must have shape [B,C,N]")
        if target_eligible.shape != rho.shape or target_eligible.dtype != torch.bool:
            raise ValueError("target_eligible must be Boolean with shape [B,C,N]")
        if not torch.isfinite(rho).all() or not ((rho >= 0) & (rho <= 1)).all():
            raise ValueError("rho must be finite and lie in [0,1]")

        patch_count = visible.shape[-2]
        index = torch.arange(patch_count, device=visible.device)
        signed_distance = index.view(1, -1) - index.view(-1, 1)
        distance = signed_distance.abs().to(dtype=visible.dtype)
        source_weight = rho.unsqueeze(-2) / (1.0 + distance.view(1, 1, patch_count, patch_count))
        source_visible = (rho > 0).unsqueeze(-2)
        left_mask = source_visible & (signed_distance < 0).view(1, 1, patch_count, patch_count)
        right_mask = source_visible & (signed_distance > 0).view(1, 1, patch_count, patch_count)

        def summarize(side_mask: Tensor) -> tuple[Tensor, Tensor]:
            weights = torch.where(side_mask, source_weight, torch.zeros_like(source_weight))
            denominator = weights.sum(dim=-1, keepdim=True)
            total = torch.einsum("bcij,bcjd->bcid", weights, visible)
            available = denominator.squeeze(-1) > 0
            normalized = total / denominator.clamp_min(torch.finfo(visible.dtype).eps)
            summary = torch.where(available.unsqueeze(-1), normalized, torch.zeros_like(total))
            return summary, available

        left, left_available = summarize(left_mask)
        right, right_available = summarize(right_mask)
        features = torch.cat(
            (
                visible,
                mask_embedding,
                prior,
                left,
                right,
                rho.unsqueeze(-1),
                left_available.unsqueeze(-1).to(visible.dtype),
                right_available.unsqueeze(-1).to(visible.dtype),
            ),
            dim=-1,
        )
        query = self.network(features)
        query = torch.where(target_eligible.unsqueeze(-1), query, torch.zeros_like(query))
        return BilateralContextOutput(
            query=query,
            left=left,
            right=right,
            left_available=left_available,
            right_available=right_available,
        )
```

Validate all shapes and finite `rho`. Build `[N,N]` signed
distance matrices on the input device, use source weights
`rho[j] / (1 + abs(j-i))`, normalize left and right independently with a
clamped denominator, and explicitly set empty summaries to zero. The `visible`
input is the latent already projected from masked patch values, exactly as
defined by the design; never substitute a clean patch latent. Concatenate five
`D`-wide tensors plus `rho`, left
availability, and right availability.

Export `BilateralContextEncoder` and `BilateralContextOutput` from
`src/malr/models/__init__.py`.

- [ ] **Step 4: Run focused context tests**

Run:

```powershell
python -m pytest tests/unit/test_filtering.py -q
```

Expected: all current filtering tests PASS.

- [ ] **Step 5: Commit the bilateral context encoder**

```powershell
git add src/malr/models/filtering.py src/malr/models/__init__.py tests/unit/test_filtering.py
git commit -m "feat: encode bilateral visible patch context"
```

---

### Task 3: Implement Deterministic Hard Filtering with Straight-Through Gradients

**Files:**
- Modify: `src/malr/models/filtering.py`
- Modify: `tests/unit/test_filtering.py`
- Modify: `src/malr/models/__init__.py`

**Interfaces:**
- Consumes: a query whose final axis is `D`, keys whose last two axes are `[M,D]`, auxiliary features whose last two axes are `[M,A]`, and base-score/eligibility tensors whose final axis is `M`; all leading batch/node axes must match.
- Produces: `FilteredEdgeSelection(weights, probabilities, hard_gate, selected, candidate_count, retained_count, keep_loss)` and `EdgeConfidenceFilter.forward(query, keys, auxiliary, base_scores, eligible)` for Tasks 4 and 5.

- [ ] **Step 1: Write failing hard-filter tests**

Add tests covering rejection, caps, ties, empty candidates, initialization, and
gradient flow:

```python
from malr.models.filtering import EdgeConfidenceFilter, hard_threshold_topk


def test_hard_threshold_topk_rejects_low_confidence_and_allows_zero() -> None:
    probabilities = torch.tensor([[0.9, 0.49, 0.8, 0.1]])
    base_scores = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    eligible = torch.ones_like(probabilities, dtype=torch.bool)

    gate, selected = hard_threshold_topk(
        probabilities, base_scores, eligible, threshold=0.5, max_edges=1
    )
    empty_gate, empty_selected = hard_threshold_topk(
        torch.full_like(probabilities, 0.1),
        base_scores,
        eligible,
        threshold=0.5,
        max_edges=2,
    )

    assert gate.tolist() == [[True, False, False, False]]
    assert selected.tolist() == [[0]]
    assert not empty_gate.any()
    assert empty_selected.tolist() == [[-1, -1]]


def test_equal_probabilities_use_base_score_then_index() -> None:
    probabilities = torch.full((1, 4), 0.9)
    base_scores = torch.tensor([[2.0, 3.0, 3.0, 1.0]])
    eligible = torch.ones_like(probabilities, dtype=torch.bool)

    _, selected = hard_threshold_topk(
        probabilities, base_scores, eligible, threshold=0.5, max_edges=2
    )

    assert selected.tolist() == [[1, 2]]


def test_filter_initialization_reproduces_base_topk_and_is_binary() -> None:
    module = EdgeConfidenceFilter(
        d_model=2,
        auxiliary_dim=1,
        max_edges=2,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )
    query = torch.zeros(1, 2)
    keys = torch.zeros(1, 4, 2)
    auxiliary = torch.zeros(1, 4, 1)
    scores = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    eligible = torch.ones(1, 4, dtype=torch.bool)

    result = module(query, keys, auxiliary, scores, eligible)

    assert result.selected.tolist() == [[1, 2]]
    assert set(result.hard_gate.unique().tolist()) <= {False, True}
    torch.testing.assert_close(
        result.probabilities,
        torch.full_like(result.probabilities, 0.9),
    )
    torch.testing.assert_close(result.weights.sum(-1), torch.ones(1))
    assert result.keep_loss.item() == pytest.approx(0.9)


def test_straight_through_filter_reaches_rejected_candidate_logits() -> None:
    module = EdgeConfidenceFilter(
        d_model=2,
        auxiliary_dim=1,
        max_edges=1,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    ).train()
    query = torch.randn(1, 2)
    keys = torch.randn(1, 3, 2)
    auxiliary = torch.randn(1, 3, 1)
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    eligible = torch.ones(1, 3, dtype=torch.bool)

    result = module(query, keys, auxiliary, scores, eligible)
    loss = (result.weights * torch.tensor([[1.0, -1.0, 2.0]])).sum()
    loss.backward()

    assert module.network[-1].weight.grad is not None
    assert torch.isfinite(module.network[-1].weight.grad).all()
    assert module.network[-1].weight.grad.abs().sum() > 0


def test_filter_with_no_eligible_candidate_is_finite_zero() -> None:
    module = EdgeConfidenceFilter(
        d_model=2,
        auxiliary_dim=1,
        max_edges=4,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )
    result = module(
        torch.zeros(1, 2),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2),
        torch.zeros(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(result.weights, torch.zeros(1, 2))
    torch.testing.assert_close(result.probabilities, torch.zeros(1, 2))
    assert result.selected.tolist() == [[-1, -1, -1, -1]]
    assert result.keep_loss.item() == 0.0
    assert torch.isfinite(result.keep_loss)
```

- [ ] **Step 2: Run the hard-filter tests and confirm they fail**

Run:

```powershell
python -m pytest tests/unit/test_filtering.py -q
```

Expected: FAIL because the selection function and confidence module are absent.

- [ ] **Step 3: Implement result types and stable hard selection**

Add:

```python
@dataclass(frozen=True)
class FilteredEdgeSelection:
    weights: Tensor
    probabilities: Tensor
    hard_gate: Tensor
    selected: Tensor
    candidate_count: Tensor
    retained_count: Tensor
    keep_loss: Tensor
```

Implement:

```python
def hard_threshold_topk(
    probabilities: Tensor,
    base_scores: Tensor,
    eligible: Tensor,
    *,
    threshold: float,
    max_edges: int,
) -> tuple[Tensor, Tensor]:
    """Threshold first; cap accepted edges by confidence, base score, index."""
```

Use stable sorts in reverse tie-break priority: ascending candidate index is
the base order, stable-sort by descending base score, then stable-sort by
descending probability. Mask ineligible or below-threshold entries before
taking the first `max_edges`. Return a dense Boolean gate and a fixed-width
selected-index tensor of width `max_edges`, padded with `-1` even when the
candidate axis is shorter.

- [ ] **Step 4: Implement the confidence MLP, straight-through gate, and stable weights**

Implement:

```python
class EdgeConfidenceFilter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        auxiliary_dim: int,
        max_edges: int,
        threshold: float,
        hidden_multiplier: int,
        dropout: float,
        temperature: float,
        initial_keep_probability: float,
    ) -> None:
        super().__init__()
        input_dim = 4 * d_model + auxiliary_dim
        hidden_dim = hidden_multiplier * d_model
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias,
            torch.logit(torch.tensor(initial_keep_probability)).item(),
        )
```

In `forward`, validate shapes; construct pair features; compute probabilities.
The pair feature order is exactly
`concat(query, key, abs(query-key), query*key, auxiliary)` after expanding the
query over the candidate axis. Set probabilities for ineligible entries to
exactly zero. Then call `hard_threshold_topk`; compute
`soft_gate = sigmoid((logits - logit(threshold)) / temperature)` and
`training_gate = hard_gate.detach() - soft_gate.detach() + soft_gate` while
training. Use the Boolean hard gate directly in evaluation.

Compute stable aggregation weights from `training_gate * exp(base_score-max)`.
When the forward denominator is zero, replace it with one using a detached
zero-denominator condition so the message is zero and backward values remain
finite. Compute `keep_loss` as the mean probability over eligible entries; use
`probabilities.sum() * 0.0` when none are eligible.

Export all three public symbols from `src/malr/models/__init__.py`.

- [ ] **Step 5: Run filtering tests and static syntax compilation**

Run:

```powershell
python -m pytest tests/unit/test_filtering.py -q
python -m compileall -q src/malr/models/filtering.py
```

Expected: both commands exit with code 0.

- [ ] **Step 6: Commit the filter primitive**

```powershell
git add src/malr/models/filtering.py src/malr/models/__init__.py tests/unit/test_filtering.py
git commit -m "feat: add straight-through hard edge filtering"
```

---

### Task 4: Integrate Filtering into Temporal and Variable Relations

**Files:**
- Modify: `src/malr/models/relations.py`
- Modify: `tests/unit/test_relations.py`

**Interfaces:**
- Consumes: `EdgeConfidenceFilter`, filter query `[B,C,N,D]`, visible sources, `rho`, and existing target eligibility.
- Produces: extended `RelationSelection` carrying filter tensors and keep loss without changing legacy callers.

- [ ] **Step 1: Write failing relation-filter integration tests**

Add a small test filter helper and assertions:

```python
from malr.models.filtering import EdgeConfidenceFilter


def _edge_filter(d_model: int, auxiliary_dim: int, max_edges: int):
    return EdgeConfidenceFilter(
        d_model=d_model,
        auxiliary_dim=auxiliary_dim,
        max_edges=max_edges,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )


def test_temporal_filter_uses_all_visible_candidates_then_caps_edges() -> None:
    relation = TemporalRelation(d_model=2, topk=2, bias=False)
    query = torch.randn(1, 1, 4, 2)
    sources = torch.randn(1, 1, 4, 2)
    rho = torch.tensor([[[1.0, 0.0, 0.5, 1.0]]])
    damaged = torch.tensor([[[False, True, False, False]]])

    result = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=2, max_edges=2),
    )

    assert result.filter_probabilities is not None
    assert result.hard_gate is not None
    assert result.retained_count is not None
    assert result.candidate_count[0, 0, 1].item() == 3
    assert result.retained_count[0, 0, 1].item() == 2
    assert result.message.shape == query.shape


def test_variable_filter_receives_source_availability_and_variable_identity() -> None:
    relation = VariableRelation(d_model=2, topk=1, bias=False)
    query = torch.randn(1, 3, 2, 2)
    sources = torch.randn(1, 3, 2, 2)
    rho = torch.tensor([[[1.0, 0.0], [0.5, 1.0], [1.0, 1.0]]])
    damaged = torch.tensor([[[False, True], [False, False], [False, False]]])
    variable_embeddings = torch.randn(3, 2)

    result = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=5, max_edges=1),
        variable_embeddings=variable_embeddings,
    )

    assert result.retained_count is not None
    assert result.retained_count[0, 0, 1].item() <= 1
    assert result.keep_loss is not None
    assert torch.isfinite(result.keep_loss)


def test_temporal_filter_initialization_matches_legacy_message() -> None:
    torch.manual_seed(31)
    relation = TemporalRelation(d_model=2, topk=2, bias=False)
    query = torch.randn(1, 1, 4, 2)
    sources = torch.randn(1, 1, 4, 2)
    rho = torch.tensor([[[1.0, 0.0, 0.5, 1.0]]])
    damaged = torch.tensor([[[False, True, False, False]]])

    legacy = relation.select(query, sources, rho, damaged)
    filtered = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=2, max_edges=2),
    )

    torch.testing.assert_close(filtered.selected, legacy.selected)
    torch.testing.assert_close(filtered.weights, legacy.weights)
    torch.testing.assert_close(filtered.message, legacy.message)


def test_variable_filter_initialization_matches_legacy_message() -> None:
    torch.manual_seed(37)
    relation = VariableRelation(d_model=2, topk=1, bias=False)
    query = torch.randn(1, 3, 2, 2)
    sources = torch.randn(1, 3, 2, 2)
    rho = torch.tensor([[[1.0, 0.0], [0.5, 1.0], [1.0, 1.0]]])
    damaged = torch.tensor([[[False, True], [False, False], [False, False]]])

    legacy = relation.select(query, sources, rho, damaged)
    filtered = relation.select(
        query,
        sources,
        rho,
        damaged,
        filter_query=query,
        edge_filter=_edge_filter(2, auxiliary_dim=5, max_edges=1),
        variable_embeddings=torch.randn(3, 2),
    )

    torch.testing.assert_close(filtered.selected, legacy.selected)
    torch.testing.assert_close(filtered.weights, legacy.weights)
    torch.testing.assert_close(filtered.message, legacy.message)
```

Also retain every existing legacy selection assertion unchanged.

- [ ] **Step 2: Run relation tests and confirm the new keyword arguments fail**

Run:

```powershell
python -m pytest tests/unit/test_relations.py -q
```

Expected: FAIL because `RelationSelection` and `select` do not expose filter fields or inputs.

- [ ] **Step 3: Extend `RelationSelection` without breaking legacy behavior**

Use optional fields with `None` defaults:

```python
@dataclass(frozen=True)
class RelationSelection:
    message: Tensor
    weights: Tensor
    selected: Tensor
    candidate_count: Tensor
    pool_count: Tensor | None = None
    filter_probabilities: Tensor | None = None
    hard_gate: Tensor | None = None
    retained_count: Tensor | None = None
    keep_loss: Tensor | None = None
```

Legacy `stable_masked_topk` calls must construct the first four fields exactly
as before and leave every optional field at `None`.

- [ ] **Step 4: Add temporal filter features and filtered aggregation**

Extend `TemporalRelation.select` with keyword-only `filter_query` and
`edge_filter`. Build auxiliary features for every target/source pair:

```text
signed_distance = (source_index - target_index) / max(N - 1, 1)
source_rho       = rho[source_index]
```

When `edge_filter is None`, run the unchanged legacy selector. Otherwise call
the filter over every eligible non-self visible source and aggregate with its
dense weights. Return original eligible candidate counts plus filter outputs;
set `pool_count` to `N-1` for damaged targets and zero for observed targets.

- [ ] **Step 5: Add variable filter features and filtered aggregation**

Extend `VariableRelation.select` with keyword-only `filter_query`,
`edge_filter`, and `variable_embeddings`. Construct candidate keys in the
existing `[B,N,target_variable,source_variable,D]` layout. The auxiliary vector
has width `2*D+1`:

```text
target variable embedding
source variable embedding
source observed fraction
```

Require `variable_embeddings.shape == [C,D]` only when the edge filter is
active. Set `pool_count` to `C-1` for damaged targets and zero for observed
targets. Preserve the old path byte-for-byte in behavior when it is absent.

- [ ] **Step 6: Run legacy and filtered relation tests**

Run:

```powershell
python -m pytest tests/unit/test_relations.py -q
```

Expected: all tests PASS, including distant temporal selection, missing-source
exclusion, zero-candidate fallback, and new filtered paths.

- [ ] **Step 7: Commit temporal and variable integration**

```powershell
git add src/malr/models/relations.py tests/unit/test_relations.py
git commit -m "feat: filter temporal and variable recovery edges"
```

---

### Task 5: Add Historical Candidate Pre-Pooling and Edge Filtering

**Files:**
- Modify: `src/malr/models/relations.py`
- Modify: `tests/unit/test_relations.py`

**Interfaces:**
- Consumes: existing factor distances and availability, `filter_query`, projected prototype centers, an `EdgeConfidenceFilter` with auxiliary width 3, and `candidate_pool`.
- Produces: final historical prototype IDs, filtered weights, retained counts, probabilities, and keep loss while preserving the legacy historical selector.

- [ ] **Step 1: Write failing tests for pre-pooling, rejection, and prototype-ID mapping**

Add this fixture and tests using a bank with more than sixteen prototypes:

```python
from collections.abc import Sequence


def _bank_with_many_centers(centers: Sequence[float]) -> PrototypeBank:
    source = np.zeros((max(128, 64 * len(centers)), 1), dtype=np.float64)
    records = []
    for prototype_id, center in enumerate(centers):
        segment = np.arange(64, dtype=np.float64)
        segment[24:40] = center
        factors = compute_context_factors(segment, np.ones(64, bool), train_std=1.0)
        records.append(
            PrototypeRecord(
                variable_id=0,
                start=prototype_id * 64,
                raw_segment=segment,
                center_start=24,
                center_stop=40,
                factors=factors,
                occurrence_count=2,
                mean_profile_distance=0.0,
            )
        )
    return PrototypeBank(
        dataset="fixture",
        train_end=source.shape[0],
        source_checksum=source_checksum(source),
        config_sha256="fixture",
        records=(tuple(records),),
        factor_mean=np.zeros((1, 7)),
        factor_std=np.ones((1, 7)),
    )


def test_historical_filter_prepool_maps_back_to_bank_prototype_ids() -> None:
    bank = _bank_with_many_centers(tuple(float(value) for value in range(20)))
    relation = HistoricalRelation(
        bank,
        topk=4,
        candidate_pool=16,
        d_model=4,
    )
    projection = nn.Linear(16, 4)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.full((1, 1, 12), PATCH_PARTIAL, dtype=torch.long)
    filter_query = torch.randn(1, 1, 12, 4)

    result = relation.select(
        context,
        context_mask,
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=projection,
        filter_query=filter_query,
        edge_filter=_edge_filter(4, auxiliary_dim=3, max_edges=4),
    )

    assert result.selected.shape == (1, 1, 12, 4)
    assert torch.all((result.selected == -1) | (result.selected < 20))
    assert result.filter_probabilities is not None
    assert result.filter_probabilities.shape[-1] == 16


def test_historical_filter_can_reject_every_prepooled_prototype() -> None:
    bank = _bank_with_many_centers(tuple(float(value) for value in range(20)))
    relation = HistoricalRelation(
        bank,
        topk=4,
        candidate_pool=16,
        d_model=4,
    )
    projection = nn.Linear(16, 4)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)
    filter_query = torch.randn(1, 1, 12, 4)
    edge_filter = _edge_filter(4, auxiliary_dim=3, max_edges=4)
    with torch.no_grad():
        edge_filter.network[-1].weight.zero_()
        edge_filter.network[-1].bias.fill_(-20.0)

    result = relation.select(
        context,
        context_mask,
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1),
        prototype_projection=projection,
        filter_query=filter_query,
        edge_filter=edge_filter,
    )

    assert result.retained_count is not None
    assert torch.count_nonzero(result.retained_count) == 0
    torch.testing.assert_close(result.message, torch.zeros_like(result.message))
    assert torch.isfinite(result.message).all()


def test_historical_filter_initialization_matches_legacy_message() -> None:
    bank = _bank_with_many_centers(tuple(float(value) for value in range(20)))
    legacy_relation = HistoricalRelation(bank, topk=4, d_model=4)
    filtered_relation = HistoricalRelation(
        bank,
        topk=4,
        candidate_pool=16,
        d_model=4,
    )
    projection = nn.Linear(16, 4)
    context = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
    context_mask = torch.ones_like(context, dtype=torch.bool)
    state = torch.ones((1, 1, 12), dtype=torch.long)
    common = {
        "context": context,
        "context_mask": context_mask,
        "state": state,
        "query_mean": torch.zeros(1, 1),
        "query_std": torch.ones(1, 1),
        "train_std": torch.ones(1),
        "prototype_projection": projection,
    }

    legacy = legacy_relation.select(**common)
    filtered = filtered_relation.select(
        **common,
        filter_query=torch.randn(1, 1, 12, 4),
        edge_filter=_edge_filter(4, auxiliary_dim=3, max_edges=4),
    )

    torch.testing.assert_close(filtered.selected, legacy.selected)
    torch.testing.assert_close(filtered.weights, legacy.weights)
    torch.testing.assert_close(filtered.message, legacy.message)
```

Keep the existing test proving historical selection does not read prototype
centers while ranking factor contexts.

- [ ] **Step 2: Run historical relation tests and confirm they fail**

Run:

```powershell
python -m pytest tests/unit/test_relations.py -q
```

Expected: FAIL because `candidate_pool`, `filter_query`, and `edge_filter` are unsupported.

- [ ] **Step 3: Separate historical factor pre-pooling from final selection**

Add `candidate_pool: int | None = None` to `HistoricalRelation.__init__`; default
to `topk` for the legacy path. Under filtering, use `stable_masked_topk` with
`largest=False`, `normalize="none"`, and width `candidate_pool` to obtain the
factor-nearest bank IDs. Gather distances, shared counts, availability counts,
and center patches for those IDs.

Keep factor distance and ID gathering inside `torch.no_grad()`. Project the
normalized centers after leaving that block so gradients still reach the
current model's patch projection. Record `pool_count` as the number of valid
pre-pooled prototype IDs for each damaged target, capped at `candidate_pool`.

- [ ] **Step 4: Apply the learned filter and preserve bank IDs**

Build historical auxiliary features of width 3:

```text
standardized factor distance
shared factor count / 7
prototype available factor count / 7
```

Call the filter on the pre-pool, aggregate projected center latents with its
weights, and map selected pool positions back to original bank prototype IDs.
Pass `-log(standardized_factor_distance + eps)` as `base_scores`. A smaller
distance is therefore larger/better, and exponentiating this score reproduces
the legacy inverse-distance aggregation at filter initialization.
Pad rejected or absent positions with `-1`. The non-filtered path must retain
the existing inverse-distance aggregation and output shapes. If the bank is
empty under filtering, return zero candidate/retained/confidence tensors,
zero message/weights, `-1` selected IDs, and a differentiable zero keep loss;
do not fall back to the shorter legacy `RelationSelection` return.

- [ ] **Step 5: Run all relation tests**

Run:

```powershell
python -m pytest tests/unit/test_relations.py -q
```

Expected: all legacy and filtered historical tests PASS.

- [ ] **Step 6: Commit historical filtering**

```powershell
git add src/malr/models/relations.py tests/unit/test_relations.py
git commit -m "feat: filter historical prototype edges"
```

---

### Task 6: Compose the Three Filters in Missing-Aware Recovery

**Files:**
- Modify: `src/malr/models/filtering.py`
- Modify: `src/malr/models/recovery.py`
- Modify: `src/malr/models/__init__.py`
- Modify: `src/malr/cli.py`
- Modify: `tests/unit/test_recovery.py`
- Modify: `tests/unit/test_teacher_and_losses.py`

**Interfaces:**
- Consumes: `EdgeFilterConfig`, three relation modules, current mask embeddings and priors, and the optional historical bank.
- Produces: `MissingAwareEdgeFilter`, `RecoveryOutput.filter_keep_loss`, and per-relation filter diagnostics for the objective and trainer.

- [ ] **Step 1: Write failing composition and fallback tests**

Add `PATCH_MISSING` and `PATCH_OBSERVED` imports from
`malr.data.patching`, then add tests that construct the new selector through the
real composition root:

```python
from malr.cli import build_experiment_model
from malr.config import load_config
from malr.models.filtering import MissingAwareEdgeFilter


def test_cli_builds_filter_only_for_missing_aware_selection() -> None:
    cfg = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )
    model = build_experiment_model(
        cfg,
        "ETTh1",
        "ours",
        bank=_empty_bank(variable_count=7),
    )

    assert model.recovery is not None
    assert model.recovery.edge_filter is not None
    assert model.recovery.global_refiner is None


def test_filtered_factory_forward_exposes_three_relation_diagnostics() -> None:
    cfg = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )
    model = build_experiment_model(
        cfg,
        "ETTh1",
        "ours",
        bank=_empty_bank(variable_count=7),
    ).eval()
    patch_count = model.core.patch_num
    values = torch.randn(1, 7, patch_count, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, 0, 1] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.full((1, 7, patch_count), PATCH_OBSERVED, dtype=torch.long)
    state[:, 0, 1] = PATCH_MISSING

    output = model(
        values,
        mask,
        rho,
        state,
        torch.zeros(1, 7),
        torch.ones(1, 7),
        raw_context=torch.zeros(1, 96, 7),
        context_mask=torch.ones(1, 96, 7, dtype=torch.bool),
        train_std=torch.ones(7),
    )

    assert output.recovery is not None
    for relation in ("temporal", "variable", "historical"):
        assert f"{relation}_filter_candidate_count" in output.recovery.diagnostics
        assert f"{relation}_filter_retained_count" in output.recovery.diagnostics
```

Place that factory test in `tests/unit/test_teacher_and_losses.py`, where the
existing `_empty_bank` fixture already lives. In `tests/unit/test_recovery.py`,
add these local helpers and fallback/leakage tests:

```python
def _filtered_replacement_recovery() -> MissingAwareRecovery:
    edge_filter = MissingAwareEdgeFilter(
        d_model=4,
        temporal_topk=2,
        variable_topk=1,
        historical_topk=1,
        temporal_threshold=0.5,
        variable_threshold=0.5,
        historical_threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )
    return MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=4,
        temporal_topk=2,
        variable_topk=1,
        use_historical=False,
        fusion="replacement_mlp",
        edge_filter=edge_filter,
        relation_selection="missing_aware_filter",
    )


def _force_all_filter_logits(module: MissingAwareRecovery, value: float) -> None:
    assert module.edge_filter is not None
    with torch.no_grad():
        for head in (
            module.edge_filter.temporal,
            module.edge_filter.variable,
            module.edge_filter.historical,
        ):
            head.network[-1].weight.zero_()
            head.network[-1].bias.fill_(value)


def _mixed_state_recovery_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    values = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
          [[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]]]]
    )
    mask = torch.tensor(
        [[[[True, True, True, True], [True, False, False, False], [False, False, False, False]],
          [[True, True, True, True], [True, True, True, True], [True, True, True, True]]]]
    )
    rho = mask.to(torch.float32).mean(dim=-1)
    state = torch.tensor(
        [[[PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
          [PATCH_OBSERVED, PATCH_OBSERVED, PATCH_OBSERVED]]]
    )
    base = torch.randn(1, 2, 3, 4)
    return values, mask, rho, state, base


def test_all_rejected_relations_fall_back_without_changing_observed_nodes() -> None:
    module = _filtered_replacement_recovery()
    _force_all_filter_logits(module, value=-20.0)
    values, mask, rho, state, base = _mixed_state_recovery_inputs()

    result = module(values, mask, rho, state, base_latents=base)

    assert torch.equal(result.latents[state == PATCH_OBSERVED], base[state == PATCH_OBSERVED])
    assert torch.isfinite(result.latents).all()
    assert torch.isfinite(result.filter_keep_loss)
    assert result.diagnostics["temporal_filter_retained_count"].sum() == 0
    assert result.diagnostics["variable_filter_retained_count"].sum() == 0


def test_filter_query_is_unchanged_when_masked_raw_values_change() -> None:
    module = _filtered_replacement_recovery()
    values, mask, rho, state, base = _mixed_state_recovery_inputs()
    changed = values.clone()
    changed[~mask] = 1e6

    first = module(values, mask, rho, state, base_latents=base)
    second = module(changed, mask, rho, state, base_latents=base)

    torch.testing.assert_close(first.messages.temporal, second.messages.temporal)
    torch.testing.assert_close(first.messages.variable, second.messages.variable)
    torch.testing.assert_close(first.latents, second.latents)


def test_filter_keep_loss_averages_nonempty_relation_types_equally() -> None:
    module = _filtered_replacement_recovery()
    assert module.edge_filter is not None
    with torch.no_grad():
        for head, probability in (
            (module.edge_filter.temporal, 0.8),
            (module.edge_filter.variable, 0.4),
        ):
            head.network[-1].weight.zero_()
            head.network[-1].bias.copy_(torch.logit(torch.tensor(probability)))
    values, mask, rho, state, base = _mixed_state_recovery_inputs()

    result = module(values, mask, rho, state, base_latents=base)

    assert result.filter_keep_loss.item() == pytest.approx(0.6)
```

The local recovery helper disables historical recovery because it does not own
raw-context and bank fixtures. The factory-forward test above exercises all
three enabled paths and requires the empty historical bank to return structured
zero diagnostics rather than bypassing the filter contract.

Parameterize the existing
`test_recovery_relations_use_immutable_pre_recovery_visible_sources` over
`filtered=False/True`. For the true case, pass a `MissingAwareEdgeFilter` built
with the test's `d_model=2`, temporal/variable/historical limits `1/1/1`, and
thresholds `0.5`, hidden multiplier `2`, dropout `0.0`, temperature `1.0`, and
initial keep probability `0.9`, plus
`relation_selection="missing_aware_filter"`. Keep its candidate-count
assertions identical in both cases; this proves filtered relations still read
the immutable pre-recovery `visible` tensor and never a recovered node.

- [ ] **Step 2: Run recovery tests and confirm construction fails**

Run:

```powershell
python -m pytest tests/unit/test_recovery.py tests/unit/test_teacher_and_losses.py -q
```

Expected: FAIL because recovery has no edge-filter composition or loss field.

- [ ] **Step 3: Add the `MissingAwareEdgeFilter` coordinator**

In `filtering.py`, implement a module owning:

```python
class MissingAwareEdgeFilter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        temporal_topk: int,
        variable_topk: int,
        historical_topk: int,
        temporal_threshold: float,
        variable_threshold: float,
        historical_threshold: float,
        hidden_multiplier: int,
        dropout: float,
        temperature: float,
        initial_keep_probability: float,
    ) -> None:
        super().__init__()
        common = {
            "d_model": d_model,
            "hidden_multiplier": hidden_multiplier,
            "dropout": dropout,
            "temperature": temperature,
            "initial_keep_probability": initial_keep_probability,
        }
        self.context = BilateralContextEncoder(d_model, hidden_multiplier)
        self.temporal = EdgeConfidenceFilter(
            auxiliary_dim=2,
            max_edges=temporal_topk,
            threshold=temporal_threshold,
            **common,
        )
        self.variable = EdgeConfidenceFilter(
            auxiliary_dim=2 * d_model + 1,
            max_edges=variable_topk,
            threshold=variable_threshold,
            **common,
        )
        self.historical = EdgeConfidenceFilter(
            auxiliary_dim=3,
            max_edges=historical_topk,
            threshold=historical_threshold,
            **common,
        )

    def context_query(
        self,
        visible: Tensor,
        mask_embedding: Tensor,
        prior: Tensor,
        rho: Tensor,
        target_eligible: Tensor,
    ) -> BilateralContextOutput:
        return self.context(visible, mask_embedding, prior, rho, target_eligible)
```

Use independent thresholds and common hidden/dropout/temperature/initial
probability settings. Expose the exact `context_query` method above and the three named filter
heads. Do not import application config classes into the model module.

- [ ] **Step 4: Extend recovery construction and forward flow**

Add `edge_filter: MissingAwareEdgeFilter | None = None` and
`relation_selection: str = "hard_topk"` to `MissingAwareRecovery.__init__`.
Require an edge filter exactly when selection is `missing_aware_filter`.

In `forward`:

1. compute `mask_embedding = W_m(mask.float())` separately;
2. preserve the existing structural prior;
3. obtain the bilateral filter query for damaged targets;
4. pass the relevant confidence head and filter query into each enabled
   relation;
5. collect each non-`None` keep loss and average enabled, non-empty relation
   terms equally;
6. expose a differentiable scalar `filter_keep_loss`, using
   `latents.sum() * 0.0` for the legacy selector;
7. keep replacement fusion and observed-node bypass unchanged.

Extend `RecoveryOutput`:

```python
@dataclass(frozen=True)
class RecoveryOutput:
    latents: Tensor
    visible: Tensor
    prior: Tensor
    recovered: Tensor
    messages: RelationMessages
    diagnostics: dict[str, Tensor]
    filter_keep_loss: Tensor
```

Store the detached filter query only for tests/debug mode if diagnostics are
already enabled for that forward path; otherwise test it through the public
context module to avoid bloating production manifests. Always store per-node
detached tensors under these exact stable names for each enabled relation:

```text
{relation}_filter_candidate_count
{relation}_filter_pool_count
{relation}_filter_retained_count
{relation}_filter_candidate_confidence_sum
{relation}_filter_retained_confidence_sum
{relation}_filter_threshold
{relation}_filter_max_edges
```

The candidate-confidence sum is `probabilities.sum(-1)` because ineligible
probabilities are exactly zero. The retained-confidence sum is
`(probabilities * hard_gate).sum(-1)`. The trainer derives means and zero-edge
rates from these sufficient statistics instead of averaging batch means. The
threshold and maximum-edge entries are detached scalar tensors copied from the
active filter head.

Extend the existing `_tiny_ours` test helper in
`tests/unit/test_teacher_and_losses.py` with `filtered: bool = False`. When true,
construct a `MissingAwareEdgeFilter` with `d_model=8`, Top-K limits `2/1/4`,
thresholds `0.5`, hidden multiplier `2`, zero dropout, temperature `1.0`, and
initial keep probability `0.9`; pass it to `MissingAwareRecovery` together with
`relation_selection="missing_aware_filter"`. When false, pass no edge filter and
keep `relation_selection="hard_topk"`. This gives Task 7 one small real filtered
model without duplicating construction code.

- [ ] **Step 5: Construct the coordinator in `build_experiment_model`**

When `config.relations.selection == "missing_aware_filter"`, construct
`MissingAwareEdgeFilter` from `config.relations.filter`, the model width, and
the three existing Top-K limits. Pass historical candidate pool 16 into
`HistoricalRelation` and pass the coordinator plus selection string into
`MissingAwareRecovery`. For legacy selection, pass no coordinator and leave
all current constructors unchanged.

- [ ] **Step 6: Run recovery, relation, and configuration tests**

Run:

```powershell
python -m pytest tests/unit/test_filtering.py tests/unit/test_relations.py tests/unit/test_recovery.py tests/unit/test_teacher_and_losses.py tests/unit/test_config.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit recovery composition**

```powershell
git add src/malr/models/filtering.py src/malr/models/recovery.py src/malr/models/__init__.py src/malr/cli.py tests/unit/test_recovery.py tests/unit/test_teacher_and_losses.py
git commit -m "feat: compose filtered recovery messages"
```

---

### Task 7: Add Filter Retention to the Stage-1 Loss and Verify End-to-End Gradients

**Files:**
- Modify: `src/malr/training/losses.py`
- Modify: `src/malr/models/stage1.py`
- Modify: `src/malr/training/trainer.py`
- Modify: `src/malr/cli.py`
- Modify: `tests/unit/test_teacher_and_losses.py`
- Modify: `tests/integration/test_gradient_flow.py`

**Interfaces:**
- Consumes: `RecoveryOutput.filter_keep_loss` and `EdgeFilterConfig.lambda_keep`.
- Produces: `Stage1Loss.filter_keep`, `Stage1Loss.lambda_keep`, and a total loss that trains the hard filter through forecast error plus the retention penalty.

- [ ] **Step 1: Write failing loss arithmetic and gradient tests**

Import `PATCH_MISSING` in `tests/integration/test_gradient_flow.py`, then add:

```python
def test_stage1_total_adds_filter_keep_penalty() -> None:
    prediction = torch.tensor([0.0, 2.0])
    target = torch.zeros_like(prediction)
    recovery = prediction.sum() * 0.0 + 3.0
    filter_keep = prediction.sum() * 0.0 + 0.25

    result = Stage1Loss.create(
        prediction,
        target,
        recovery,
        filter_keep,
        damaged_patch_count=1,
        lambda_rec=2.0,
        lambda_keep=0.1,
    )

    assert result.total.item() == pytest.approx(8.025)
    assert result.filter_keep.item() == pytest.approx(0.25)
    assert result.lambda_keep == pytest.approx(0.1)


def test_forecast_and_keep_loss_reach_edge_filter_parameters() -> None:
    torch.manual_seed(23)
    model = _tiny_ours(fusion="replacement_mlp", filtered=True).train()
    batch_size = 2
    patch_count = model.core.patch_num
    values = torch.randn(batch_size, 1, patch_count, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 1] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.zeros(batch_size, 1, patch_count, dtype=torch.long)
    state[:, :, 1] = PATCH_MISSING
    forecast_target = torch.randn(batch_size, 1, 4)
    patch_valid = torch.ones_like(state, dtype=torch.bool)

    result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
        forecast_target,
        None,
        patch_valid,
        use_recovery_loss=False,
        lambda_keep=0.0001,
    )
    result.loss.total.backward()

    filter_output = model.recovery.edge_filter.temporal.network[-1]
    assert filter_output.weight.grad is not None
    assert torch.isfinite(filter_output.weight.grad).all()
    assert filter_output.weight.grad.abs().sum() > 0
    assert model.core.encoder.layers[0].self_attn.W_Q.weight.grad is not None
    assert model.core.head.linear.weight.grad is not None
```

- [ ] **Step 2: Run loss and gradient tests and confirm signature failures**

Run:

```powershell
python -m pytest tests/unit/test_teacher_and_losses.py tests/integration/test_gradient_flow.py -q
```

Expected: FAIL because `Stage1Loss.create` and `training_objective` do not accept filter retention inputs.

- [ ] **Step 3: Extend `Stage1Loss` and its metrics**

Use:

```python
@dataclass(frozen=True)
class Stage1Loss:
    total: Tensor
    forecast_mse: Tensor
    recovery_mse: Tensor
    filter_keep: Tensor
    damaged_patch_count: int
    lambda_keep: float
```

`create` validates both coefficients as non-negative and computes:

```python
total = (
    prediction_loss
    + float(lambda_rec) * recovery_mse
    + float(lambda_keep) * filter_keep
)
```

Include `filter_keep` and `lambda_keep` in `detached_metrics`. Update all
existing test calls to pass a differentiable zero filter loss and
`lambda_keep=0.0`; do not silently change old arithmetic.

- [ ] **Step 4: Route the loss through model and trainer**

Add `lambda_keep: float = 0.0` to `Stage1Model.training_objective` and
`Trainer.__init__`. Select `output.recovery.filter_keep_loss` only for the
filtered `ours` path; otherwise create a differentiable zero. Pass
`lambda_keep` into `Stage1Loss.create`.

In `run_training`, set:

```python
lambda_keep = (
    config.relations.filter.lambda_keep
    if selected is ModelVariant.OURS
    and config.relations.selection == "missing_aware_filter"
    else 0.0
)
```

Pass that value to `Trainer` and include it in checkpoint metadata.

- [ ] **Step 5: Run loss, gradient, and training smoke tests**

Run:

```powershell
python -m pytest tests/unit/test_teacher_and_losses.py tests/integration/test_gradient_flow.py tests/integration/test_training_smoke.py -q
```

Expected: all selected tests PASS with finite, non-zero edge-filter gradients.

- [ ] **Step 6: Commit objective integration**

```powershell
git add src/malr/training/losses.py src/malr/models/stage1.py src/malr/training/trainer.py src/malr/cli.py tests/unit/test_teacher_and_losses.py tests/integration/test_gradient_flow.py tests/integration/test_training_smoke.py
git commit -m "feat: train edge filters with forecast loss"
```

---

### Task 8: Persist Validation Filter Diagnostics and Prove Mask Fairness

**Files:**
- Modify: `src/malr/training/trainer.py`
- Modify: `src/malr/cli.py`
- Modify: `tests/integration/test_training_provenance.py`
- Modify: `tests/unit/test_masks.py`

**Interfaces:**
- Consumes: per-node `{relation}_filter_*` tensors in `RecoveryOutput.diagnostics` and existing `EpochMaskManager`.
- Produces: `ValidationMetrics.diagnostics`, `TrainingResult.best_val_diagnostics`, checkpoint key `validation_diagnostics`, and matching run-manifest training diagnostics.

- [ ] **Step 1: Write failing diagnostics persistence and mask-equality tests**

Add:

```python
def test_epoch_mask_sequence_is_independent_of_model_rng_consumption() -> None:
    first = EpochMaskManager(200, 7, 0.10, 5, base_seed=2021)
    torch.manual_seed(1)
    _ = torch.randn(1000)
    first_masks = [first.mask_for_epoch(epoch) for epoch in range(3)]

    second = EpochMaskManager(200, 7, 0.10, 5, base_seed=2021)
    torch.manual_seed(999)
    _ = torch.randn(2000)
    second_masks = [second.mask_for_epoch(epoch) for epoch in range(3)]

    for first_mask, second_mask in zip(first_masks, second_masks, strict=True):
        np.testing.assert_array_equal(first_mask, second_mask)
```

In `tests/integration/test_training_provenance.py`, import `PrototypeBank` and
`source_checksum`, define the representative values once, and replace the
skipped-training helper so legacy variants receive an empty dictionary while a
filtered `ours` run receives real-looking diagnostics:

```python
FILTER_DIAGNOSTICS = {
    "temporal_keep_rate": 0.5,
    "temporal_zero_edge_rate": 0.25,
    "variable_keep_rate": 0.4,
    "variable_zero_edge_rate": 0.1,
    "historical_keep_rate": 0.2,
    "historical_zero_edge_rate": 0.3,
}


def _empty_bank(variable_count: int) -> PrototypeBank:
    source = np.zeros((64, variable_count), dtype=np.float64)
    return PrototypeBank(
        dataset="fixture",
        train_end=64,
        source_checksum=source_checksum(source),
        config_sha256="fixture",
        records=tuple(() for _ in range(variable_count)),
        factor_mean=np.zeros((variable_count, 7)),
        factor_std=np.ones((variable_count, 7)),
    )


def _skip_expensive_training(
    self: Trainer, train_batches_for_epoch: object, validation_batches: object
) -> TrainingResult:
    filtered = (
        self.model.recovery is not None
        and self.model.recovery.edge_filter is not None
    )
    diagnostics = dict(FILTER_DIAGNOSTICS) if filtered else {}
    save_checkpoint(
        self.checkpoint_path,
        self.model.state_dict(),
        self.tracker.metadata,
        epoch=0,
        val_mse=0.5,
        val_mae=0.4,
        validation_diagnostics=diagnostics,
    )
    return TrainingResult(
        best_checkpoint=self.checkpoint_path,
        best_val_mse=0.5,
        best_val_mae=0.4,
        best_val_diagnostics=diagnostics,
        best_epoch=0,
        epochs_completed=1,
        training_seconds=0.0,
        trainable_parameters=sum(
            parameter.numel() for parameter in self.model.parameters()
        ),
    )
```

Leave the existing clean/missing provenance test in place and assert its
checkpoint `validation_diagnostics` and manifest
`best_val_diagnostics` are both `{}`. Add a filtered-ours provenance test that:

1. loads `configs/stage1_sr010_replacement_filter_no_recloss.yaml`;
2. creates the ETTh1 fixture series and validation mask by reusing the existing
   `_series`, `_mask_identity`, `generate_variable_missing_mask`, and
   `save_fixed_mask` calls;
3. monkeypatches `cli.load_bank` to return `_empty_bank(7)` and `Trainer.fit` to
   `_skip_expensive_training`;
4. calls `cli.run_training(config, series, protocol, "ETTh1", "ours",
   artifacts, device="cpu", bank_path=tmp_path / "bank.npz")`;
5. loads the returned checkpoint and manifest and makes these assertions:

```python
assert checkpoint["validation_diagnostics"] == FILTER_DIAGNOSTICS
assert manifest["training"]["best_val_diagnostics"] == FILTER_DIAGNOSTICS
```

- [ ] **Step 2: Run focused tests and confirm diagnostics fields fail**

Run:

```powershell
python -m pytest tests/unit/test_masks.py tests/integration/test_training_provenance.py -q
```

Expected: the mask test PASS under the current manager; provenance additions
FAIL because validation diagnostics are not stored.

- [ ] **Step 3: Aggregate diagnostics with correct denominators**

Extend:

```python
@dataclass(frozen=True)
class ValidationMetrics:
    mse: float
    mae: float
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint: Path
    best_val_mse: float
    best_val_mae: float
    best_val_diagnostics: dict[str, float]
    best_epoch: int
    epochs_completed: int
    training_seconds: float
    trainable_parameters: int
    test_evaluated: bool = False
```

During validation, aggregate totals rather than averaging batch means. For each
relation compute:

```text
candidate_availability = total eligible candidates / max(total pool positions, 1)
keep_rate       = total retained edges / max(total eligible candidates, 1)
zero_edge_rate  = damaged targets retaining zero edges / max(damaged targets, 1)
mean_confidence = sum eligible probabilities / max(eligible probability count, 1)
mean_retained_confidence = sum retained probabilities / max(total retained edges, 1)
```

Repeat retained-count means for the four approved damaged-target `rho` bands,
using diagnostic suffixes `rho_eq_0`, `rho_0_025`, `rho_025_050`, and
`rho_050_100` for `rho == 0`, `0 < rho <= 0.25`, `0.25 < rho <= 0.50`, and
`0.50 < rho < 1`, respectively. Omit filter
keys for non-filtered variants rather than writing misleading zeros.
Copy the threshold and maximum-edge scalar for each relation into the final
diagnostic dictionary and assert that it is constant across validation batches.

Run `rg -n "TrainingResult\(|ValidationMetrics\(" src tests` and update every
constructor found: legacy paths pass `diagnostics={}` and
`best_val_diagnostics={}`. Do not give these dictionaries mutable dataclass
defaults.

- [ ] **Step 4: Store diagnostics with the MSE-selected checkpoint**

Add `val_diagnostics: Mapping[str,float]` to
`BestCheckpointTracker.consider`. Save a plain dictionary as
`validation_diagnostics` beside `val_mse` and `val_mae`. Track the matching
dictionary in `best_diagnostics` and return it from `Trainer.fit` as
`best_val_diagnostics`. Keep checkpoint selection based only on MSE.

Print one compact validation line containing three keep rates and three
zero-edge rates only for filtered runs. `asdict(TrainingResult)` then carries
the selected diagnostics into the existing manifest `training` object.

- [ ] **Step 5: Run provenance, mask, checkpoint, and trainer tests**

Run:

```powershell
python -m pytest tests/unit/test_masks.py tests/integration/test_training_provenance.py tests/integration/test_checkpoint_protocol.py tests/integration/test_training_smoke.py -q
```

Expected: all selected tests PASS; the stored diagnostics correspond to the
same epoch as the best validation MSE.

- [ ] **Step 6: Commit diagnostics and fairness guards**

```powershell
git add src/malr/training/trainer.py src/malr/cli.py tests/integration/test_training_provenance.py tests/unit/test_masks.py
git commit -m "feat: persist edge filter diagnostics"
```

---

### Task 9: Complete End-to-End Verification and Document the Validation-First Run

**Files:**
- Modify: `README.md`
- Modify: `tests/integration/test_synthetic_end_to_end.py`
- Modify: `tests/integration/test_training_smoke.py`

**Interfaces:**
- Consumes: the completed config, model factory, training objective, checkpoint diagnostics, and current CLI.
- Produces: a documented reproducible experiment path and a final full-suite verification record.

- [ ] **Step 1: Add a synthetic filtered-run acceptance test**

Create a synthetic test that loads the new config, builds a small training-only
prototype bank fixture, constructs `ours` through `build_experiment_model`,
runs one `training_objective` call with dynamic missing inputs, performs one
optimizer step, and asserts:

```python
assert torch.isfinite(result.loss.total)
assert result.loss.filter_keep.item() > 0.0
assert result.output.recovery is not None
assert result.output.recovery.filter_keep_loss.requires_grad
assert result.output.forecast.shape == forecast_target.shape
assert result.output.recovery.latents.shape[-1] == model.core.d_model
```

Also assert the original missing and hard-Top-K `ours` synthetic tests remain
unchanged.

- [ ] **Step 2: Run the synthetic acceptance test**

Run:

```powershell
python -m pytest tests/integration/test_synthetic_end_to_end.py -q
```

Expected: PASS because Tasks 1-8 already supply every interface. If it fails,
return to the task that owns the first failing interface instead of adding an
ad-hoc compatibility path in this acceptance task.

- [ ] **Step 3: Add the exact user-run protocol to README**

Document PowerShell commands with a fresh artifact namespace:

```powershell
$CFG="configs/stage1_sr010_replacement_filter_no_recloss.yaml"
$ART="artifacts/etth1_sr010_replacement_filter_no_recloss"

malr make-eval-masks --config $CFG --dataset ETTh1 --artifacts $ART
malr train-clean --config $CFG --dataset ETTh1 --artifacts $ART --device cuda
malr build-bank --config $CFG --dataset ETTh1 --artifacts $ART
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant missing --device cuda
malr train --config $CFG --dataset ETTh1 --artifacts $ART --variant ours --device cuda
```

State explicitly that the user compares `best_val_mse` and `best_val_mae`
from the two training results and does not call `malr evaluate` unless Filter
beats Missing on both. Do not add an automatic test invocation to the training
commands.

- [ ] **Step 4: Run targeted model verification**

Run:

```powershell
python -m pytest tests/unit/test_filtering.py tests/unit/test_relations.py tests/unit/test_recovery.py tests/unit/test_teacher_and_losses.py tests/integration/test_gradient_flow.py tests/integration/test_synthetic_end_to_end.py tests/integration/test_training_smoke.py -q
```

Expected: all targeted tests PASS.

- [ ] **Step 5: Run the entire repository test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all repository tests PASS; the optional S4M checkout test may remain
skipped under its existing documented condition.

- [ ] **Step 6: Check formatting, syntax, and the final diff**

Run:

```powershell
python -m compileall -q src tests
git diff --check
git status --short
git diff --stat
```

Expected: compilation exits 0, `git diff --check` emits no errors, and the
status contains only files named by this plan plus the two pre-existing
untracked configuration files.

- [ ] **Step 7: Commit the end-to-end coverage and README**

```powershell
git add README.md tests/integration/test_synthetic_end_to_end.py tests/integration/test_training_smoke.py
git commit -m "test: verify missing-aware filter workflow"
```

- [ ] **Step 8: Record the implementation verification evidence**

Run:

```powershell
git log --oneline --decorate -12
git status --short
python -m pytest -q
```

Expected: the feature commits are present, only the two pre-existing untracked
configs remain, and the fresh full-suite result has zero failures.

Do not run ETTh1 training during implementation verification. The user runs
the documented GPU commands after reviewing the completed implementation.
