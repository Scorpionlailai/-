# Missing-Aware Edge Filter Design

Date: 2026-09-06

## 1. Objective

Replace the stage-1 fixed hard Top-K relation selector with a learned,
patch-specific, missing-aware edge filter. A temporal, variable, or historical
edge may contribute to recovery only when the information that remains around
the damaged target patch provides sufficient evidence that the edge is useful.

The filter is part of the recovery front-end. The PatchTST encoder and
forecasting head remain unchanged and train jointly from scratch with the
front-end.

## 2. Motivation

The current selectors always retain up to a fixed number of eligible edges:

- temporal: 6;
- variable: 4;
- historical: 4.

This is a relative ranking rule. It cannot distinguish a genuinely useful
candidate from the least-bad candidate in a poor candidate set. The problem is
most severe for fully missing patches: their current relation query contains
the mask and structural prior but little target-specific content. Fixed Top-K
can therefore inject weak or systematically biased messages.

Recent ETTh1 results support changing selection before increasing recovery
capacity. Replacement fusion improved validation MSE over the missing
PatchTST baseline, while the gain shrank on test and test MAE reversed. Adding
a global refiner did not materially improve validation. These observations are
consistent with unreliable input messages rather than an underpowered fusion
network.

TimeFilter motivates patch-specific filtering of irrelevant dependencies. Its
published method dynamically chooses dependency regions for each patch rather
than assuming that the largest edge weights are always useful. This design
adapts that principle to missing-data recovery at individual-edge granularity;
it does not copy TimeFilter's relation-type router.

## 3. Scope and Non-Goals

### In scope

- Construct a filter query from the target patch's visible values, bilateral
  visible context, missing mask, observed fraction, and structural prior.
- Learn a separate confidence function for temporal, variable, and historical
  edges.
- Use deterministic hard edges in the forward pass and at evaluation time.
- Allow zero through the configured maximum number of edges for each relation.
- Train the filter jointly from scratch using forecast loss and a small edge
  retention penalty.
- Preserve the existing candidate availability rules, prototype bank leakage
  guards, and prohibition on recovered nodes becoming sources.
- Add configuration, diagnostics, provenance, and automated tests.

### Out of scope

- Modifying the PatchTST encoder or forecasting head.
- Initializing `ours` from a clean PatchTST checkpoint.
- Enabling latent recovery loss in the first filter experiment.
- Enabling the global latent refiner in the first filter experiment.
- Adding multiple graph-message-passing rounds.
- Copying TimeFilter's GNN forecasting backbone.
- Tuning filter decisions against the official test set.

## 4. Definitions

Let:

- `B` be batch size;
- `C` be the number of variables;
- `N` be the number of PatchTST patches;
- `P` be patch length;
- `D` be latent width;
- `rho[b,c,n]` be the observed fraction of a patch;
- `z_vis[b,c,n]` be the patch latent produced from the masked input;
- `state[b,c,n]` be observed, partial, or fully missing.

Only partial or fully missing nodes are filter targets. A source node is
eligible only when `rho > 0`. A recovered latent is never recycled as a source.

## 5. Bilateral Target Context

For target node `i=(c,n)`, summarize all eligible same-variable patches on the
left and right separately. The fixed distance weight is

```text
a(i,j) = rho[j] / (1 + abs(j - i)).
```

The normalized summaries are

```text
c_left[i]  = sum(j < i) a(i,j) * z_vis[j] / max(sum(j < i) a(i,j), eps)
c_right[i] = sum(j > i) a(i,j) * z_vis[j] / max(sum(j > i) a(i,j), eps).
```

If a side has no eligible source, its summary is the zero vector. A binary
side-availability flag is supplied to the context encoder so that a genuine
zero-valued summary is distinguishable from an absent side.

The target filter query is

```text
q[i] = ContextMLP(concat(
    z_vis[i],
    mask_embedding[i],
    structural_prior[i],
    c_left[i],
    c_right[i],
    rho[i],
    left_available[i],
    right_available[i]
)).
```

`ContextMLP` outputs width `D`. It is used to judge edges, not as the recovered
latent itself. All context inputs are available from the masked look-back
window at inference time; no clean target value enters the query.

## 6. Relation-Specific Edge Confidence

Each relation owns an independent filter MLP. The common pair representation
contains

```text
q[i], k[j], abs(q[i] - k[j]), q[i] * k[j], auxiliary_features[i,j].
```

The filter produces one scalar logit and probability:

```text
u_r[i,j] = FilterMLP_r(pair_features[i,j])
p_r[i,j] = sigmoid(u_r[i,j]).
```

The auxiliary features are relation-specific:

- **Temporal:** normalized signed patch distance and source observed fraction.
- **Variable:** source observed fraction plus target and source variable
  embeddings.
- **Historical:** standardized factor distance, shared valid factor count,
  and prototype factor availability.

Temporal and variable keys are learned projections of eligible visible source
latents. Historical keys are projected prototype-center latents. Historical
factor computation and candidate retrieval remain based only on the
training-built prototype bank and the masked query context.

Each filter MLP uses LayerNorm, one hidden linear layer, GELU, optional dropout,
and a scalar output. Its hidden width is `filter.hidden_multiplier * D`.

## 7. Candidate Pools

The filter evaluates these pools before final hard selection:

- temporal: every eligible non-self patch in the same variable;
- variable: every eligible non-self variable at the same patch position;
- historical: the nearest 16 available prototypes under the existing factor
  distance, or all available prototypes when fewer than 16 exist.

The historical pre-pool is a compute bound, not a final acceptance rule. A
prototype in the pool still contributes only if the learned filter accepts it.

## 8. Hard Selection and Straight-Through Training

For relation `r`, use the configured threshold `tau_r`:

```text
accepted_r[i,j] = p_r[i,j] >= tau_r.
```

If more accepted edges remain than the relation maximum, retain those with the
largest filter probabilities. Equal probabilities are broken by the existing
base relation score and then by ascending candidate index. This yields
deterministic selection.

The limits remain:

```text
0 <= temporal edges   <= 6
0 <= variable edges   <= 4
0 <= historical edges <= 4.
```

The forward hard gate is `g_hard`. To train the filter through forecast loss,
use the straight-through surrogate

```text
g_soft = sigmoid((u - logit(tau)) / temperature)
g_st   = stop_gradient(g_hard - g_soft) + g_soft.
```

The numerical forward value of `g_st` equals the binary hard gate. Its
backward derivative follows `g_soft`, including for candidates that do not
survive the hard maximum-edge cap. Validation and test use `g_hard` directly.

The scalar output weight of every filter is initialized to zero, and its bias
is initialized to `logit(initial_keep_probability)`. With the approved initial
probability of `0.9` and thresholds of `0.5`, all eligible candidates initially
pass the threshold. The maximum-edge cap therefore reproduces the existing
hard Top-K selection at initialization, using base scores as the tie-breaker.

No noisy routing is used in the first version. Dynamic training masks already
provide input variation, and deterministic routing isolates the effect of the
new filter.

## 9. Message Aggregation and Fallback

For a relation with base score `s_base` and value vector `v`, aggregate

```text
numerator   = sum(j) g_st[i,j] * exp(s_base[i,j]) * v[j]
denominator = sum(j) g_st[i,j] * exp(s_base[i,j])
message[i]  = numerator / max(denominator, eps).
```

The implementation must use a numerically stable masked softmax equivalent.
If no edge is retained, the relation message is exactly zero and contains no
NaN or infinite value.

If all three relation types retain zero edges, replacement fusion continues to
receive the visible target latent and structural prior. It must not invent a
dummy edge or fail. Observed target nodes still bypass recovery exactly.

## 10. Training Objective

The primary objective remains standardized-space forecast MSE. Let `p_r`
contain probabilities over eligible candidates for damaged targets. Compute
the mean probability separately per enabled relation, excluding empty
relation-target sets, and then average enabled relation terms equally:

```text
L_keep = mean_r(mean_eligible(p_r))
L_total = L_forecast + lambda_keep * L_keep.
```

The initial value is

```text
lambda_keep = 1e-4.
```

This is a weak edge-cost prior: forecast loss determines which relationships
are worth retaining, while `L_keep` prevents an unpenalized all-open solution.
Zero-edge recovery remains legal. The first experiment keeps
`use_recovery_loss: false`; no clean teacher latent or clean-checkpoint
initialization is introduced.

## 11. Configuration Contract

Existing configurations remain backward compatible with
`relations.selection: hard_topk`. Add a new selection value and nested filter
configuration:

```yaml
relations:
  selection: missing_aware_filter
  temporal_topk: 6
  variable_topk: 4
  historical_topk: 4
  recovered_nodes_can_be_sources: false

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

Validation rules require thresholds and initial probability in `(0,1)`, a
positive candidate pool, multiplier, and temperature, dropout in `[0,1)`, and
non-negative `lambda_keep`. The first implementation requires
`straight_through: true` for `missing_aware_filter`.

Create a new ETTh1 experiment configuration named
`stage1_sr010_replacement_filter_no_recloss.yaml`. It keeps replacement fusion,
disables the global refiner and recovery loss, and enables all three relation
sources.

## 12. Component Boundaries

Add `src/malr/models/filtering.py` with:

- `BilateralContextEncoder` for target queries;
- `EdgeConfidenceFilter` for one relation type;
- `MissingAwareEdgeFilter` for the three relation filters and selection;
- immutable result structures containing hard gates, probabilities, selected
  indices, counts, and retention loss.

Update existing modules as follows:

- `relations.py`: expose candidate keys, values, base scores, eligibility, and
  relation auxiliary features; preserve legacy hard Top-K behavior.
- `recovery.py`: construct bilateral context, invoke enabled filters, aggregate
  filtered messages, and return filter diagnostics and loss.
- `stage1.py`: add the weighted filter retention term to the training loss for
  `ours` when the new selector is active.
- `config.py`: parse and validate the new contract with backward-compatible
  defaults.
- training/reporting code: persist filter configuration, loss, and summary
  diagnostics in checkpoints and run manifests.

PatchTST code receives the final recovered latents through its current
interface and is otherwise unchanged.

## 13. Diagnostics

Record detached diagnostics for each enabled relation:

- eligible candidate count;
- candidate-pool count;
- retained edge count;
- retained fraction;
- mean candidate confidence;
- mean retained confidence;
- zero-edge target fraction;
- threshold and maximum-edge setting.

Also summarize retained counts by target observed-fraction bands:

```text
rho = 0
0 < rho <= 0.25
0.25 < rho <= 0.50
0.50 < rho < 1
```

These diagnostics must not become training inputs. They detect all-open,
all-closed, relation domination, and missing-severity inversions.

## 14. Reproducibility and Leakage Rules

Keep the approved ETTh1 protocol:

- `seq_len=96` and `pred_len=96`;
- `start_rate=0.10` and `block_len=5`;
- seed `2021`;
- train-fitted standardization and standardized-space MSE/MAE;
- dynamic training masks;
- fixed validation and test masks;
- existing one-year ETTh1 training split;
- training-only historical prototype construction.

The existing dynamic mask manager derives the timeline mask from its own base
seed and epoch. It must remain independent of model initialization and must be
covered by a cross-variant equality test. Fixed validation/test mask checksums
must match across compared runs.

The clean checkpoint may remain a provenance dependency of bank construction,
but it does not initialize `ours`. The prototype bank stores training-derived
raw centers and factors; prototype latents are projected by the jointly trained
current model.

## 15. Automated Tests

Add tests that prove:

1. bilateral summaries use only eligible visible sources and separate left
   from right;
2. absent sides return zeros with a false availability flag;
3. clean target values never affect a filter query under a fixed masked input;
4. each relation probability and gate has the expected shape;
5. probabilities below threshold are rejected;
6. retained count is between zero and the configured maximum;
7. equal-confidence selection is deterministic;
8. no accepted edge returns an exact zero message without NaN;
9. observed nodes remain exactly unchanged;
10. recovered nodes cannot become sources;
11. hard forward values are binary while forecast loss reaches filter
    parameters through the straight-through path;
12. initialization reproduces legacy hard Top-K indices and messages;
13. `L_keep` excludes ineligible entries and weights relation types equally;
14. legacy `hard_topk` configurations and checkpoints remain valid;
15. dynamic training masks and fixed evaluation masks match across variants;
16. checkpoint, leakage, gradient-flow, synthetic end-to-end, and training
    smoke tests remain green.

## 16. Experiment and Decision Protocol

Use a new artifact directory and compare:

1. `missing`: PatchTST trained from scratch on masked inputs;
2. `replacement`: the current replacement MLP with fixed hard Top-K;
3. `filter`: replacement MLP with the missing-aware edge filter.

All three use protocol-identical masks and training settings. The first filter
run enables temporal, variable, and historical sources. `missing_pretrained`
is retained in the codebase but is not a primary comparison.

Inspect training and validation only. The filter advances only when

```text
filter validation MSE < missing validation MSE
filter validation MAE < missing validation MAE.
```

The preferred outcome also beats the current replacement model on both
validation metrics. Diagnostics must show neither persistent all-open nor
persistent all-closed behavior. After the configuration is frozen, evaluate
the selected checkpoint once on the fixed test mask. A failed validation run
does not authorize test evaluation.

If the all-source filter fails, use validation-only source ablations in this
order: temporal plus variable, temporal only, variable only, then historical
reintroduction. Do not increase network depth before those diagnostics are
complete.

## 17. Risks and Mitigations

- **All-open filter:** the retention penalty and retained-fraction diagnostics
  expose and discourage this outcome.
- **All-closed filter:** high initial keep probability, the small regularizer,
  and the forecast objective provide a stable starting point and fallback.
- **Straight-through mismatch:** validation uses the exact inference-time hard
  gate; temperature remains configurable but fixed in the first experiment.
- **Historical compute:** factor-distance pre-pooling bounds prototype scoring
  to 16 candidates per target before the learned filter.
- **Selection instability:** deterministic evaluation, stable tie-breaking,
  and no routing noise make repeated runs reproducible under the fixed seed.
- **Test overfitting:** architecture, thresholds, and loss settings are chosen
  using validation only; the previous test observations are not used to tune
  this new filter.

## 18. References

- Hu, Y. et al. *TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration
  for Time Series Forecasting*. ICML 2025.
  https://proceedings.mlr.press/v267/hu25ac.html
- Official TimeFilter implementation:
  https://github.com/TROUBADOUR000/TimeFilter
