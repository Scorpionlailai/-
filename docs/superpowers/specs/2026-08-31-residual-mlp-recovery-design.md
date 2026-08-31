# Residual-MLP Missing-Patch Recovery Design

Date: 2026-08-31

## 1. Objective

Revise the stage-1 missing-patch recovery front-end so that it cannot destroy the useful representation already produced by the PatchTST patch projection at initialization. The first experiment will compare only:

1. PatchTST trained from scratch on dynamically masked inputs (`missing`), and
2. PatchTST trained from scratch with the revised recovery front-end (`ours`).

S4M is out of scope. `missing_pretrained` is retained in the codebase but is not part of this first experiment.

## 2. Motivation

Under the fixed ETTh1 protocol with `seq_len=96`, `pred_len=96`, `start_rate=0.10`, seed `2021`, and test final missing rate `0.4091301843`, the current additive recovery front-end did not consistently improve on the PatchTST missing baseline. The current implementation replaces a damaged patch latent with a direct sum of a learned prior and randomly initialized fused relation messages. This introduces a representation distribution shift before the relation branch has learned a useful correction.

The revised design must start from exactly the PatchTST missing baseline and learn only a residual correction for damaged patches.

## 3. Scope and Non-Goals

### In scope

- Add a configurable residual-MLP fusion path to `MissingAwareRecovery`.
- Preserve the existing temporal, variable, and historical hard-top-k relation selectors.
- Preserve joint from-scratch optimization of the recovery front-end and PatchTST.
- Preserve the existing dynamic training masks, fixed validation/test masks, standardization, split protocol, and leakage guards.
- Retain the legacy additive fusion as an explicit ablation option.
- Add diagnostics and automated tests for residual behavior.

### Out of scope

- S4M experiments or integration work.
- Soft or differentiable top-k selection.
- New graph layers, additional message-passing rounds, or attention-based fusion.
- Clean-checkpoint initialization of `ours`.
- Historical-bank redesign.
- Recovery-loss redesign.
- Hyperparameter sweeps on the test set.

## 4. Configuration Contract

Add a top-level recovery configuration:

```yaml
recovery:
  fusion: residual_mlp
  hidden_multiplier: 2
  dropout: 0.0
  output_init: zero
  damage_scaling: missing_fraction
```

Supported values for the first implementation are:

- `fusion`: `additive` or `residual_mlp`.
- `hidden_multiplier`: positive integer; the residual hidden width is `hidden_multiplier * d_model`.
- `dropout`: value in `[0, 1)`.
- `output_init`: `zero` for the residual path.
- `damage_scaling`: `missing_fraction`, defined as `1 - rho`.

The existing additive implementation remains available so previous ablations remain reproducible. New stage-1 experiments default to `residual_mlp` only in the new experiment configuration; existing configuration files keep their recorded behavior unless explicitly changed.

## 5. Residual Recovery Architecture

For every patch node, retain the existing messages:

- temporal message `m_t`,
- variable message `m_v`,
- historical message `m_h`, and
- structural prior `p`, comprising variable, patch-position, and patch-state embeddings.

Construct the residual input:

```text
m = concat(m_t, m_v, m_h, p)       # [..., 4 * d_model]
```

The correction network is:

```text
h       = GELU(Linear(LayerNorm(m)))
h       = Dropout(h)
delta_z = Linear(h)                 # [..., d_model]
```

The first linear layer maps `4 * d_model` to `hidden_multiplier * d_model`. The output projection maps the hidden width to `d_model`. Both weight and bias of the output projection are initialized to zero.

Let `rho` be the observed fraction of a patch. The final latent is:

```text
damage_strength = 1 - rho
z_out = z_base + damage_strength * delta_z
```

The state contract additionally enforces:

- observed patches return `z_base` exactly;
- partially observed patches receive a correction proportional to their missing fraction;
- fully missing patches receive the full correction;
- the structural prior is an input to the residual branch and is never directly added to the output latent.

Because the output projection is zero-initialized, `delta_z` is exactly zero at initialization. Therefore, the initialized `ours` forward path is identical to the PatchTST missing baseline for all patch states.

## 6. Data and Gradient Flow

1. The standardized masked input is patchified with the existing PatchTST patch length, stride, and end-replication rule.
2. PatchTST's shared patch projection produces `z_base`.
3. Existing hard-top-k temporal, variable, and historical selectors produce relation messages without allowing recovered nodes to become sources.
4. The residual MLP produces `delta_z`.
5. Only damaged nodes receive the scaled residual correction.
6. The unchanged PatchTST Transformer encoder and forecasting head consume `z_out`.
7. Forecast MSE backpropagates through PatchTST, the residual MLP, and the enabled relation modules.

The zero-initialized output layer receives gradients on the first update. Earlier residual-MLP layers begin receiving non-zero gradients after the output layer departs from zero; this short staged start is intentional and keeps initialization equivalent to the baseline.

## 7. Loss and Training Protocol

The first experiment uses forecast loss only:

```yaml
ablations:
  use_temporal: true
  use_variable: true
  use_historical: true
  use_recovery_loss: false
```

The existing latent recovery loss remains implemented and configurable but is disabled for this experiment. It will be reintroduced only after residual fusion demonstrates a validation improvement over the missing baseline.

Both `missing` and `ours` are trained from scratch. They use the same:

- seed;
- dynamically regenerated training mask for each epoch;
- fixed validation and test masks;
- PatchTST hyperparameters;
- optimizer and learning-rate schedule;
- train/validation/test split;
- train-fitted standardization statistics.

The clean PatchTST checkpoint may still be produced as a dependency for historical prototype selection, but it is not used to initialize `ours` and is not an additional competing method in this experiment.

## 8. Diagnostics

Expose the following detached diagnostics without using them as training inputs:

- `residual_delta_norm` per patch;
- `residual_applied_norm` after multiplication by `1 - rho`;
- mean applied residual norm for partial patches;
- mean applied residual norm for fully missing patches;
- existing candidate counts, selected indices, and hard-top-k weights for each relation type.

These diagnostics are intended to distinguish three failure modes:

1. the residual remains effectively zero;
2. the residual grows excessively and destabilizes the PatchTST latent scale;
3. one relation source dominates or supplies no candidates.

## 9. Automated Tests

Add or revise tests to prove:

1. residual output has the expected shape;
2. the residual output projection is zero-initialized;
3. at initialization, recovered latents equal `z_base` for observed, partial, and fully missing patches;
4. observed patch latents remain exactly unchanged after residual parameters become non-zero;
5. `rho=1` yields zero applied correction;
6. `rho=0` yields the full correction;
7. an intermediate `rho` scales the correction proportionally;
8. forecast loss produces gradients for the residual output projection;
9. after a non-zero output projection is present, forecast loss reaches the earlier residual-MLP layer;
10. the additive fusion mode continues to satisfy its previous tests;
11. existing gradient-flow, leakage-guard, checkpoint, and synthetic end-to-end tests remain green.

## 10. Experiment Sequence

Use a new configuration and artifact directory so no finalized run is silently reused.

1. Generate or copy protocol-identical fixed validation/test masks into the new artifact namespace with checksum verification.
2. Train the clean PatchTST dependency only if the historical bank builder requires it.
3. Build the historical prototype bank under the existing leakage-safe protocol.
4. Train a fresh PatchTST `missing` baseline.
5. Train `ours` with residual MLP and recovery loss disabled.
6. Compare validation MSE before authorizing test evaluation.
7. Evaluate both selected checkpoints on the same fixed test mask.

No S4M run or result enters this comparison.

## 11. Acceptance Criteria

Implementation acceptance requires all automated tests to pass and the dry-run/config inspection to show the intended residual settings.

Experimental acceptance requires:

- primary: `ours` validation MSE is lower than the fresh `missing` validation MSE;
- final: `ours` test MSE is lower than `missing` on the identical fixed mask;
- guardrail: test MAE does not materially worsen;
- provenance: matching seed, config identity, source CSV hash, mask hash, metric space, split bounds, and window construction.

If the residual version does not improve validation MSE, do not add network depth. The next step is a controlled relation-source ablation in the order temporal-only, variable-only, temporal-plus-variable, and then historical, using validation data only.
