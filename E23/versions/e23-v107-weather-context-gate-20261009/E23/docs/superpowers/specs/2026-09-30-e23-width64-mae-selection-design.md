# E23 Width 64 Validation-MAE Selection Design

Date: 2026-09-30. This is a separate checkpoint-selection experiment. The existing width 64 MSE-selected Stage 2 checkpoint and its one-shot test are immutable.

## Reason and alternatives

The width 64 model beat SAITS on selected-test MSE (`0.429693777` versus `0.440584382`) but not on MAE (`0.437349478` versus SAITS `0.436957896` and E22 `0.433248216`). During the completed width 64 Stage 2 run, the fixed-validation MSE minimum was epoch 26 (`0.760599101`, MAE `0.593048152`), while the validation MAE minimum was epoch 87 (`0.590439670`, MSE `0.764769530`). These are validation observations; epoch 87's test result is unknown. A new run may predeclare validation MAE as its selector without using the prior selected test to choose a checkpoint within that run.

1. **Select by fixed-validation MAE while keeping forecast-MSE training (chosen).** This isolates the selection rule and reuses the checked width 64 Stage 1 state. The selected checkpoint may trade some MSE for MAE.
2. Add MAE to the Stage 2 training loss. This changes the parameter trajectory and obscures whether selection alone helps; reserve it for a later trial.
3. Use a validation MSE/MAE composite or Pareto threshold. That adds an arbitrary weight or cutoff and more tuning freedom; reserve it until a simple MAE selector is evaluated.

## Contract

The P4 source, normalization, masks, split, patching, predictor, width 64 H/V/T repair model, seed `2021`, batch `128`, Stage 2 LR `0.0001`, optimizer reset, 100 epochs and pure forecast-MSE loss remain unchanged. Stage 2 consumes the already selected width 64 Stage 1 manifest and its paired training-only H bank/frozen key, exactly as the MSE-selected run did. It performs a fresh 100-epoch Stage 2 training run into a new `e23_multi_expert_width64_mae_v3` artifact root. No train or test data are used to choose the epoch; the earliest epoch with the lowest fixed-validation forecast MAE wins. The selected checkpoint and manifest contain both validation MSE and MAE, selector `mae`, checkpoint hash, Stage 1 manifest/checkpoint hashes and unchanged H state hashes. The old manifests without an explicit selector retain the exact meaning `mse`.

`train_forecast(..., selection_metric="mse" | "mae")` and CLI `train-forecast --selection-metric` expose the policy; default `mse` preserves old behavior. The loader rejects an unsupported policy or any mismatch between manifest and checkpoint before evaluation. Selected evaluation records the policy and remains write-once. The new run profiles fixed test inputs without computing metrics, then evaluates its selected checkpoint exactly once and reports forecast MSE/MAE, both shared-reference DRRs, expert diagnostics and speed. The result is compared with SAITS and E22 on the same locked P4 test, without rerunning baseline tests.

## Limits

Reuse of the selected Stage 1 checkpoint avoids inventing a different pretraining model; total work attributed to this candidate includes that Stage 1 training plus the new Stage 2 run. This is one seed and one validation-selected test, so a success needs independent replication. If MAE still misses E22 or MSE loses the SAITS margin, retain this result and design another independent variant rather than reselecting with its test labels.
