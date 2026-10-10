# E23 v12 V-source diagnostic and predictor-head scope

Date: 2026-10-01. This is a read-only analysis of the selected v12 checkpoint on the 2,688 fixed ETTh1 P4 validation windows. It does not train a variant or read test labels. The diagnostic script is `scripts/diagnose_e23_v12_variable.py`; its JSON is in the ignored v12 trial root at `benchmarks/v12_variable_source_diagnostic_v3.json`. The train-only pairwise fits use deterministic epoch-0 training masks and jointly observed historical coordinates; validation target values are used only to measure diagnostic errors. Script SHA-256: `014775b61ef0745a478ff7942da6661070c1f92d0cfc92aedb264a8236291618`; JSON SHA-256: `c3a808b42adcc4d0baeee3bcc3c4686ac95965eea9b98e8199034f53ff1ce321`.

## Findings

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Damaged target patches where contextual and global top-4 V source sets match | 85.38% | The contextual source rule changes the chosen set on a minority of patches. |
| Selected v12 checkpoint, original contextual V: validation MSE/MAE | 0.745447729 / 0.588212069 | Exact reproduction of the archived selected validation result. |
| Same checkpoint with V source choice swapped to global top-4: validation MSE/MAE | 0.745463689 / 0.588180523 | This is an inference intervention; the altered model was not retrained. |
| Selected v12 checkpoint, contextual V minus global top-4 at inference: validation MAE | +0.00003155 | A source swap inside the already trained v12 model has almost no aggregate effect and is marginally worse on MAE. |
| Same source swap: validation MSE | −0.00001596 | The MSE effect is also very small. This is an inference intervention, not a retrained ablation. |
| Train-observed pairwise affine source mapping versus raw source on hidden target coordinates | Lower MAE for all 7 targets | Raw cross-variable numerical agreement is a weak reconstruction proxy. |
| Pairwise affine source mapping versus zero baseline on hidden target coordinates | Lower MAE for 6 of 7 targets | Source relationships carry some signal beyond shrinkage, but target variable 4 does not benefit. |
| Negative slopes among the 42 directed non-self train fits | 10 | Raw source-target closeness can miss inverse relationships. |

The v11/v12 archived validation error maps show that v12's MSE improvement is concentrated in target variable index 4 (the fifth variable; MSE difference −0.14665) and later horizons (steps 49–72: −0.02875; steps 73–96: −0.03730). It is not uniform across variables or missingness buckets. In the diagnostic's source-value proxy, that fifth target variable has a large normalized squared-error tail and the affine source map is slightly worse than the zero baseline there. These proxy errors are not the E23 repair errors or forecast errors and should not be presented as such.

The near-zero inference source-swap effect does not establish why v12 improved over v11: v12 trained a fresh Stage 1 and Stage 2, and its learned parameters could adapt to the contextual selection rule during training. Conversely, the existing v11/v12 comparison does not isolate contextual selection from training trajectory. Therefore the current evidence cannot claim that changing V's top-4 set caused the forecast gain. A new training run that only changes the V ranking formula has weak justification from this diagnostic. The next method experiment should investigate the usefulness of V's transformed message or the H/V/T router, with a retrained, parameter-matched control, before adding another selection heuristic.

## Paper positioning and predictor interface

The defensible research target is a missing-history recovery encoder for forecasting. H/V/T evidence, observed-value preservation, an explicit repaired sequence with the original observation mask, and forecast-driven adaptation define the module. An original forecast head is not required to establish this contribution. Adding a new head now would confound recovery and prediction changes, and the existing PatchTST head should remain the controlled evaluation interface.

The present `E23ForecastModel` and `MaskAwareForecastModel` are still coded against `PatchTSTCore` and patch-level mask projection. The encoder outputs the repaired sequence, original mask and validity, but a second adapter/head has not been tested. The manuscript can presently say the encoder **has a separable repair output and a replaceable-head design goal**. The user plans to add other predictor-head experiments later. Claiming head-agnostic or plug-and-play empirical behavior requires at least one non-PatchTST predictor under the same input/mask protocol, with jointly trained and frozen-encoder transfer settings distinguished. If a second head requires retraining the encoder, call it architecture compatibility rather than zero-shot plug-and-play reuse.

The currently strongest E23 checkpoint remains v12. No v16 model was trained or tested as part of this diagnostic.
