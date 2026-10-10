# E23 v12 expert contribution on fixed ETTh1 P4 validation

Read-only diagnostic on the selected v12 checkpoint. Each variant masks one expert's availability **before the router softmax** and leaves all weights, the fixed H memory/key, the input masks, the predictor, and the other experts unchanged. All rows use the same 2,688 validation windows and 1,806,336 future coordinates. The full row exactly reproduces the selected manifest. No test evaluation or retraining was performed.

| Variant | Forecast MSE ↓ | Δ MSE vs full | Forecast MAE ↓ | Δ MAE vs full |
|---|---:|---:|---:|---:|
| Full H/V/T | 0.745448 | — | 0.588212 | — |
| Without H | 0.834571 | +0.089123 | 0.616441 | +0.028229 |
| Without V | 0.796146 | +0.050698 | 0.620341 | +0.032129 |
| Without T | 0.765649 | +0.020202 | 0.598615 | +0.010403 |
| Without all three | 0.941697 | +0.196249 | 0.666856 | +0.078643 |

On this frozen model, T has the smallest marginal contribution to **both** forecast metrics. H has the largest MSE effect, while V has the largest MAE effect. T still helps: removing it worsens MAE by `0.010403`. Its validation MAE contribution is about `0.012441`, `0.010680`, `0.009609`, and `0.008883` over forecast steps 1–24, 25–48, 49–72, and 73–96. Thus the marginal T effect decreases slightly with horizon.

This diagnostic ranks the current model's dependency on each expert, not the quality of retrained architectures. The router reallocates weight when an expert is masked, and overlapping evidence can make a useful expert look marginal. The planned next controlled model trial should target T's evidence selection within each temporal band while retaining v12 H, V, router, repair width 64, training objective and predictor. Current T takes one source per directional band, with local slots chosen by nearest position and distant slots chosen by latent similarity; visible coverage and observed agreement enter mainly after this selection. A train-only or current-window observed-evidence score before band selection is a concrete hypothesis. Predeclare one scoring rule and use the same validation MAE improvement plus no-MSE-regression gate before any new test.

Artifact: `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v3_window_v12/benchmarks/v12_validation_expert_ablations.json` (SHA-256 `18df0b9cf9826aa8bc181219a804f1b591dd611ce4f03845ccd34a16e5501e18`). Script: `scripts/diagnose_e23_v12_experts.py` (SHA-256 `7d779d83bef716340150410e36294dd050b92bf6469428e2f696c08f82228388`). Selected Stage-2 checkpoint SHA-256 `c3ae09195b793b144936f38ceba0909f377d7fb02cfea626689fb21e70af3c35`; H key SHA-256 before/after `db6b8c905b6b4074160815a370973a2d76ed3bff5e30fa61d22dabafcd2ca396`.
