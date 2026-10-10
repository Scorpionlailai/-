# ETTh1 P4 exploratory single-seed forecast comparison

| Model | MSE | MAE | DRR-MSE % | DRR-MAE % |
| --- | ---: | ---: | ---: | ---: |
| PatchTST Clean | 0.391587 | 0.406349 | 100.000000 | 100.000000 |
| PatchTST Missing | 0.519669 | 0.466895 | 0.000000 | 0.000000 |
| E22 | 0.443195 | 0.433248 | 59.706940 | 55.572312 |
| SAITS+Predictor | 0.440584 | 0.436958 | 61.745540 | 49.445248 |
| S4M | 0.985358 | 0.753972 | -363.586714 | -474.147929 |
| S4M Clean | 1.145705 | 0.845474 | — | — |

Exploratory single-seed comparison; ETTh1 P4 test set participated in earlier E22 development.

S4M DRR uses the shared PatchTST references only; S4M Clean is auxiliary.
