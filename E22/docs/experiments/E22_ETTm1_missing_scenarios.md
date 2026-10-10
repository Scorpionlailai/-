# ETTm1 E22 missing-scenario results

All five experiments use ETTm1, `seq96 -> pred96`, seed 2021 and metrics in the training-standardized space. The shared Clean test result is MSE `0.320842756059`, MAE `0.359332665076`.

| Start rate / block length | Actual missing rate | Missing MSE / MAE | E22 MSE / MAE | DRR-MSE / DRR-MAE | E22 evaluation |
|---|---:|---:|---:|---:|---|
| `0.10/3` | 27.14% | 0.347349 / 0.379665 | 0.331809 / 0.368264 | 58.63% / 56.07% | Zero-shot |
| `0.10/5` | 40.93% | 0.368345 / 0.392402 | 0.334266 / 0.369290 | 71.74% / 69.89% | Zero-shot |
| `0.10/8` | 56.85% | 0.437194 / 0.429110 | 0.363905 / 0.385846 | 62.99% / 62.00% | Trained on this protocol |
| `0.15/8` | 72.46% | 0.520699 / 0.464993 | 0.432652 / 0.419700 | 44.06% / 42.87% | Zero-shot |
| `0.20/8` | 83.12% | 0.646878 / 0.511563 | 0.581539 / 0.478279 | 20.04% / 21.86% | Zero-shot |

DRR is calculated separately for each scenario and metric: `(Missing - E22) / (Missing - shared Clean) * 100`. Every scenario has its own trained Missing baseline and fixed test mask. E22 is trained only on the main `0.10/8` scenario; the other four E22 results evaluate that same checkpoint without additional training. The source Stage 2 run ID is `ETTm1-ours-be0375037c72` and its checkpoint SHA-256 is `59e9fede9f98739b26f99f1241ef54ccd8e78045805d92af549a4b5817ecaeaf`.

The Excel and CSV files use a shared Clean row at the top and then show the Missing and E22 rows for each scenario. Their `Source JSON` column links each metric back to its evaluation record. The five test mask hashes, target configuration hashes and source checkpoint hash were checked against the recorded results before this summary was produced.

Observed trend: the E22 model improves both MSE and MAE over Missing in all five scenarios. DRR falls at the two highest missing pressures, from roughly 63% in the training scenario to 20–22% at `0.20/8`; this is evidence of reduced recovery under severe missingness, while the zero-shot benefit remains positive.
