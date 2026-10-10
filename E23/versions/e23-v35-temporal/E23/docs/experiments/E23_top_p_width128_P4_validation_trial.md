# E23 H/V Top-p and Width 128: ETTh1 P4 Validation Trial

Date: 2026-09-30. The objective was to improve forecast accuracy over selected E23 v3. The comparison changed historical (H) and current-window variable (V) candidate selection from fixed top-k to nucleus top-p, enlarged repair width from 64 to 128, and tested their combination. Temporal (T), the flat `none/H/V/T` router, explicit repair, PatchTST predictor width 16, ETTh1 `96→96`, patch 16/stride 8, fixed masks and training recipe were held fixed. All new runs used seed 2021, batch 128, 100 Stage 1 epochs and 100 Stage 2 epochs. Stage 1 selected earliest lowest hidden-coordinate validation repair MSE; Stage 2 trained forecast MSE and selected earliest lowest validation forecast MAE.

## Predeclared decision and results

The gate was validation MAE **strictly below 0.5904396656148828** and validation MSE **no higher than 0.7647695325129157**, both from E23 v3. At most one passing variant would receive a selected ETTh1 test and test-input profile. None passed, so no new test or profile was run. E23 v3 remains the selected model; its prior test MSE/MAE are 0.4272647293246522/0.43444857629788974. The new rows below are **validation**, not test, results.

| Repair width / candidate rule | Stage 1 selected epoch / repair MSE | Stage 2 selected epoch | Validation MSE | Validation MAE | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 64 / top-k (v3, existing) | Existing selected Stage 1 | 87 | 0.764769533 | **0.590439666** | Reference |
| 64 / H+V top-p 0.9 (v8) | 99 / 107.141909 | 87 | 0.763754851 | 0.590740494 | MAE misses by 0.000301 |
| 128 / top-k (v9) | 99 / 107.020833 | 58 | 0.770957069 | 0.596344020 | Both metrics miss |
| 128 / H+V top-p 0.9 (v10) | 99 / 107.028029 | 18 | **0.753857477** | 0.596780276 | MAE misses by 0.006341 |

The Stage 1 repair MSE and Stage 2 forecast MSE use different targets and denominators, so the Stage 1 numbers are not directly comparable with the forecast scores. V8 and v10 improve validation forecast MSE, but their selected MAE is higher than v3. The width increase alone worsened both selected forecast metrics in this one-seed trial.

## Candidate behavior on fixed validation inputs

The nucleus is the shortest stable sorted prefix reaching 0.9 softmax mass among eligible candidates. H first retrieves at most eight members from the best causal same-variable prototype group, scores them by key similarity minus visible-coordinate error, then applies top-p. V applies top-p to the directed prior over six other variables; Stage 1 may append one rotating source for exploration, while validation uses only the nucleus. Invalid and unselected slots have zero final weight. Top-k retains four H and four V sources. T retains ten directional/distance slots in every row.

The following means count **valid candidates per damaged validation patch** (203,715 patch-variable positions). V counts local source-offset edges, up to three per selected source. H and V availability is the fraction of damaged positions with at least one valid candidate.

| Trial | H mean | V mean | T mean | H availability | V availability | V selected sources by target variable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8, 64/top-p | 5.701 | 6.541 | 4.746 | 88.074% | 99.757% | 2, 4, 1, 3, 3, 3, 2 |
| v9, 128/top-k | 3.237 | 10.210 | 4.746 | 88.066% | 100.000% | 4, 4, 4, 4, 4, 4, 4 |
| v10, 128/top-p | 5.507 | 7.290 | 4.746 | 88.066% | 99.973% | 3, 4, 2, 3, 3, 3, 2 |

Top-p selected more H prototypes and fewer V source edges than top-k in the width-128 pair. The H 0.9 probability mass was often spread across seven or eight retrieved members. This is an observed mechanism, not proof that the larger H set caused the MAE difference. The V prior stayed directed and excluded self; the unchanged T counts corroborate that its selection rule was not modified.

## Provenance and resources

All three runs used source CSV SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`, validation mask SHA-256 `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c`, and code revision `44241f583a686c0a9265cf646a219b801e4019fb`. The fixed test mask is carried in each manifest but was not read for a new evaluation. Every Stage 2 loader verified its Stage 1 manifest/checkpoint, source/config/mask identity and frozen H memory/key hashes; validation diagnostics confirmed H state unchanged. Full epoch trajectories and job exit codes are in each ignored `jobs/` directory. All six training jobs exited 0 and completed 100 epochs per stage.

| Trial | Config SHA-256 | Stage 1 checkpoint SHA-256 | Stage 2 checkpoint SHA-256 | Stage 1 / Stage 2 seconds |
| --- | --- | --- | --- | ---: |
| v8 | `5f4454739f6f1c837ef0fab2699f4d77068b330dc58764dbb7659618285e3704` | `ac6430d11e70e0583308754801ebb740d9e7e62e7773bf306f515be0ca9edfdd` | `9000e7170dacd67edc53c0966e397b4baccd171b15398135be4845ce06fc11ed` | 505 / 1473 |
| v9 | `3d19d153992bf7b5bbc4e923a49b60eb86db3624cad98f4f25ef8ef19ee27ef0` | `07d808720d05836d308fbfdafbf1f429ce20adc7800cb4a585449157425e3923` | `5524e84ba8a8aaa15f6cf8c46a139988988dbdf186e2d50e3a6b556c19fbfa31` | 521 / 1448 |
| v10 | `c0cc8b9852749dd4a77525dc84b5d2559655ccbc519df01a809a80be17070ae3` | `8bc8d105eb538cbd51043808717c11e39d803ee087235889145c7ce674a54eef` | `ca3b77431dab053aeba691a1f90ff797e48d4e0d497008f62bd0a2383626d00d` | 1464 / 193 |

Artifact roots are `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_nucleus_d64_topp_v8/`, `e23_nucleus_d128_topk_v9/` and `e23_nucleus_d128_topp_v10/`. Each contains `stage1_p4/runs/e23-stage1.json`, `stage2_p4/runs/e23-stage2.json`, paired checkpoints, job logs and `benchmarks/validation_candidate_diagnostics.json`. Repair parameters are 113,241 at width 64 and 423,065 at width 128; the predictor remains 35,427 parameters. A real P4 batch-128 CUDA forward/backward smoke for 128/top-p had finite loss and decoder gradients and peak **allocated** memory 1,135,501,312 bytes. **Protocol deviation:** full-training peak GPU memory was not instrumented and cannot be recovered from the completed jobs; this smoke value must not be treated as a training peak. Wall times above include concurrent GPU jobs and are not controlled speed comparisons.

After training, review found a floating-point boundary case in the general nucleus helper at other probability thresholds. The exclusive cumulative mass calculation was corrected and covered by a failing-then-passing regression test. Replaying the selected v8 and v10 checkpoints across all fixed validation batches with the training-time and corrected helper produced exactly equal MSE/MAE pairs, so the recorded accuracy decision is unchanged. The run manifests retain their original code revision above.

## Interpretation and limits

V8 missed the MAE gate narrowly, so it is worth keeping as a research lead, but the predeclared rule retains v3. V10's lower MSE with substantially higher MAE suggests fewer large errors alongside broader absolute error; this is an inference from the paired metrics and needs error-distribution analysis before a mechanism claim. Each variant used one seed on ETTh1, and earlier E23 design decisions already used ETTh1 feedback. No claim of generalization or test improvement follows from these validation outcomes. Any subsequent architectural trial should be registered before using the ETTh1 test again.
