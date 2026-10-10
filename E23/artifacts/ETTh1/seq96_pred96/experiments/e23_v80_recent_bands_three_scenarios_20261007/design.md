# E23-v80: recent-band level correction

## Hypothesis
Fixed-validation interventions on v79 identify the last 16 points and recent level as the strongest repair contributors to forecasting. Keep its last-32 global-level correction and add independent corrections for positions 64:80 and 80:96. The new branch uses real visible levels from the last 8, last 16, preceding 16 points across seven variables, and coarse repaired levels; no hidden truth or future enters inference.

## Controlled experiment
Only recent_mean_projection changes to global_recent_bands_v3. The branch adds 6,210 parameters and starts at zero; common weights, predictor initialization, observed values and padding remain unchanged. Repair parameters total 430,448. Stage1 uses global MSE, existing retrieval/structure losses, seed 2023, batch 128 and 100 epochs. Stage2 uses pure MSE, full repair freeze, unchanged predictor, seed 2023, batch 128, lr 1e-4, 100 epochs. MAE selects the main checkpoint; MSE selects a separately reported checkpoint from the same trajectory. Three scenes: 0.15/8, 0.20/8, 0.10/16.

## Comparison criteria
Forecast must move toward the matching clean reference (test MSE 0.37530333418130185, MAE 0.39439000007740505). This empirical reference is not a theoretical bound. Repair may retreat relative to earlier E23 versions but must beat T1; report MSE and MAE separately, with global standardized hidden-coordinate validation metrics. Existing T1 repair weights and matched frozen/pure-MSE Stage2 records are reused. No T1 re-training or forecast-head change.

## Verification and reporting
20 targeted integration tests passed before launch. CUDA preflight checks two real repair updates, band gradients, original head initialization and frozen repair state after a forecast update. Lock checkpoints using validation before testing the shared old 20 mask groups. These groups are development results; the previously exposed v79 confirmation bank is not a new blind test. Record both stages, wall time, peak allocated/reserved VRAM and source/checkpoint/mask hashes. Training logs are inspected every 300 seconds.
