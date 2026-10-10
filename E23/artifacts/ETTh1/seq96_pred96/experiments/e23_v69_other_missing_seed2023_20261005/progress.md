# Experiment ledger — plan: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v69_other_missing_seed2023_20261005/plan.json

User approved the scenario expansion from the prior summary.
Task1 complete: isolated exact v69 source copy and fixed three-scene plan; cached T1 checkpoints/banks verified.
Task2 pending: scenario preflight, head identity, finite recovery gradients, complete repair freeze.
Task3 pending: three sequential 100+100 epoch training trajectories; checkpoint locks before tests.
Task4 pending: paired 20-position results, recovery diagnostics, full cost CSV and final audit.
Ruling: use existing frozen folder copies, not source edits or new git branches; experiment-only configuration.
Ruling: main table uses MAE-selected checkpoint; secondary MSE is separately labelled, no test-based choice.
Ruling: cached T1 workflow times retain original device records; low-missing cached T1 stages used different physical GPUs.
Task2 complete: four orchestration checks RED→GREEN; three actual CUDA scenario preflights passed, identical head, finite losses, full repair freeze; no test scoring.
2026-10-05T09:15:42.927467+00:00 Task3 start sr010_bl8 stage1
Ruling correction: cached lower-missing T1 resource device UUID is GPU1 in BOTH stages. hardware_consistent=False is due to batch16/128 difference (resources.py also compares batches), not different physical GPUs. Report physical-device consistency separately and preserve the raw flag.
2026-10-05T09:25:03.809465+00:00 Task3 start sr010_bl8 stage2
2026-10-05T09:29:43.911903+00:00 Task3 start sr010_bl8 test
2026-10-05T09:29:56.868247+00:00 Task3 start sr010_bl8 test
2026-10-05T09:30:09.828131+00:00 Task3 start sr010_bl8 diagnostics
2026-10-05T09:30:13.318031+00:00 Task3 complete sr010_bl8; both checkpoints locked before tests, frozen repair,20 paired positions.
2026-10-05T09:30:13.333089+00:00 Task3 start sr020_bl8 stage1
Task4 partial complete: first scenario MAE primary MSE0.3862432559060942/MAE0.39974427525959316 vs T1 0.3889840393920604/0.417673173350812; paired wins11/20 and20/20; selected epoch98. Report verified 214 frozen repair tensors for both selection manifests.
Ruling: cached low-missing reference Stage1 repair-MSE field is absent in source CSV; preserve missing value rather than coercing an empty string or inventing a score. Publishing failure fixed without training changes.
2026-10-05T09:39:46.632274+00:00 Task3 start sr020_bl8 stage2
2026-10-05T09:44:36.221216+00:00 Task3 start sr020_bl8 test
2026-10-05T09:44:49.083512+00:00 Task3 start sr020_bl8 test
2026-10-05T09:45:01.956109+00:00 Task3 start sr020_bl8 diagnostics
2026-10-05T09:45:05.441739+00:00 Task3 complete sr020_bl8; both checkpoints locked before tests, frozen repair,20 paired positions.
2026-10-05T09:45:05.456874+00:00 Task3 start sr010_bl16 stage1
2026-10-05T09:54:25.308288+00:00 Task3 start sr010_bl16 stage2
2026-10-05T09:59:02.784397+00:00 Task3 start sr010_bl16 test
2026-10-05T09:59:15.745749+00:00 Task3 start sr010_bl16 test
2026-10-05T09:59:28.703107+00:00 Task3 start sr010_bl16 diagnostics
2026-10-05T09:59:32.166989+00:00 Task3 complete sr010_bl16; both checkpoints locked before tests, frozen repair,20 paired positions.
Task3 complete: all three fresh Stage1/Stage2 trajectories finished100/100 epochs; all MAE/MSE selections locked before tests; no baseline retraining.
Task4 complete: nine new-scene comparison rows,180 draw rows,60 paired rows,six selection rows;12-row four-scene aggregate and240 draws. Source517/protected612 and T1 provenance checked.
Final scope: original v69 remained unchanged; one training seed per method, shared20 test positions in each scene, same test time interval. Independent read-only review pending.
Independent final review complete: no Critical/Important findings;4/4 CPU controller tests. All source/reference/selection/mask/paired-stat/resource checks passed. Minor review-request summary corrected: long-block MAE epoch96 vs MSE epoch58; delivered selections were already accurate. See final-review.md.
Delivery complete: all three fixed-recipe comparisons, four-scenario aggregate, draw/paired/selection CSVs and checkpoint index saved. No further experiment started.
