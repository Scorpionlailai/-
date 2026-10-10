# E23-v72 long-gap moments

Approved: prioritize long continuous missingness. One Stage1 long-gap moment loss experiment on existing block16 scenario. v69 architecture, original head and Stage2 frozen recipe.
Task1 isolated517sources copied, source/old artifacts protected. Task2 RED/GREEN loss and integration tests. Task3 full regression+CUDA preflight+source freeze. Task4 one100+100 training and locked20-mask primary/secondary tests. Task5 reports/independent review/fresh verification.
Task2 complete: four orchestration checks RED→GREEN; one actual CUDA scenario preflight passed, identical head, finite losses, full repair freeze; no test scoring.
2026-10-05T15:23:03.817060+00:00 Task2/3 checks:8new behavioral RED→GREEN,4controller RED→GREEN,real populated-bank CUDA preflight passed. Full1152pass/9known failure names/1skip, archived detailed test-audit. Formal run awaiting implementation review.
Task2 complete: four orchestration checks RED→GREEN; one actual CUDA scenario preflight passed, identical head, finite losses, full repair freeze; no test scoring.
2026-10-05T15:29:41.494169+00:00 Task3 start sr010_bl16 stage1
2026-10-05T15:39:06.109834+00:00 Task3 start sr010_bl16 stage2
2026-10-05T15:43:42.693976+00:00 Task3 start sr010_bl16 test
2026-10-05T15:43:55.791234+00:00 Task3 start sr010_bl16 test
2026-10-05T15:44:08.936381+00:00 Task3 start sr010_bl16 diagnostics
2026-10-05T15:44:12.509629+00:00 Task3 complete sr010_bl16; both checkpoints locked before tests, frozen repair,20 paired positions.

2026-10-06T01:09:40.605262+00:00 Task5 complete: independent final-review.md has no remaining Critical/Important/Minor; fresh delivery-verification.json passed 2417 unique frozen paths, six CSV inventories and both 214-tensor repair freeze checks.
Final gates supersede earlier pre-fix progress counts: 9 new behavioral +4 controller checks passed; complete regression 1153 passed/9 inherited failures/1 skipped.
One Stage1 +one Stage2, 100+100 epochs, seed2023; no baseline retraining. Primary MAE epoch94 test MSE/MAE=0.5615661518914359/0.49878249595495044; secondary MSE epoch95=0.5648511231351061/0.5027809115494188. Retained long-scene candidate by fixed validation gate; test not used for promotion, T1 remains better, original E23 source not integrated. Reports archived in docs/experiments/ETTh1_E23_v72_long_gap_moments_seed2023_20261005*.
