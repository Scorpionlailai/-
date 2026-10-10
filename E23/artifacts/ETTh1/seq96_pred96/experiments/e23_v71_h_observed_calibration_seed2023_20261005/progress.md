# E23-v71 H observed calibration

Approved task: improve H candidate quality/confidence based on fixed v69; keep forecaster and Stage2 recipe. Bounded implementation choices: offset=sum(visible residual)/(n+4), clipped to two robust local scales; confidence from exact observed leave-one-out offsets. Single73% scene,seed2023,100+100epochs.
Task1 isolated exact517-source copy complete. Task2 behavioral RED/GREEN tests and config/identity wiring. Task3 focused/full regression and CUDA preflight. Task4 fixed run and20-mask test after checkpoint locks. Task5 accuracy/cost/repair report and independent review.
Task2 complete: four orchestration checks RED→GREEN; one actual CUDA scenario preflight passed, identical head, finite losses, full repair freeze; no test scoring.
2026-10-05T14:28:42.093650+00:00 Task3 start sr015_bl8 stage1
2026-10-05T14:38:11.927643+00:00 Task3 start sr015_bl8 stage2
2026-10-05T14:43:00.160137+00:00 Task3 start sr015_bl8 test
2026-10-05T14:43:13.306059+00:00 Task3 start sr015_bl8 test
2026-10-05T14:43:26.433873+00:00 Task3 start sr015_bl8 diagnostics
2026-10-05T14:43:29.997873+00:00 Task3 complete sr015_bl8; both checkpoints locked before tests, frozen repair,20 paired positions.

2026-10-05T14:46:42.041302+00:00 Task4 reports published: 3 main,60 draw,40 paired,2 selection,2 repair-diagnostic rows. Tests locked before scoring; MAE/MSE both selected epoch100,289 model tensors equal. Validation gate rejects promotion (MSE improves,MAE +0.0000477237); fixed v69 retained. Final focused checks11 passed. Task5 independent artifact review pending.
2026-10-05T14:49:33.824507+00:00 Task5 complete: independent trained-artifact review found no Critical/Important/Minor issues; fresh delivery verification passed1883 immutable locks, all report/checkpoint hashes, one new Stage1/Stage2 and zero T1 runs. Reports and weights archived, v69 retained by fixed validation gate, v71 preserved as exploratory candidate.
