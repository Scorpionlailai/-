# Pure MSE frozen Stage2: baseline before v69 iterations

Task1: isolate code and freeze 4-scene identity; both methods use pure MSE, original heads, frozen repair, same Stage2 seed2023/Adam lr1e-4/batch128/100 epochs and MAE/MSE checkpoint tracking.
2026-10-07T04:58:07.707137+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/preflight.log
2026-10-07T05:00:57.377354+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/T1/sr010_bl8/training.log
- Task1 preflight complete: 8 real CUDA checks across 4 scenarios, objective exactly MSE, repair unchanged, identical head SHA. Fresh read-only reviewer found no critical/important issue; minor archive note in review-ledger.md.
2026-10-07T05:05:36.540236+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr010_bl8/stage2.log
2026-10-07T05:05:49.516115+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr010_bl8/evaluation_mae.log
2026-10-07T05:06:02.463830+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr010_bl8/evaluation_mse.log
2026-10-07T05:08:58.648024+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/T1/sr015_bl8/training.log
2026-10-07T05:13:39.983797+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr015_bl8/stage2.log
2026-10-07T05:13:52.952542+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr015_bl8/evaluation_mae.log
2026-10-07T05:14:05.932931+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr015_bl8/evaluation_mse.log
2026-10-07T05:17:08.256505+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/T1/sr020_bl8/training.log
2026-10-07T05:21:56.802230+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr020_bl8/stage2.log
2026-10-07T05:22:09.708436+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr020_bl8/evaluation_mae.log
2026-10-07T05:22:22.618493+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr020_bl8/evaluation_mse.log
2026-10-07T05:25:11.551792+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/T1/sr010_bl16/training.log
2026-10-07T05:29:50.852682+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr010_bl16/stage2.log
2026-10-07T05:30:03.833171+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr010_bl16/evaluation_mae.log
2026-10-07T05:30:16.792270+00:00 completed /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007/E23-v69/sr010_bl16/evaluation_mse.log
- Task1 complete: 8 Stage2 runs/100 epochs, all 16 MAE/MSE evaluations verified. No all-four dual-metric win; proceed to single shared v79 repair branch. Matched pure-MSE report saved.
