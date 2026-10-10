# E23-v80 toward clean

- Exactly one configured repair change from v79; per-band levels after existing point/expert/global-level repair. Additional6210params, total430448. Same head initialization/architecture.
- Stage1/Stage2 each100epochs,6600updates,seed2023/batch128. Stage2 pureMSE/full repair state freeze. ValidationMAE main/MSEsecondary; old20 development test after both locks.
- Same three scenes .15/8,.20/8,.10/16. Compare both global Stage1 metrics to fixed FP32 T1 inference. Parent may retreat but must beat T1. Comparable clean reference0.3753033342/0.3943900001.
- Meaningful feature checks:20 targeted tests passed. Real CUDA preflight required before formal. Monitor formal logs/results every300seconds per user.
2026-10-07T07:40:03.595769+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/preflight.log
2026-10-07T07:49:44.826650+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr015_bl8/E23-v80/stage1.log
2026-10-07T07:54:39.505448+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr015_bl8/E23-v80/stage2.log
2026-10-07T07:54:52.987756+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr015_bl8/E23-v80/evaluation_mae.log
2026-10-07T07:55:06.469632+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr015_bl8/E23-v80/evaluation_mse.log
2026-10-07T08:04:52.691201+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr020_bl8/E23-v80/stage1.log
2026-10-07T08:09:52.871890+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr020_bl8/E23-v80/stage2.log
2026-10-07T08:10:06.274871+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr020_bl8/E23-v80/evaluation_mae.log
2026-10-07T08:10:19.693350+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr020_bl8/E23-v80/evaluation_mse.log
2026-10-07T08:19:52.773436+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr010_bl16/E23-v80/stage1.log
2026-10-07T08:24:36.774294+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr010_bl16/E23-v80/stage2.log
2026-10-07T08:24:50.273295+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr010_bl16/E23-v80/evaluation_mae.log
2026-10-07T08:25:03.769161+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v80_recent_bands_three_scenarios_20261007/sr010_bl16/E23-v80/evaluation_mse.log
