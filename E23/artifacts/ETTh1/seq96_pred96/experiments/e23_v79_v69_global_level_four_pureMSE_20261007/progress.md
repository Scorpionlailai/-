# E23-v79 preparation

- Single configured change from v69: global-level mean compensation. No gap-moment loss or auxiliary forecast. Pure MSE frozen Stage2; four identical model configs.
- Candidate preparation only; training starts after pure-MSE baseline is complete and confirms improvements are needed.
- Existing feature checks freshly run: 15 cross-variable-global-level/recent-mean tests passed; byte-identical source copied to candidate. No candidate training launched.
2026-10-07T05:30:40.623535+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/preflight.log
2026-10-07T05:40:02.119917+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl8/E23-v79/stage1.log
2026-10-07T05:44:40.928162+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl8/E23-v79/stage2.log
2026-10-07T05:44:53.969745+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl8/E23-v79/evaluation_mae.log
2026-10-07T05:45:06.999559+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl8/E23-v79/evaluation_mse.log
- First scene sr010_bl8 complete: Stage1 repair global MSE 0.1695992503 (epoch91); Stage2 pure-MSE frozen MAE-selected epoch98, validation MSE/MAE 0.7441835187/0.5719284817, old20 test MSE/MAE 0.3859191942/0.3988386969. T1 pure frozen 0.3869915499/0.4004140219; paired wins 14/20 MSE and 18/20 MAE. Fresh two-stage measured cost 839.4554s, allocated/reserved 4.5131/7.2480GiB. Both selection metrics picked epoch98. Source/checkpoint/replay/masks/resource records checked.
- Fixed validation is slightly worse than parent v69 in first scene; development test is better. Retain full four-scene evaluation before a shared-candidate ruling. Reserved new20 each scene have not been evaluated: e23_four_scene_confirmation_reserved_20261007/reserved-lock.json.
2026-10-07T05:54:39.112708+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr015_bl8/E23-v79/stage1.log
2026-10-07T05:59:23.541650+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr015_bl8/E23-v79/stage2.log
2026-10-07T05:59:36.601130+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr015_bl8/E23-v79/evaluation_mae.log
2026-10-07T05:59:49.724659+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr015_bl8/E23-v79/evaluation_mse.log
2026-10-07T06:09:28.059255+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr020_bl8/E23-v79/stage1.log
2026-10-07T06:14:21.299015+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr020_bl8/E23-v79/stage2.log
2026-10-07T06:14:34.277975+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr020_bl8/E23-v79/evaluation_mae.log
2026-10-07T06:14:47.272448+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr020_bl8/E23-v79/evaluation_mse.log
- First3 scenarios complete, shared v79 config: all3 development20 dual-metric wins versus matched frozen pure-MSE T1. sr015_bl8 test0.4053804638/0.4113107711, 14/20 MSE and20/20 MAE wins; full855.6991s,4.6231/6.4043GiB. sr020_bl8 test0.4755418979/0.4497960603,20/20 both; full870.7223s,4.6729/7.7656GiB. All6 candidate selections, validation replay, windows, resources and Stage1 parent hashes checked. Fourth scene running; confirmation masks remain unscored.
2026-10-07T06:24:09.060214+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl16/E23-v79/stage1.log
2026-10-07T06:28:47.996775+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl16/E23-v79/stage2.log
2026-10-07T06:29:01.065309+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl16/E23-v79/evaluation_mae.log
2026-10-07T06:29:14.124473+00:00 complete /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007/sr010_bl16/E23-v79/evaluation_mse.log
