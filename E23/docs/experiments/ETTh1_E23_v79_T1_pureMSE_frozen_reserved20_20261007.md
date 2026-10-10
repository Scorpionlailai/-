# E23-v79 / T1：封存新20组确认

新位置在候选训练前封存，未用于开发或checkpoint选择。先锁定四个场景的MAE-selected权重，再逐组测试；不重训、不根据新位置改配置。

双方Stage2纯MSE、完整冻结修复器、原预测头和同初始SHA、seed2023、Adam lr1e-4、batch128、100轮/6600更新。T1保留原复制补齐，E23保留零补齐。

| 场景 | E23 MSE | T1 MSE | E23 MAE | T1 MAE | MSE / MAE胜出组数 |
|---|---:|---:|---:|---:|---:|
| sr010_bl8 | 0.389033 | 0.388936 | 0.400397 | 0.401594 | 11/20 · 17/20 |
| sr015_bl8 | 0.407524 | 0.411845 | 0.412073 | 0.415469 | 16/20 · 20/20 |
| sr020_bl8 | 0.481826 | 0.502194 | 0.452810 | 0.460352 | 20/20 · 20/20 |
| sr010_bl16 | 0.557570 | 0.582249 | 0.492074 | 0.500389 | 20/20 · 20/20 |

## 原训练资源

| 场景 | 模型 | 两阶段记录成本/分钟 | allocated / reserved GiB |
|---|---|---:|---:|
| sr010_bl8 | E23-v79 | 13.99 | 4.513 / 7.248 |
| sr010_bl8 | T1-pureMSE-frozen | 12.47 | 0.392 / 0.576 |
| sr015_bl8 | E23-v79 | 14.26 | 4.623 / 6.404 |
| sr015_bl8 | T1-pureMSE-frozen | 7.99 | 0.392 / 0.574 |
| sr020_bl8 | E23-v79 | 14.51 | 4.673 / 7.766 |
| sr020_bl8 | T1-pureMSE-frozen | 9.48 | 0.392 / 0.574 |
| sr010_bl16 | E23-v79 | 14.00 | 4.589 / 8.682 |
| sr010_bl16 | T1-pureMSE-frozen | 15.94 | 0.392 / 0.574 |

四场景新位置均值双指标低于T1：**False**。

- 单训练种子；20组标准差反映缺失位置变化，测试时间区间仍为同一ETTh1留出区间。
- 本次确认只做推理。CSV中的时间/显存来自对应原训练记录：E23候选两阶段实测，T1为历史Stage1加本次纯MSE冻结Stage2；同物理RTX5090。
- 相同缺失参数会产生略有不同的实际缺失比例，新组比例单列CSV。
- 权重、配置、mask、逐窗口误差及SHA锁定文件：/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_four_scene_confirmation_reserved_20261007

[CSV](./ETTh1_E23_v79_T1_pureMSE_frozen_reserved20_20261007.csv) · [逐组误差](./ETTh1_E23_v79_T1_pureMSE_frozen_reserved20_20261007_all_draws.csv) · [配对差值](./ETTh1_E23_v79_T1_pureMSE_frozen_reserved20_20261007_paired.csv)
