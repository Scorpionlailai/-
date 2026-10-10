# E23 Stage1修复迭代（GPU0）

更新时间UTC：2026-10-07T14:38:25.987803+00:00；状态：preset_trials_completed_need_next_design。Weather使用GPU1，本队列不修改E24。

First sr015_bl8; continue other 3 scenes only if full and last32 validation MSE/MAE all improve v79. Screening never uses test metrics.

Stage1固定100轮/seed2023/batch128。只按验证全局修复MSE选权重；20组旧测试仅作开发评估，未执行Stage2。

| 模型 | 场景 | 状态 | 验证全窗口 MSE/MAE | 验证末32 MSE/MAE | 测试20组全窗口 MSE/MAE | 测试20组末32 MSE/MAE |
|---|---|---|---:|---:|---:|---:|
| E23-v85 | sr015_bl8 | repair_audited | 0.331218 / 0.364225 | 0.336004 / 0.366175 | 0.240075 / 0.315433 | 0.244225 / 0.319389 |
| E23-v85 | sr010_bl16 | queued | — / — | — / — | — / — | — / — |
| E23-v85 | sr020_bl8 | queued | — / — | — / — | — / — | — / — |
| E23-v85 | sr010_bl8 | queued | — / — | — / — | — / — | — / — |
| E23-v86 | sr015_bl8 | repair_audited | 0.327339 / 0.361286 | 0.341878 / 0.365726 | 0.225460 / 0.304718 | 0.230776 / 0.310645 |
| E23-v86 | sr010_bl16 | queued | — / — | — / — | — / — | — / — |
| E23-v86 | sr020_bl8 | queued | — / — | — / — | — / — | — / — |
| E23-v86 | sr010_bl8 | queued | — / — | — / — | — / — | — / — |
| E23-v87 | sr015_bl8 | repair_audited | 0.320167 / 0.356425 | 0.334285 / 0.361232 | 0.229611 / 0.304433 | 0.233539 / 0.307112 |
| E23-v87 | sr010_bl16 | queued | — / — | — / — | — / — | — / — |
| E23-v87 | sr020_bl8 | queued | — / — | — / — | — / — | — / — |
| E23-v87 | sr010_bl8 | queued | — / — | — / — | — / — | — / — |

资源详见同名CSV与stage1/resources；两张卡同时运行，CPU/磁盘可能共享，耗时不作同卡独占效率结论。
