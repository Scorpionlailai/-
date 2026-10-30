# R3 最终定版

2026-10-30 用户选定：完整 R3 / H_GLOBAL。

代码与多服务器六数据集运行说明见 [E24/versions/r3_20261030/README.md](E24/versions/r3_20261030/README.md)。

- 发布分支：`release/r3-final-20261030`；每台服务器应固定本次发布 commit。
- 原 R2 分支和历史实验不变。定版模型保留学习式 Confidence、候选目标上下文和单支64后端。
- 六数据集：ETTh1、Weather、Exchange-Rate、BeijingAirQuality、PAMAP2、PhysioNet2012。
- 数据、checkpoint不上传；各数据集先做 selfcheck 和 smoke，再正式四场景训练。
- 当前默认单训练 seed2023；正式多种子另行统一。20组测试mask与训练seed分开统计。
- 本次发布不代表六数据集已经重跑完成，不包含预测Stage2。
