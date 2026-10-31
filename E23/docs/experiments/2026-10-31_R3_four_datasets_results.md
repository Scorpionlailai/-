# R3四数据集正式修复结果归档

2026-10-31：ETTh1、BeijingAirQuality、PhysioNet2012、Exchange-Rate四场景×三训练seed全部完成，共48个模型。

[实验结果、资源表与审核记录](../../artifacts/performance/R3_final_four_datasets_3seed_20261031/README.md) · [最终full96结果CSV](../../artifacts/performance/R3_final_four_datasets_3seed_20261031/summary/R3_four_datasets_final_full96.csv)

模型固定完整R3/H_GLOBAL，主干100轮、完整粗模型冻结后校准100轮；按既定验证指标选模，每seed评价20组原基线一致遮挡。没有运行预测Stage2。Exchange校准退步和种子波动均保留；不能仅展示有利种子。

本次上传结果与代码/协议证据，不上传原始数据、权重或预测缓存。
