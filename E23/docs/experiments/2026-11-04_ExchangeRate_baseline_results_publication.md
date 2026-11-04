# Exchange-Rate六基线结果归档

按用户2026-11-04指令发布本机已有SAITS、BRITS、CSDI、ImputeFormer、TimesNet、T1的四场景结果。本次未启动新训练，未改选checkpoint。

[四场景结果与口径总览](../../artifacts/performance/ExchangeRate_baselines_20261104/README.md)

- [修复：24组](../../artifacts/performance/ExchangeRate_baselines_20261104/summary/repair_results.csv)
- [预测主表：验证MAE选模24组](../../artifacts/performance/ExchangeRate_baselines_20261104/summary/forecast_results_mae.csv)
- [预测次选：验证MSE选模24组](../../artifacts/performance/ExchangeRate_baselines_20261104/summary/forecast_results_mse_secondary.csv)
- [1440条逐mask指标](../../artifacts/performance/ExchangeRate_baselines_20261104/summary/per_mask_metrics.csv)
- [实际训练轮数、配置、种子及选模](../../artifacts/performance/ExchangeRate_baselines_20261104/summary/training_profiles_and_selection.csv)
- [完整核验](../../artifacts/performance/ExchangeRate_baselines_20261104/completion_audit.json)

每组使用已发布的同20组测试遮挡，各模型沿用原配置单训练seed，不是多训练seed平均。标准差列是mask变化的样本标准差。R3三seed结果已单独归档，本次不把v125旧成绩改称R3，也不混合不同统计口径。

CSDI使用后续预算调整目录的最终结果：两场景Stage1复用200轮，两场景Stage1为100轮；50步扩散，预测训练采样1个、评价5个。BRITS保留TorchScript加速和原早停。实际配置逐项归档，不将旧CSDI未完成结果混入主表。

全模型预测阶段冻结修复器、纯MSE损失、100轮4000更新。已逐项验证选择文件与checkpoint哈希引用、状态冻结、各20组mask SHA、数据SHA、1423个测试窗口，以及均值和样本标准差重算。资源字段保留原batch、精度及硬件一致性记录，不伪装成统一batch/精度下的效率对照。

归档包含原评价JSON、选模文件、运行脚本、源配置、耗时/显存和来源清单。权重只记录路径/哈希，不上传原始数据、权重、优化器状态或预测缓存。
