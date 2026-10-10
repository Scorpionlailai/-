# E23-v99 局部混合精度试验结果（2026-10-08）

## 配置与比较范围

采用 [预先固定的设计](/home/xyw/paper/E23/docs/experiments/2026-10-08_E23_v99_selective_precision_plan.md)：v97 原64维八头结构无参数压缩；仅 Stage1 训练的窗口编码/候选读出启用 BF16，并编译候选读出。所有验证/测试、历史查询、损失、权重、Adam 以及 Stage2 均保持原精度。Stage2 整个修复器冻结、原PatchTST、纯MSE、启用固定验证缓存。

单场景 ETTh1，seq96/pred96，起点率0.15/块长8，seed2023/batch128；粗修复、细化和预测各100轮。粗修复选第96轮、细化选第18轮，但三阶段均完整跑满100轮。20组测试缺失位置沿用开发实验，非盲测、非训练多种子。

## 修复测试20组均值

| 指标 | v97 | v99 | T1 已固定参考 |
|---|---:|---:|---:|
| all_hidden MSE | 0.176235 | 0.174876 | 0.193593 |
| all_hidden MAE | 0.264315 | 0.263794 | 0.280481 |
| last32_hidden MSE | 0.180031 | 0.181054 | 0.196546 |
| last32_hidden MAE | 0.269464 | 0.270642 | 0.282877 |

FP32验证全窗口 hidden MSE/MAE = 0.242310/0.311240；近32点 = 0.268413/0.326514。正式checkpoint按验证全hidden MSE锁定，测试未用于选epoch。

## 冻结修复器后的预测测试

| 模型/选模 | MSE | MAE |
|---|---:|---:|
| v97 验证MAE | 0.398613 | 0.407094 |
| v99 验证MAE（主结果） | 0.398394 | 0.407443 |
| v99 验证MSE（副结果） | 0.398755 | 0.407593 |
| T1 冻结纯MSE已固定参考 | 0.408589 | 0.414075 |

主结果比v97两项更好：False；比T1两项更好：True。缺失位置标准差在结果CSV。预测头初始SHA与原v97一致，Stage2修复可训练参数为0。

## 包含父粗修复的完整训练资源

| 模型 | 完整训练分钟 | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|
| v97 | 19.483 | 4.623 | 6.404 |
| v98 | 17.816 | 4.371 | 6.162 |
| v99 | 17.011 | 4.074 | 5.834 |

v99 相比v97完整时间减少 12.69%、峰值allocated减少 11.87%。这是单次、GPU0实测；包含所有三阶段准备、训练、验证、首次编译和保存，不包含最终测试及独立微测。v99时间收益同时包含局部混合精度、编译和Stage2验证缓存，不能归因于一个模块。自动 evaluation JSON 聚合的训练资源只包含细化与预测，不能用作总成本；正式引用 [complete_training_cost.json](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/complete_training_cost.json)。T1本轮未用同样执行优化重跑，不据此比较架构效率。

## 查验与后续

- [精度CSV](/home/xyw/paper/E23/docs/experiments/E23_v99_selective_precision_results_20261008.csv)
- [资源CSV](/home/xyw/paper/E23/docs/experiments/E23_v99_selective_precision_resources_20261008.csv)
- 所有运行命令、模型权重、散列冻结源码及原始指标：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008`。

只完成一个场景，尚无其余三个场景或完整多变量训练结果。保留v97的四场景证据；BF16改变训练数值，需要后续数据集验证。E24运行源码和封存v79均未修改。
