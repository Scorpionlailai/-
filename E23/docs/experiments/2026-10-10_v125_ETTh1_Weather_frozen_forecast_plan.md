# v125 ETTh1 / Weather 双GPU冻结预测

用户2026-10-10授权：GPU0跑ETTh1四场景，GPU1跑Weather四场景。顺序均0.10/8、0.15/8、0.20/8、0.10/16。现有v125 Stage1不重训，分别加载各自已有同场景的固定1:1组合权重。

Stage2固定100轮、batch128、Adam lr0.0001 wd0、FP32、seed2023、纯MSE、原PatchTST D16/head4/FF128/layers3/P16/S8/drop0.3。完整修复器冻结/eval，不联合训练，无辅助或保留损失，不改预测接口。原预测器初始SHA fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561、35427参数。验证MAE主checkpoint，独立MSE次checkpoint；都保存并在对应原20组测试位置评估。

继续同数据划分/训练标准化/固定验证/原缺失位置。ETTh1原drop_last口径每组2688窗口；Weather完整评估每组10444窗口。训练epoch变化缺失mask不缓存，固定验证采用已验证等价的冻结修复输出cache；记录cache成本和完整资源。

执行任务：

1. 新的隔离实验目录保存plan、组合加载桥、train/eval worker、controller；封存源代码不改。
2. 桥只在本进程显式允许预先固定父组合记录加载并替换修复器构建，原train_forecast与原优化/选模代码复用。仅允许唯一指定父记录，核验组合spec/checkpoint SHA；构建额外repair时保存恢复RNG，预测头初始化和原模型一致。
3. 开跑前CPU preflight：严格组合恢复、原预测头SHA/参数、冻结优化器仅预测头更新、一次反向后repair无grad，所有可见值保持；核验各场景父文件和mask。
4. 每GPU独立顺序完成四场景Stage2及MAE/MSE两份20组预测测试；检查加载后与Stage1修复state逐tensor完全一致，并重放验证。
5. 报告平均预测MSE/MAE及位置标准差、保留全96/近32修复指标、Stage2及完整Stage1+2耗时、allocated/reserved，训练cost去重共用粗与校准。正式对照必须同mask SHA/窗口/坐标数；没有该场景T1结果则注明缺失，不宣称胜出。

不使用测试结果选checkpoint。20组是缺失位置不确定性，不是20独立训练seed。

## 已执行预检与启动

八个场景CPU预检全部通过：严格组合恢复、原预测头SHA/35427参数、冻结优化器仅预测器、repair无grad、一次反向及step后repair state不变、可见值精确保持。新worker首次预检发现预测接口返回E23RepairOutput而非ExplicitRepairResult，已在开跑前按原接口取output.explicit修正并重跑通过；封存模型代码无修改。

2026-10-10 11:44（北京时间）启动，两张GPU已有实际训练batch：ETTh1首场景已进入第3轮，Weather首场景第1轮。每场景完整修复器冻结/eval、纯MSE，训练缺失mask仍逐epoch变化；固定验证修复输出缓存成本计入Stage2资源。

[ETTh1实时状态](/home/xyw/paper/E23/artifacts/performance/v125_forecast_two_datasets_20261010/ETTh1/status.json)

[Weather实时状态](/home/xyw/paper/E23/artifacts/performance/v125_forecast_two_datasets_20261010/Weather/status.json)

[组合加载与训练/评估入口](/home/xyw/paper/E23/artifacts/performance/v125_forecast_two_datasets_20261010/worker.py)

[双队列共用调度](/home/xyw/paper/E23/artifacts/performance/v125_forecast_two_datasets_20261010/controller.py)

每GPU依次四场景；每场景训练100后，分别重放MAE/MSE选模验证并在原20组缺失位置测试，逐tensor核对repair和Stage1相同。每数据集forecast_results.csv逐场景更新，包含预测与修复均值/std、Stage2完整时间、Stage1+2源总成本与allocated/reserved。最终测试时间单独报告，不混入训练成本；资源累计不等于两个GPU并行的墙钟。
