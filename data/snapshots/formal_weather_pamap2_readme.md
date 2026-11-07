# R3：Weather、PAMAP2 四场景三种子正式结果

归档日期：2026-11-01。24个模型、48个训练阶段全部完成，每阶段100轮。模型固定为完整R3 / H_GLOBAL，代码commit `59419e9bbc8f2e8833c8265505d0e84fa2b04de4`，未搜索架构或挑选seed。

每个训练seed先平均20组与其他基线逐值相同的测试遮挡，再对训练seed2023、2024、2025求均值及样本标准差（ddof=1）。结果在训练集全局标准化尺度，仅对隐藏且有原始真值的坐标评分。不运行预测Stage2。

|数据集|场景|测试MSE ± seed SD|测试MAE ± seed SD|有效参数|
|---|---|---:|---:|---:|
|Weather|sr010_bl8|0.042431 ± 0.000577|0.043479 ± 0.000306|447,356|
|Weather|sr015_bl8|0.052501 ± 0.000742|0.053946 ± 0.000418|447,356|
|Weather|sr020_bl8|0.064593 ± 0.001175|0.070304 ± 0.000500|447,356|
|Weather|sr010_bl16|0.078572 ± 0.000796|0.092897 ± 0.000638|447,356|
|PAMAP2|sr010_bl8|0.227562 ± 0.002636|0.181577 ± 0.001163|575,074|
|PAMAP2|sr015_bl8|0.326661 ± 0.001309|0.230819 ± 0.000952|575,074|
|PAMAP2|sr020_bl8|0.466878 ± 0.004067|0.292715 ± 0.000651|575,074|
|PAMAP2|sr010_bl16|0.532003 ± 0.003002|0.334631 ± 0.001549|575,074|

## 记录与检查

- [三种子汇总](summary/three_seed_summary.csv)、[逐训练seed](summary/per_seed.csv)、[480组逐mask结果](summary/per_mask.csv)，均包含校准前后诊断。
- [48阶段选模与资源](summary/training_stages.csv)、[4800轮训练记录](summary/training_epochs.csv)、[checkpoint摘要](checkpoint_manifest.json)。权重未上传。
- [完成检查](completion_audit.json)：完整预算、验证选模、冻结、配对训练mask指纹和导出核验均通过。
- [原基线遮挡对齐](audit/baseline_alignment.json)：Weather采用2026100801–2026100820，PAMAP2采用六基线的2026101401–2026101420。发布默认PAMAP2的20261018银行未用于本次正式评价。
- [执行适配与队列代码](runtime/)仅为本次服务器运行来源记录，不是可直接替代发布入口的新模型；原R3架构代码不变。

## 口径与限制

双RTX4090D运行，Python3.11.17、Torch2.5.1+cu124、CUDA12.4，batch128，Adam lr0.001、weight_decay0。Weather每阶段28600次更新，PAMAP2每阶段8500次。主干按验证full96 MSE选，冻结粗模型/历史库后，校准按验证full96 MAE选；测试不改选。

训练循环seconds含轮内验证和保存，不含全部准备；job_wallclock_seconds是该模型工作进程记录的端到端时间。sum_job_wallclock_seconds是串加的模型耗时，不是双卡实验墙钟。显存为各阶段实测allocated/reserved峰值。本次未独立测推理p50，不借用R2或其他服务器的延迟。

遮挡组属于同一训练模型内重复评价，不是训练重复；若其他基线只有seed2023，不能把其mask标准差等同本表的训练seed波动。不据均值接近宣称统计等效。

Exchange-Rate按用户要求中途移出本机，历史完成的两个模型未纳入本归档；该数据集由其他服务器结果归档负责。原始数据、mask二进制、checkpoint、逐点预测及训练记忆均不上传。

Git发布缺失的11个协议参考JSON曾按release_lock声明的SHA原样恢复，详见release_integrity；这不改变模型代码。

同发布分支上其他四数据集结果见 [既有四数据集归档](../R3_final_four_datasets_3seed_20261031/README.md)，本次未覆盖其内容。
