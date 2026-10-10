# Weather 0.15/8：E24-v79与7基线

用户授权2026-10-08。沿用0.10/8协议，只有缺失起点率变为0.15、块长仍8；各场景从头训练，不复用其他场景权重。

- 原Weather52696行21变量，70/10/20时间划分，96→96，全变量；标准化只拟合训练行。
- 变量独立块缺失、重叠允许；20组测试时间线缺失平均72.774823%，不是15%最终缺失。窗口加权实际比例在测试报告另列。
- 固定验证seed2021；20组测试seed2026100801—2026100820，所有模型共用本场景的已封存NPZ。
- 验证保留5175窗口；每组测试10444窗口；测试指标不参与选模。
- E24-v79沿用全部模块和参数：Stage1 100轮、batch128、lr.001；Stage2 100轮、batch128、lr1e-4，纯MSE、完全冻结修复器，原D16 PatchTST；主验证MAE、同轨迹副验证MSE。
- T1、SAITS：作者Stage1 + 同一冻结公共Stage2；T1修复seed502，公共Stage2 seed2023。
- S4M、CRIB、CoIFNet、BiTGraph、ChannelTokenFormer沿用上一场景锁定作者原生模型/损失/精度/轮数/早停设置，不追加公共Stage2。其他训练seed2023。
- **BRITS暂停，本轮和后续默认不加入，直到用户明确恢复。** 旧场景部分权重保留。
- 训练资源记录全流程时间、阶段时间、allocated/reserved峰值显存、GPU UUID、精度和batch；20组最终测试MSE/MAE均值及mask标准差、全96及近32修复指标（有显式输出时）。未训练Weather clean，DRR留空。

配置：`configs/baselines/weather_sr015_bl8_author_20261008.json`。作者参数依据和不同任务模式解释沿用`2026-10-08_Weather_sr010_bl8_baselines.md`。

新增适配在`integrations/e24_sr015`与`integrations/weather_baselines_sr015`，不修改0.10/8封存代码/结果。数据manifest：`artifacts/Weather/seq96_pred96/sr015_bl8/shared_protocol_v1/dataset_manifest.json`。

顺序：E24-v79全流程→导出E24结果→T1→SAITS→S4M→CRIB→CoIFNet→BiTGraph→ChannelTokenFormer。GPU1映射虚拟cuda0。控制器独立后台运行，每小时记录一次；模型完成后自动20组测试并更新CSV，基线阶段失败同配置最多重试一次后继续其余模型。

控制器启动PID：1691975。总日志/状态：`artifacts/Weather/seq96_pred96/sr015_bl8/all_models_controller.log`、`all_models_status.json`。E24阶段日志和独立源代码封存在`e24_v79_weather_sr015_20261008/seed_2023/E24-v79/`；之后基线队列在`weather_author_comparison_sr015_20261008/`。

汇总：`docs/experiments/Weather_sr015_bl8_E24_baselines_20261008.csv`；E24结果先完成并导出前，条目保持待运行，不填猜测值。
