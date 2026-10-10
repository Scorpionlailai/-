# Weather v125固定设计：0.20/8与0.10/16

2026-10-09。用户要求两个场景均试验；沿用已确认v125设计，主要评价Stage1修复。当前是已启动的实验计划，结果尚未完成。

## 实验口径

- 数据Weather，21变量，seq96/pred96，沿用E24各场景sealed shared_protocol_v1（原数据划分、训练标准化、seed2021固定验证和20份原测试缺失位置）。起点率不等同于实际缺失率。
- 分别从头训练同场景v79粗修复100轮，seed2023/batch128/lr0.001，原粗损失和固定验证全局修复MSE选模。
- 从各自同场景父权重训练两条独立校准路径：64/256维，各100轮，28600次更新，lr0.001。冻结专家和库，旧均值校准与新点校准训练，full96 MAE + 0.25 last32 MAE，全局训练标准化尺度，按固定验证MAE保存主权重，同时保存MSE次权重。
- 固定权重0.5/0.5组合；共享粗修复只执行一次，无场景比例搜索，不跨场景复用权重。结构和loss不改。
- 每场景20组缺失位置，10444窗口/组，评价隐藏有效坐标的full96/last32 MSE/MAE，位置标准差ddof1；这不是20次独立训练。
- 统计全部Stage1源成本，粗训练去重一次，共100+100+100轮；allocated/reserved峰值。预测未安排。未来Stage2仍冻结完整修复器、原预测头、纯MSE。

## 执行与隔离

GPU0（GPU-382ce2ea-81a9-82ea-073f-729eaefd0268）；先0.20/8再0.10/16。每场景粗训练后两条校准同时在GPU0运行。GPU1已有E24基线队列不修改、不重复。源文件及配置在开跑前散列锁定，父权重在完成后且校准前锁定。

总调度：[controller.py](/home/xyw/paper/E23/artifacts/Weather/v125_two_scenes_20261009/controller.py)

实时状态：[status.json](/home/xyw/paper/E23/artifacts/Weather/v125_two_scenes_20261009/status.json)

场景目录：

- [/sr020_bl8/](/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr020_bl8/e23_v125_fixed_transfer_20261009)
- [/sr010_bl16/](/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl16/e23_v125_fixed_transfer_20261009)

调度进程30秒落盘状态；监测任务每5分钟检查PID/log，汇报每10分钟。完成后各自产出repair_results.csv/result.md。基线未完成前不作优于T1/SAITS的结论；获得同场景结果后需配对核验原20组mask SHA及坐标数后才能正式比较。

## 待办

- [ ] 0.20/8粗训练及64/256校准完成
- [ ] 0.20/8固定组合严格恢复、正式20组修复结果
- [ ] 0.10/16粗训练及64/256校准完成
- [ ] 0.10/16固定组合严格恢复、正式20组修复结果
- [ ] 同场景T1/SAITS结果配对核验与汇总
