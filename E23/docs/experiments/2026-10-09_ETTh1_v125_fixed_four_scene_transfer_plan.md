# ETTh1 v125固定设计迁移：四场景

用户在2026-10-09要求：Weather结束后，再迁移到ETTh1四个缺失场景比较。

## 固定设计与训练

采用Weather上v125共享粗修复与64/256维校准的固定设计；不是迁移Weather权重。各场景从头训练粗修复100轮，seed2023/b128/lr0.001，固定验证全局隐藏坐标MSE选粗权重；两条校准各100轮/lr0.001，冻结专家/库，仅校准可训练，目标full96 MAE + 0.25 last32 MAE，固定验证MAE选各主权重，同时保存MSE次权重。固定0.5/0.5组合，不重新搜索比例。

变量21->7只改变相关输入/输出大小；两条校准参数数经实际模型构建核验为97,616和1,176,464。每条校准6600次更新，原训练8449窗口中按batch128使用8448。实际窗口数以协议和运行记录为准。

不安排Stage2；若后续预测，完整修复器冻结/eval，原预测头，纯MSE。测试位置不用于checkpoint选择。

## 数据与四场景

沿用原ETTh1 P4 seq96/pred96、patch16/stride8、训练区间0:8640/验证8640:11520/测试11520:14400、训练全局标准化和固定验证。场景为0.10/8、0.15/8、0.20/8、0.10/16。起点率不是实际缺失比例。

使用原封存20组development缺失位置，与既有T1 Stage1权重的修复audit逐组核对mask SHA、窗口和隐藏有效坐标数。保留原ETTh1 evaluation_drop_last=True，测试每组2688窗口；Weather的10444完整窗口口径不移植到ETTh1。

报告：全96及近32隐藏有效坐标MSE/MAE、20组位置标准差ddof1、完整Stage1源训练时间与allocated/reserved峰值。粗成本去重一次，总100+100+100轮。E23 seed2023、T1作者seed502；20组是缺失位置重复，不是独立训练种子。

## 排队与结果

调度等待以下Weather总状态真正completed才启动；若Weatherfailed，停止依赖链并留错误，不自动跳过。

[Weather状态](/home/xyw/paper/E23/artifacts/Weather/v125_two_scenes_20261009/status.json)

[ETTh1计划](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v125_fixed_four_scene_transfer_20261009/plan.json)

[ETTh1状态](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v125_fixed_four_scene_transfer_20261009/status.json)

[ETTh1调度](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v125_fixed_four_scene_transfer_20261009/controller.py)

GPU0四场景依次运行，每场景粗训练后两条校准并行。GPU1原基线队列不修改。每场景产出matched_repair_comparison.csv（E23-v125与T1）、paired_repair_differences.json、repair_results.csv、result.md和完整20组JSON。已有T1审计经封存引用，不额外重训T1。

## 状态

- [x] 四场景20组原mask库加载核验
- [x] 7变量模型实例参数数核验，脚本编译
- [x] 训练/组合/评估/比较/报告完整调度准备，开跑前source freeze
- [ ] Weather完成后依次训练、评估四场景
- [ ] 汇总四场景正式结果；结果未知，不预先宣称优于T1

## 接续启动修正

Weather总调度已于2026-10-09 20:54:44（北京时间）完成。ETTh1首次接续在参数解析阶段失败，未开始训练：工作目录优先加载生产目录的旧integrations.e23.run，而非PYTHONPATH所指向的封存入口。修正为独立coarse_entry.py，显式将封存source置于sys.path首位，并断言实际run.__file__来自该源；仍保留原seed2023、repair-batch128等参数。原错误日志、初始workflow与source freeze已归档startup_failure_attempt1。仅启动入口及workflow更新为freeze revision2；模型/数据/损失/训练配置未改。重新启动后已实际完成前两轮训练与固定验证，首场景0.10/8持续运行。
