# R3定版：本机三数据集、三训练seed正式重跑

用户于2026-10-30指定本服务器负责ETTh1、BeijingAirQuality、PhysioNet2012，双GPU并行；遮挡必须与其他基线一致，并确认采用训练seed2023、2024、2025。

模型固定为完整R3 / H_GLOBAL，模型发布commit `59419e9bbc8f2e8833c8265505d0e84fa2b04de4`。后续 `836f3c4bb65867a6a21863cb9aaacfd05cfa769a` 只追加Exchange原基线遮挡，未改变模型。

## 固定预算和统计

- 三数据集 × 四场景（sr010_bl8、sr015_bl8、sr020_bl8、sr010_bl16）× 三训练seed = 36个模型。
- 每模型主干100轮＋冻结完整粗模型和历史库的单支64校准100轮；batch128、既有损失/选模不变。
- 每训练seed评价相同20组测试遮挡，先对20组取均值，再对三个训练seed汇总均值±样本标准差；20组遮挡不能当作20次独立训练。
- 全96点/近32点、校准前后指标均保留；原生缺失不作为监督或指标真值。
- 记录逐轮曲线、checkpoint绑定、峰值allocated/reserved显存、GPU/Torch、每阶段循环耗时和包括数据准备/验证/保存/测试/导出的端到端墙钟时间，以及同GPU/batch128的推理p50。
- 本轮不运行预测Stage2，不搜索结构、阈值或训练seed，不按测试结果更改定版。

## 遮挡与数据

ETTh1直接逐值比对原T1等基线实验保存的实际npz；Beijing按原六模型共享manifest的content_checksum核对；PhysioNet按原共享manifest的content_sha256核对。每数据集四场景共84份验证/测试遮挡，全部通过后才放行该数据集队列。

- ETTh1测试seed2026100401–2026100420。
- BeijingAirQuality测试seed2026101301–2026101320。
- PhysioNet2012测试seed2026101501–2026101520。发布包原入口的2026101801–2026101820来自后续消融，本次运行数据适配器已改为原基线银行；这不是模型修改。
- 验证mask seed2021，训练沿用动态epoch遮挡与原生成器，人工训练遮挡不随模型seed重新抽选。
- Beijing只在独立data_root中规范化旧sidecar的日期序列化；原始mask逐位保留并逐站对照原12站点CSV。
- PhysioNet用原始12000条记录和原预处理器重建；所有数组文件SHA、患者划分、统计量和prepared manifest均要求与原基线完全匹配。

## 任务分配和运行位置

实验根目录：`/home/xyw/paper/E23/artifacts/performance/R3_final_three_datasets_3seed_20261030`。

两条持久systemd用户服务各占一张物理GPU，每条顺序执行18个模型。先ETTh1，再BeijingAirQuality，再PhysioNet2012；每个数据集的全部scene/seed均覆盖。

- GPU0：每seed的sr015_bl8、sr010_bl8。
- GPU1：每seed的sr020_bl8、sr010_bl16。
- 当前服务：`r3-formal-gpu0-20261030-v2.service`、`r3-formal-gpu1-20261030-v2.service`。
- 原首次调度因脚本命名queue.py遮蔽Python标准库queue而在导入阶段退出，未开始训练；已更名run_formal_queue.py，原失败日志保留。模型源码及预算未受影响。

plan.json固定实验设计；implementation/为运行源码快照；baseline_references/保存基线来源；audit/保存掩码核对结果；queue_gpu{0,1}.json保存队列状态；logs/为逐模型日志。outputs/runs/保存权重和阶段资源，outputs/evaluation/保存校准前后20mask结果、导出与推理资源。completed/保存模型全流程墙钟；failures/保存失败记录。

汇总由multiseed_summary_gpu{0,1}.json和summary_gpu{0,1}.csv生成；两个文件扫描同一结果树，每次完成模型后更新，其内容仅代表该次刷新时刻。正式比较只接受n_training_seeds=3的组。

单模型失败会明确登记并继续其他已放行任务；失败不能视作已完成。所有模型完成后须核对36份逐模型结果、每份20组mask、三seed完整性与冻结校验，再发布最终表。

## 用户追加Exchange-Rate

2026-10-30用户要求原三个数据集全部跑完后，继续Exchange-Rate。追加四场景×seed2023/2024/2025，共12个模型，总计48个模型。原36个任务及其运行源码不变。

Exchange数据、划分、训练统计和原基线银行84份验证/测试mask均已逐值核对。原银行发布commit为`836f3c4bb65867a6a21863cb9aaacfd05cfa769a`。新计划在exchange_extension_plan.json；遮挡审核在audit/Exchange-Rate_mask_alignment.json。

两条追加持久服务`r3-exchange-followup-gpu0-20261030`、`r3-exchange-followup-gpu1-20261030`先等待原两队列全部成功完成36个模型，才同时放行Exchange任务。若原队列有失败，停止放行并明确记录，不跳过失败把实验当作完成。追加阶段GPU0负责sr015_bl8/sr010_bl8，GPU1负责sr020_bl8/sr010_bl16，各6个模型。等待状态在exchange_start_gate_gpu{0,1}.json，启动后进度在exchange_queue_gpu{0,1}.json。
