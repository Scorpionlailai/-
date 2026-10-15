# E24

以封存 **E23-v79** 为起点的新实验工程。模型/训练核心从v79正式冻结源码包提取，实验输出与历史E23分开。

## 先读

- [PAMAP2 六模型、四场景修复与预测完整结果（2026-10-15）](docs/experiments/2026-10-14_PAMAP2_reproduction.md)
- [Beijing Air Quality 六模型、四场景修复与预测结果（2026-10-14）](docs/experiments/2026-10-13_Beijing_AirQuality_reproduction.md)
- [ETTh1 / Weather 四个新增修复基线最终结果与v125、T1对照（2026-10-13）](docs/experiments/2026-10-13_ETTh1_Weather_repair_baselines.md)
- [E23→E24交接：构成、历史实验、最终结果、目录和下一步](docs/2026-10-07_E23_to_E24_handoff.md)
- [实验规范v1：冻结Stage2、纯MSE、数据/mask/选模/统计/资源](docs/EXPERIMENT_PROTOCOL.md)
- [机器可读固定设置](configs/experiment_protocol_v1.json)
- [只读继承验收](migration/e24_inheritance_acceptance_20261007.json)

## 入口

```bash
cd /home/xyw/paper/E24
export PYTHONPATH="$PWD/src:$PWD"
/home/xyw/miniforge3/envs/malr/bin/python -m integrations.e24.run --help
```

公共Stage2仅训练原预测头：纯MSE、100轮、batch128、lr1e-4；修复器全部参数/缓冲/记忆/检索冻结。入口拒绝joint、慢更新和混合预测损失。底层保留E23模块/manifest命名以加载原权重，不代表继续向E23写实验结果。

基线作者仓库位于 `../T1`、`../BRITS`、`../SAITS`、`../S4M`、`../CRIB`、`../CoIFNet`、`../BiTGraph`、`../ChannelTokenFormer`；共享数据 `data -> ../data`。E23历史权重仅在 `artifacts/parent_e23_v79` 查看，新实验使用独立输出目录。

当前ETTh1四场景继承已验收。Weather 的21变量适配与冻结检查已通过，0.10/8 从头100＋100轮及20组测试已完成：预测MSE0.199519、MAE0.247416；见 [Weather 实验设计与记录](docs/experiments/2026-10-07_Weather_v79_sr010_bl8.md) 和 [结果CSV](docs/experiments/Weather_sr010_bl8_E24_v79_seed2023_results.csv)。其他数据集仍需单独封存数据/mask身份并验收。

Weather 全流程入口为 `python -m integrations.e24.weather_workflow --dataset-manifest <manifest> --output <新的空目录> --device cuda:1`，会依次完成100轮Stage1、100轮冻结Stage2及20组测试。具体命令/产物路径见实验记录。历史E23队列继续暂停。

Weather 0.10/8 的八基线后台队列：`python -m integrations.weather_baselines.queue launch`；状态：`python -m integrations.weather_baselines.queue status`。详细作者配置/冻结规则/日志目录见 [基线实验记录](docs/experiments/2026-10-08_Weather_sr010_bl8_baselines.md)，汇总自动写入 `docs/experiments/Weather_sr010_bl8_E24_baselines_20261008.csv`。
