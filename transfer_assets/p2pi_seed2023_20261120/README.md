# P2PI seed 2023 原始选定权重交接包

P2PI 沿用原归档名称 R3 / H_GLOBAL。保留原始路径与文件字节。
Weather 4组、PAMAP2 3组；7份完整 model.pt，同时附原 main/best.pt 与 calibration/best_mae.pt。
未重训、未重选 checkpoint；未改写原结果或论文。无缺项。

每组附原 results.json、两阶段 complete.json / binding.json / model.json、完整选模曲线与逐 epoch 记录、原任务配置。
runtime snapshot、适配代码、execution_lock 和原归档 checkpoint_manifest.json 原样附带。
checkpoint_manifest 中 checkpoints_uploaded=false 是原历史记录，未修改；本次交付状态见 ASSET_INDEX.json。

共享测试遮挡：Weather 2026100801–2026100820；PAMAP2 2026101401–2026101420；每组另含 validation_2021.npz。
全部147份NPZ均与原共享基线目录字节一致。Weather manifest 中 legacy_file_sha256 指更早的NPZ封装，
本包提供正式R3实际使用且与共享基线一致的NPZ，校验以 manifest.sha256 和 SHA256SUMS 为准。
未打包完整原始或处理后数据，需沿用已有数据路径。PAMAP2恢复时须使用随附 adapter.py / baseline_pamap_data.py 安装原1401遮挡银行。

| 数据集 | 场景 | main原选定epoch | calibration原选定epoch |
|---|---|---:|---:|
| Weather | sr010_bl8 | 42 | 95 |
| Weather | sr015_bl8 | 86 | 95 |
| Weather | sr020_bl8 | 89 | 89 |
| Weather | sr010_bl16 | 83 | 99 |
| PAMAP2 | sr010_bl8 | 99 | 41 |
| PAMAP2 | sr015_bl8 | 98 | 98 |
| PAMAP2 | sr010_bl16 | 93 | 85 |

原模型代码 commit: `59419e9bbc8f2e8833c8265505d0e84fa2b04de4`。

逐文件核验：在解压目录运行 `sha256sum -c SHA256SUMS`。归档校验见包旁 ARCHIVE_SHA256SUMS。

[下载交接包](https://github.com/Scorpionlailai/-/releases/download/p2pi-seed2023-20261120/P2PI_Weather_PAMAP2_seed2023_selected_R3_20261120.tar.gz)
