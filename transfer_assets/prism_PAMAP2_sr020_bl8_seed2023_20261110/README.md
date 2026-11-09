[下载完整交接包](https://github.com/Scorpionlailai/-/releases/download/prism-pamap2-sr020-bl8-seed2023-20261110/PRISM_PAMAP2_sr020_bl8_seed2023_R3_20261110.tar.gz) · [资产索引](asset_index.json) · [归档SHA256](ARCHIVE_SHA256SUMS)

请下载归档并在新目录解压后执行恢复脚本。以下路径相对于解压目录。

# PRISM / PAMAP2 / sr020_bl8 / seed2023 原正式实验权重交接

本包交付完整 R3（H_GLOBAL，论文名 PRISM），来自本机保留的原正式实验 `/home/xyw/r3_final_runs_20261030`。原归档为 `E23/artifacts/performance/R3_final_weather_pamap2_3seed_20261101`，原发布记录 commit `dc97c14fa84bf901d03deef3f95f414845be890d`，模型源代码 commit `59419e9bbc8f2e8833c8265505d0e84fa2b04de4`。权重及原记录按字节保留。

## 选定权重

首选完整导出：
`r3_final_runs_20261030/outputs/evaluation/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/model.pt`
SHA256：`cf4ae288225ac7f4f3410b9676fbdef373ff7800c1c307a42ee5592b1831f159`

同时提供原所选阶段权重及原型/检索状态，供恢复对照：
- `r3_final_runs_20261030/outputs/runs/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/main/best.pt`：`159667fcd23372010421ee876596da8f5d76c105167afbd07f2962f846e368a6`
- `r3_final_runs_20261030/outputs/runs/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/calibration/best_mae.pt`：`1a05192b33cba1fb9eb31d28b1904abdbb0373833627c555b0f58ceb0478debd`

原选定 main epoch=99，calibration epoch=34（保留原记录口径）。原 `complete.json`、`results.json`、`checkpoint_manifest.json` 在包根目录提供同字节便捷副本；两个阶段各自的 complete/binding/model/curve/epochs 记录保留在原相对目录下。`checkpoint_manifest.json` 原文的 `checkpoints_uploaded: false` 描述的是 2026-11-01 历史发布，原文未改。

## 原 runtime 与配置锁

`r3_final_runs_20261030/snapshot/E24/versions/r3_20261030/` 含完整冻结源码、vendor、原配置、release_lock/vendor_lock/provenance/model_identity 和依赖要求。原执行适配器 `adapter.py`、`baseline_pamap_data.py`、protocol、execution_lock 等保持原字节。`original_publication/` 保留从原发布 commit 读取的原归档文件；对应5份目标原结果/完成/曲线证据与原发布 source_evidence_hashes 匹配。

全部218项release锁、158项vendor锁及原阶段binding与执行锁已核验。新增 `verify_restore.py` 仅负责路径配置、读取原权重、完整性与推理回放检查，调用原冻结模型实现。不要把其他分支的最新版模型文件加入 Python 导入路径。

## 数据、窗口和原遮挡

包内仅提供 PAMAP2/sr020_bl8 的21份原始NPZ：固定验证seed2021，20组测试seed2026101401–2026101420。它们与原三基线共享bank及本机收到的同一bank逐文件SHA256一致；原发布默认2026101801银行不用于本包恢复路径。

原数据/预处理文件哈希、原始Protocol/Optional记录的SHA256、受试者划分及记录边界见 `data_dependencies/pamap2_10hz_recordwise/manifest.json`。`DATA_DEPENDENCIES.json` 列出当前服务器应复用的三个NPY、文件大小及哈希。完整原始数据和处理后NPY不重复打包。原窗口清单和索引数组SHA256见原 `outputs/assets/PAMAP2/sr020_bl8/manifest.json`；40变量，train10958 / validation1306 / test3349，seq96、stride16，记录与受试者边界保持原协议。

## 本机恢复

在新空目录解压，先核验包内清单：
```bash
sha256sum -c SHA256SUMS
```
原环境为Python3.11.17 / torch2.5.1+cu124 / numpy1.26.4，原依赖清单位于冻结snapshot的requirements.txt。已在匹配环境CPU上完成恢复检查；`recovery_environment_freeze.txt` 记录本次恢复环境。

当前服务器执行：
```bash
/home/xyw/.local/share/mamba/envs/paper/bin/python -B verify_restore.py   --data-root /home/xyw/paper --device cpu   --mask-seed 2026101401 --report /tmp/prism_restore_check.json   --save-window /tmp/prism_sr020_seed2023_mask2026101401.npz
```
若接收服务器环境路径不同，用匹配环境的Python执行同一命令。`--data-root` 指向已有 `data/processed/pamap2_10hz_recordwise/` 的工程根目录。默认读取完整model.pt并用所附两个best权重独立恢复作对照；不会调用训练或原evaluate.main，也不会写回原结果。恢复脚本会拒绝非20261014测试seed。

`--save-window` 可选：保存原标准化尺度的truth、observed、native_valid、prediction、starts、columns和mask_seed，供与T1/ImputeFormer/TimesNet按相同记录、窗口、变量及同一人工mask对齐绘图。该文件是推理辅助产物。原20组正式结果保留在原results.json。

恢复检查覆盖验证及test2026101401各2窗：完整导出与所选阶段重组模型的完整状态及输出逐位一致，可见值保持，原型/检索状态推理前后不变；详见restore_verification.json。没有重训、重新选模、使用latest或0.15/8替代；未覆盖原实验结果或修改论文。
