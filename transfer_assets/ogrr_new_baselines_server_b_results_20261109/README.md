# Server B 新基线＋OGRR 完整结果（2026-11-09）

BeijingAirQuality、Exchange-Rate 各12组，合计24/24完成，失败0组。每组包含原模型与冻结骨干后的OGRR，以及共同20组测试遮挡的配对评价；未运行其他四数据集，未修改论文或覆盖旧基线结果。

源交接 commit：`f46822e78a4cb96b931ba8f54e233e1778ccf15c`。封存NN、mask生成、损失、选模、配置均未修改，无路径适配；独立环境Python3.10.21 / torch2.7.1+cu128 / numpy2.2.6。SAITS骨干seed26，ModernTCN/TimeMixer++骨干seed2021（后者是预先固定缺省，不是作者最佳seed），OGRR seed2023。

## 配对结果

以下为每个组合20组遮挡均值的相对降幅，再对四场景作算术平均；不跨数据集混合，也不是合并坐标后的全局降幅。正值表示改善，负值表示退化。

|数据集|模型|MSE平均降幅|MAE平均降幅|
|---|---|---:|---:|
|BeijingAirQuality|SAITS|4.18%|1.94%|
|BeijingAirQuality|ModernTCN|55.82%|41.35%|
|BeijingAirQuality|TimeMixerPP|52.57%|39.67%|
|Exchange-Rate|SAITS|64.73%|50.08%|
|Exchange-Rate|ModernTCN|-25.88%|-32.66%|
|Exchange-Rate|TimeMixerPP|61.40%|50.82%|

20/24组的MSE和MAE均改善。Exchange-Rate/ModernTCN四场景均退化，负结果完整保留。单训练种子，20份mask的样本标准差不等于训练seed波动。

逐场景指标与mask间样本标准差见 [comparison.csv](comparison.csv)，逐20mask指标见 `cases/<dataset>/<scene>/<model>/results.json`。索引见 [asset_index.json](asset_index.json)。

## 归档下载

- [server_b_BeijingAirQuality_20261109_results.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-new-baselines-server-b-results-20261109/server_b_BeijingAirQuality_20261109_results.tar.gz) — SHA256 `b9a932826a0835be3e8a094846725fb970adb36bdb5c6388cb1a4ceb58a246c6`
- [server_b_Exchange-Rate_20261109_results.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-new-baselines-server-b-results-20261109/server_b_Exchange-Rate_20261109_results.tar.gz) — SHA256 `7d8635cbdb27bf535384d06a83317d935671f59817ec3338814d1cda9841e34d`

每包包含原模型/OGRR selected与latest权重、原配置与source/data/mask哈希绑定、每轮记录、逐20mask配对结果、耗时、显存、同设备batch128推理p50、原始日志及运行环境；`FILE_MANIFEST.json`、`SHA256SUMS`在包内和本目录均可查。

下载后先执行 `sha256sum -c BUNDLE_SHA256SUMS`。在独立目录解压，再执行 `sha256sum -c SHA256SUMS`。若合并到原服务器，包内 `runs/`、`logs/` 相对于 `paper/E23/artifacts/performance/OGRR_new_baselines_20261109/`；先核对已有文件并保留差异文件。两个包内根级清单与报告不同，请分别解压核验后再合并各数据集的 `runs/`、`logs/`，不要直接相互覆盖。

## 核验

交接包976文件及168份验证/测试mask核验通过，模型测试5项通过。上传前再次核验48份selected权重、24份结果文件、480条配对mask记录及冻结骨干摘要；两个结果归档内4615个清单文件逐一核验SHA256，另核验包内报告与清单。
