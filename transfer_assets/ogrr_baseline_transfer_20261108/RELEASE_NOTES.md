# OGRR 原始基线资产传输 — 2026-11-08

仅归档既有选定修复权重；未训练、未重新选模，未覆盖原结果。Exchange-Rate 按要求暂不重复上传。Stage2 预测头不作为修复骨干交付。

资产索引：[asset_index.json](asset_index.json)。逐文件来源、字节数和 SHA256 见 [manifests](manifests/)，每个附件内也包含 `SHA256SUMS` 和 `FILE_MANIFEST.json`。所有附件解压到同一空目录，保留原仓库相对路径；代码按原封存 SHA256 去重归档。

|数据集|权重+配置+选模/验证记录+遮挡齐全|必要推理代码齐全|整个历史 source seal 齐全|
|---|---:|---:|---:|
|ETTh1|16/20|16/20|4/20|
|Weather|16/20|16/20|6/20|
|BeijingAirQuality|20/20|20/20|20/20|
|PAMAP2|20/20|20/20|20/20|
|PhysioNet2012|20/20|20/20|20/20|

齐全数按 4 场景 × 5 模型计。每个组合沿用原训练 seed；历史 ETTh1/Weather T1 为 seed502，其余为 seed2023。20 组遮挡是同一权重的重复测试，不是训练 seed。

## 缺项与协议

ETTh1、Weather 的 T1 四场景原 checkpoint 和原验证/20组测试遮挡缺失，选定 epoch 和原 SHA256 仍从原选模记录保留。ETTh1 T1 使用历史遮挡，不能拿重建的 ETTh1 bank 替代；Weather 历史遮挡的文件 SHA256 与新 bank 不同，也未替代。详细原路径和缺失种类见索引与 [missing_assets.json](missing_assets.json)。

原封存 `E24/integrations/repair_baselines/parallel_queue.py` 的一个历史版本未找到，影响 ETTh1 12 组合和 Weather 10 组合的完整 source seal，但属于训练队列调度代码，冻结骨干推理不依赖它。对应原适配器和模型代码的封存 SHA256 均已找到。历史 T1 另有缺失适配器/配置封存版本及 Weather 的4个 `dataset_manifest.sha256` 辅助文件，逐项列在 `manifests/source_code.json`；当前代码仅作补充，未冒充原版本。

## 下载与核验

- [ETTh1.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-baseline-transfer-20261108/ETTh1.tar.gz) — `24852ae9174255757917f548b4e7aa8fbaa0b9051d595862f6bb446ee74dbe35`
- [Weather.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-baseline-transfer-20261108/Weather.tar.gz) — `177daaaf05e5cb85f369f9a90c161edab63d46e8c1a89c1dd4cda118148924b6`
- [BeijingAirQuality.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-baseline-transfer-20261108/BeijingAirQuality.tar.gz) — `615ef7ea474f364fe06ad465bf727477d317b69c48a5907624659c3587300053`
- [PAMAP2.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-baseline-transfer-20261108/PAMAP2.tar.gz) — `55800a833ae2511e43bfeb50128b5d60696a6ec898ad0ae4ede67e7bf838f655`
- [PhysioNet2012.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-baseline-transfer-20261108/PhysioNet2012.tar.gz) — `6c9d092241fa669b204660cb3312e1a1e427241b90088c3d00f0459d65bc4ed7`
- [source_code.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-baseline-transfer-20261108/source_code.tar.gz) — `c16f3a544fa8d84cc2212425cf82de6c9275a14e946236399f29ce3b169241ff`

下载全部附件后执行：

```bash
python verify_packages.py /path/to/downloads
```

解压后也可在各包目录执行 `sha256sum -c SHA256SUMS`。`source_code.tar.gz` 中 `source_by_sha256/<hash>/<filename>` 是逐字节验证的原封存版本；`supplementary_current/` 是补充源码，不能用于宣称历史封存等价。索引保留原 source seal 映射与上游 commit；原实验未记录完整仓库 commit 的，不由本次交付基准 commit 冒充原训练版本。

原文件和报告原文（包括其中的旧路径与旧政策）保持不变；本次用户上传授权适用于此次独立分支与 Release。
