# Paper：缺失回溯窗口修复与预测实验

本仓库用于多服务器分工：远端服务器复现基线，本服务器迭代自己的修复模型。当前首要比较 Stage 1 修复；预测作为后续评估，Stage 2 必须冻结修复器，使用同一个原预测头和纯 MSE。

## 目录

| 目录 | 内容 |
|---|---|
| E22 | 早期模型及实验记录 |
| E23 | 模型迭代、历史版本、v125、实验文档 |
| E24 | 独立实验工程及统一实验规范 |
| T1 / SAITS | 现有修复基线 |
| BRITS / ImputeFormer / TimesNet / CSDI | 补充的修复基线官方源码 |
| CRIB / CoIFNet / BiTGraph / ChannelTokenFormer / S4M | 其他已研究基线 |
| data | 共享原始数据 CSV |

先读 [实验规范](E24/docs/EXPERIMENT_PROTOCOL.md)、[新增修复基线交接](E23/docs/experiments/2026-10-10_repair_baselines_remote_reproduction_handoff.md) 和 [跨服务器使用说明](docs/MULTI_SERVER_SETUP.md)。

## 快照范围

这是从本机 paper 导出的源码和记录快照。包含配置、文档、关键文本实验记录和数据 CSV；不包含训练权重、缓存、二进制实验产物及运行日志；另排除 ImputeFormer 的大幅宣传海报以缩小网络传输。因此可以用于从头训练，不能直接加载原服务器的既有权重完成评估。新增三个基线已获取官方源码，BRITS 已存在；新增共享掩码适配器尚未完成，不能将下载等同于复现完成。

历史工作树源码已经实体化；共享数据软链接改为仓库内相对链接。各作者原仓库的 `.git` 未嵌套上传，源码与许可证实际包含在本仓库中，不依赖未注册的 submodule。官方来源及 commit 见 [基线登记](E23/docs/experiments/repair_baseline_registry_20261010.csv)。

`GITHUB_SNAPSHOT_MANIFEST.json` 记录导出文件 SHA256、原始来源、链接及排除项。历史记录中的原服务器绝对路径用于溯源，不能当成新服务器可直接使用的路径。
