# 计划数据集原始资产

本次下载对应 2026-10-10 确定的跨领域计划。模型以 E23-v125 为当前定版；下载不启动训练，不生成划分、标准化或测试掩码。

## 位置

| 数据集 | 原始位置 | 说明 |
|---|---|---|
| ETTh1 | `raw/ETTh1.csv` | 既有 CSV，校验后保留原样 |
| Weather | `raw/weather.csv` | 既有 CSV，校验后保留原样 |
| Exchange-Rate | `raw/Exchange-Rate/exchange_rate.txt` | 作者8变量矩阵；源文件没有时间戳，不编造真实日期 |
| Beijing Air Quality | `raw/Beijing_AirQuality/PRSA_Data_20130301-20170228/` | 12个站点原始CSV；以PRSA站点文件为准，UCI包里的data.csv/test.csv不代替完整数据 |
| METR-LA | `raw/METR-LA/` | Torch Spatiotemporal官方加载器指向的v1.0.0包；保留HDF5、传感器ID、位置和距离文件 |
| PAMAP2 | `raw/PAMAP2/PAMAP2_Dataset/` | 原始Protocol/Optional和作者说明；已验证9个Protocol受试者、共14个记录文件 |
| PhysioNet 2012 | `raw/PhysioNet2012/` | 官方A/B/C各4000患者，另保存Outcomes；不能把患者拼成一条连续时间序列 |

原始压缩包存放 `downloads/planned_20261010/`。`metadata/*_download_status.json` 可实时查看下载状态；完整汇总与每个文件的SHA256在 `metadata/planned_datasets_20261010.json`。汇总的state=complete才表示所有数据已通过本次校验；.part表示未完成，不能用于实验。

复用下载脚本：

```bash
python scripts/download_planned_datasets_20261010.py
```

在paper根目录执行。脚本依赖Python标准库和curl，使用官方公开来源。已下载的完整包复用并重新校验；服务器若不支持Range，未完成下载的恢复可能需重新获取。

## 保留原始观测规则

- Air Quality原始NA、PAMAP2的NaN、METR-LA的零值及原时间轴、PhysioNet不规则记录均不改写。
- METR-LA后续采用作者的缺失定义时另生成可见掩码，不能在此阶段前向填补后当真值。
- PAMAP2原始54列含时间、活动标签和52观测列；后续剔除作者标明无效的姿态通道，并把活动标签与修复输入分开。
- PhysioNet outcomes只作为原始附件保存，不能混入修复输入。训练/验证/测试按患者划分，原型库与标准化只基于训练患者。
- 原生缺失没有真值。修复主指标只对原本有观测、再人工遮挡的位置计算。
- 数据尚未接入统一训练入口；原始包下载完成不代表v125与各基线已经能直接在所有数据集运行。

## 下载来源

- Exchange-Rate：https://github.com/laiguokun/multivariate-time-series-data
- Beijing Air Quality：https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- METR-LA：当前官方加载器 https://github.com/TorchSpatiotemporal/tsl/blob/main/tsl/datasets/metr_la.py 指向Hugging Face版本固定数据包。与本地ImputeFormer旧代码中的失效SWITCH链接区分记录。
- PAMAP2：https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- PhysioNet 2012：https://physionet.org/content/challenge-2012/1.0.0/ ，本次公开目录实际提供set-c.tar.gz，因此一并下载，不能沿用旧镜像“C不公开”的说明。

原始压缩包同步到 GitHub Release `datasets-20261010`。PAMAP2按32 MiB分卷；分卷与完整包的SHA256记录在 `metadata/github_datasets_20261010.json`。解压后原始大文件不会直接进入Git历史。

## 本次下载完成清单

2026-10-10：5套新数据全部下载、解压并校验完成；ETTh1和Weather既有文件已核对。汇总表：[planned_datasets_20261010.csv](metadata/planned_datasets_20261010.csv)。每个压缩包及原始文件的SHA256见JSON清单。METR-LA的HDF5结构另见[结构核验](metadata/METR-LA_structure_audit.json)。PAMAP2采用UCI原始PAMAP2_Dataset.zip官方入口，来源已记录到清单。

## 从 GitHub 同步到其他服务器

```bash
git clone https://github.com/Scorpionlailai/-.git paper
cd paper
python scripts/restore_datasets_from_github.py
```

如果已有仓库，先运行 `git pull --ff-only`，再运行恢复命令。需要Python 3.9以上与curl；脚本只下载、校验和解压，不启动训练。Release页面：https://github.com/Scorpionlailai/-/releases/tag/datasets-20261010 。归档恢复完成后可用 `python scripts/restore_datasets_from_github.py --verify-only` 再核对压缩包。
