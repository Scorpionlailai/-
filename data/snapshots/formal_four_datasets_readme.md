# R3正式修复实验：四数据集、四场景、三训练seed

归档日期：2026-10-31。48个模型全部完成，96个训练阶段完整执行100轮；未运行预测Stage2。用户已确定完整R3 / H_GLOBAL，本轮未搜索结构或随机种子。

模型代码发布commit：`59419e9bbc8f2e8833c8265505d0e84fa2b04de4`。原Exchange基线遮挡银行commit：`836f3c4bb65867a6a21863cb9aaacfd05cfa769a`。

模型入口在仓库 `E24/versions/r3_20261030`。运行只对数据入口作必要适配，见runtime_data_overlays/和runtime_source_lock.json；原R3架构代码未修改。PhysioNet测试遮挡采用原基线2026101501–2026101520，不能误用后续消融20261018银行。

## 最终修复结果

全96点隐藏且原本存在真值的坐标，训练集全局标准化尺度。每个训练seed先平均相同20组测试遮挡，再对独立训练seed2023/2024/2025计算均值±样本标准差。遮挡组不等于独立训练重复。

| 数据集 | 场景 | MSE | MAE |
|---|---|---:|---:|
| ETTh1 | sr010_bl8 | 0.116781 ± 0.001319 | 0.205465 ± 0.001606 |
| ETTh1 | sr015_bl8 | 0.172229 ± 0.001825 | 0.256546 ± 0.001479 |
| ETTh1 | sr020_bl8 | 0.261890 ± 0.001713 | 0.322662 ± 0.001965 |
| ETTh1 | sr010_bl16 | 0.334986 ± 0.006672 | 0.372039 ± 0.004057 |
| BeijingAirQuality | sr010_bl8 | 0.077236 ± 0.000584 | 0.144566 ± 0.000524 |
| BeijingAirQuality | sr015_bl8 | 0.094839 ± 0.000677 | 0.163089 ± 0.000209 |
| BeijingAirQuality | sr020_bl8 | 0.122833 ± 0.001111 | 0.188204 ± 0.000303 |
| BeijingAirQuality | sr010_bl16 | 0.146832 ± 0.000833 | 0.206913 ± 0.000600 |
| PhysioNet2012 | sr010_bl8 | 0.386760 ± 0.003590 | 0.343365 ± 0.001006 |
| PhysioNet2012 | sr015_bl8 | 0.476532 ± 0.001034 | 0.403849 ± 0.000135 |
| PhysioNet2012 | sr020_bl8 | 0.564794 ± 0.000846 | 0.455283 ± 0.000148 |
| PhysioNet2012 | sr010_bl16 | 0.581826 ± 0.001249 | 0.469871 ± 0.000192 |
| Exchange-Rate | sr010_bl8 | 0.009212 ± 0.002400 | 0.057764 ± 0.012104 |
| Exchange-Rate | sr015_bl8 | 0.015368 ± 0.002077 | 0.073880 ± 0.007749 |
| Exchange-Rate | sr020_bl8 | 0.054194 ± 0.001231 | 0.098458 ± 0.001036 |
| Exchange-Rate | sr010_bl16 | 0.162275 ± 0.004526 | 0.172044 ± 0.001131 |

## 读取顺序

- `summary/R3_four_datasets_final_full96.csv`：论文主表候选，最终修复三seed汇总。
- `summary/R3_four_datasets_all_stages.csv`：粗恢复/校准后 × full96/recent32，使用同一已选模型，未另挑校准前最优。
- `summary/R3_four_datasets_per_training_seed.csv`：逐seed阶段结果。
- `summary/R3_resources_per_training_seed.csv` 和 `R3_resources_three_seed_summary.csv`：耗时、显存和推理p50。
- `outputs/evaluation/**/results.json`：原始逐mask结果、checkpoint选择和导出校验；`resources.json`：原始延迟与显存记录。
- `outputs/runs/**/{complete,binding,model}.json`：训练预算、源码/数据绑定和冻结证据。
- `training_epochs.jsonl.gz`：9600份完整逐轮JSON无损归档，每行含原相对path、原文件SHA256及record。
- `checkpoint_manifest.json`：96份选定粗恢复/校准权重的原路径及SHA256。权重保留在原服务器，**没有上传**。
- `plan.json` + `exchange_extension_plan.json` + `experiment_design.md`：36个原模型和12个追加模型的完整设计。
- `data_manifests/`、`audit/`、`preparation/`：原始数据、预处理和遮挡一致性证明。
- `archive_audit.json`、`archive_manifest.json`：完成核验和本次上传文件完整性。

## 如实保留的现象与边界

ETTh1、BeijingAirQuality、PhysioNet2012四场景的校准后MSE和MAE均优于对应粗输出。Exchange前三场景校准后MSE更差，四场景MAE均更差；低缺失场景存在明显训练seed波动，不能只报告有利seed。模型仍按预定验证指标选取，没有因测试表现改换权重。

本次只上传修复实验；没有新增预测结果。已归档基线通常是单训练seed，不得把本轮三seed与旧单seed的数值差异直接表述为统计显著性。人工遮挡的起点率不是最终缺失比例。

`end_to_end_seconds` 包含进程导入/数据准备、主干与校准训练、轮内验证和保存、20组测试、导出及推理基准；阶段 `*_loop_seconds` 仅是对应训练循环，二者不要混用。主干/校准峰值显存分别原样报告，reserved可能包含同进程分配器缓存。延迟为batch128、5次预热+25次同步计时，物理GPU索引和环境随记录保存。

没有上传原始数据、处理后的数据数组、mask二进制、训练权重或预测缓存；原Exchange遮挡银行已在单独目录发布。历史本机绝对路径是来源记录，不是新服务器上的有效路径。新服务器从头运行时，须准备相同数据并校验预处理/遮挡哈希；已有权重恢复则需另行同步checkpoint_manifest所列资产。

重建逐轮JSON示例：

```python
import gzip, json
from pathlib import Path
with gzip.open('training_epochs.jsonl.gz', 'rt') as f:
    for line in f:
        item = json.loads(line)
        target = Path('replayed_epoch_records') / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(item['record'], indent=2) + '\n')
```

上例恢复JSON内容；重新序列化的文件字节可能与原文件不同，原SHA用于核对原始记录。
