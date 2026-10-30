# R2：可迁移训练与消融版本（2026-10-26）

这是当前 **R2＋后端邻点机制**，不是历史 v125、旧 A 链或来源摘要替代版。
发布入口独立于原实验服务器；冻结模型实现及其运行依赖放在 `vendor/`，路径均相对本目录解析。
数据、训练记忆、checkpoint、凭据及其他实验目录不随代码发布。

## 另一台服务器开始运行

```bash
git clone --branch release/r2-portable-20261026 https://github.com/Scorpionlailai/-.git
cd ./-/E24/versions/r2_20261026
python -m venv .venv
source .venv/bin/activate
# 按服务器 CUDA 环境安装 torch；原实验使用 Python 3.11 / torch 2.5.1+cu124。
pip install -r requirements.txt

# /srv/r2-data/data/raw/ETTh1.csv 必须存在；不要用重新排序或改列名的 CSV。
python selfcheck.py --data-root /srv/r2-data
python train.py --dataset ETTh1 --scene sr015_bl8 --variant R2 \
  --seed 2023 --stage both --device cuda:0 \
  --data-root /srv/r2-data --output-root /srv/r2-runs
python evaluate.py --dataset ETTh1 --scene sr015_bl8 --variant R2 \
  --seed 2023 --device cuda:0 --data-root /srv/r2-data --output-root /srv/r2-runs
```

先用 `--smoke-batches 1` 运行 train 和 evaluate，可验证两轮、每轮一个训练 batch 的小流程。
smoke 使用单独目录并明确标记，不能作为正式实验成绩。正式运行不传此参数，主干与校准均完整100轮。
自带 `selfcheck.py` 是小样本正确性检查，不替代正式训练。

双卡跑三个配对 seed（独立模型并行，不是单模型 DDP）：

```bash
python launch.py --dataset ETTh1 --scene sr015_bl8 --variant R2 \
  --seeds 2023 2024 2025 --gpus 0 1 \
  --data-root /srv/r2-data --output-root /srv/r2-runs
```

launch 会按依赖训练主干与校准，然后测试和导出每个 seed；不改变选模规则。
可用场景：`sr010_bl8`、`sr015_bl8`、`sr020_bl8`、`sr010_bl16`。
评估逐模型先平均20组测试mask；比较三个seed时使用训练seed的样本标准差，不能将mask当成训练seed。
同一输出目录允许中断恢复；源码、协议或数据改变后应使用新的输出目录。

## 当前结构与消融入口

```text
可见值、mask → 归一化、重叠patch → 共享前端 F、z0
  ├─ base decoder(z0)
  └─ T / V / H候选 → 逐点八头注意力（＋null）
                         → [Query, readout]联合残差头
base＋残差 → confidence重叠融合
          → 单支64校准：局部卷积＋一层Transformer＋后端邻点
          → 尺度还原、原始观测硬覆盖
```

固定主干64、校准64；保留Query直达联合输出头。
无片段级证据仲裁、T/V顺序预更新、逐点专家再路由、独立均值/global修正、尾部精修CNN或双支集成。

| 文件 | 职责 |
|---|---|
| `r2_variants.py` | `DirectCore` 为实际R2前向；`Calibration`保留单层校准 |
| `interfaces.py` | `DirectReadout`、`DirectConfidence`；没有片段权重接口 |
| `source_content.py` | 来源内部候选表示、质量与权重 |
| `variants.py` | `FULL`（CLI别名`R2`）、`SUM`、`CTX`三个受控配置 |
| `single.py` | 冻结主干、单支校准和观测保护 |
| `train.py` / `reuse.py` | 完整训练、恢复、严格绑定与本地checkpoint读取 |
| `evaluate.py` | 选定checkpoint的校准前后误差、20mask和精确导出检查 |
| `vendor/` | 当前运行所需的冻结底层代码，`vendor_lock.json`验证其原始字节 |

默认 `--variant R2` 等同 `FULL`，是要消融的共同基准。
`--variant SUM` 是当前R2上的来源摘要读取：按已有归一化来源权重聚合候选及质量，屏蔽无效来源，读取T/V/H摘要＋null；未新增聚合器。
`--variant CTX` 是无候选容量对照：不构造T/V/H、无历史记忆，以实际参与前向的上下文网络替换读取路径。
ETTh1有效参数：R2/SUM 353,248；CTX 353,256。主干、前端和校准宽度仍为64；CTX内部385宽隐藏层仅用于预先解析匹配容量。
模型初始化仍借用冻结旧构造器以保留初始化顺序，随后删除旧模块；这不代表旧仲裁或后处理参与R2前向。
不要把 `r2_variants.py` 中旧构造辅助函数的 R0/R1 当成本次主入口。
如修改 `vendor/`，需显式更新其hash锁并记录差异；优先在外部R2代码中实现消融，使用独立输出目录，不覆盖基准。

## 训练协议

- seed：2023、2024、2025；训练mask及数据顺序使用2023＋epoch，配对一致。
- batch128，Adam，lr0.001，weight_decay0；主干100轮，再冻结主干及历史库训练校准100轮。
- ETTh1每阶段66更新/轮，共6600更新；其他数据集按同batch128完整训练集更新数确定。
- 主干按验证full96 MSE最小选；校准按验证full96 MAE最小选，另保留MSE选择checkpoint作敏感性记录。
- 主干损失：hidden MSE＋recent32均值残差平方＋0.1辅助对齐；校准损失：hidden MAE＋0.25 recent32 MAE。
- `evaluate.py`固定使用MAE选择的校准；校准前结果使用相同已选主干，只作诊断，不二次选模。
- 验证替代门槛沿用三seed均值MSE、MAE均不超过基准1%；跨场景逐场景检查，不能用平均抵消失败。
- 延迟需在同一GPU、batch128、5次预热＋25次同步计时、无其他GPU进程竞争下另行测量；参数匹配不等于计算量匹配。

## 数据

`--data-root` 指向包含 `data/` 子目录的位置。ETTh1/Weather需要 `data/raw/ETTh1.csv` / `data/raw/weather.csv`。
原始CSV来源、列、训练集标准化、时间划分及mask参考信息随 `protocol_references/` 的JSON记录发布，数据文件不发布。
`selfcheck.py`会核对ETTh1源文件hash。不要把不同的下载内容当成完全一致复现。

PAMAP2使用40变量、按subject划分及record-local 10Hz/stride16处理：

```bash
python prepare_data.py --source /srv/r2-data/data/raw/PAMAP2/PAMAP2_Dataset \
  --output /srv/r2-data/data/processed/pamap2_10hz_recordwise
python train.py --dataset PAMAP2 --scene sr015_bl8 --variant R2 --seed 2023 \
  --data-root /srv/r2-data --output-root /srv/pamap2-runs --device cuda:0
```

BeijingAirQuality和PhysioNet2012保留原数据加载器，需要已有一致的processed数据及manifest；此发布已核验ETTh1迁移流程与PAMAP2加载/前向，不把未重新运行的其他数据集声称为已验收迁移。

## 已有成绩和已知限制

`evidence/ETTh1_sr015_bl8.csv`保存三个seed逐项成绩与配对差；这是已查看的探索性场景，不是新的独立确认集。
完整R2：测试MSE 0.173034 ± 0.001244，MAE 0.257867 ± 0.000293。
PAMAP2 sr015_bl8 的完整R2：MSE 0.325488 ± 0.003916，MAE 0.229902 ± 0.001160；有效参数575,074。
旧“R0”在不同迭代中含义不同，请按 `provenance.json` 和实际入口绑定当前R2。

**H检索限制保留**：legacy_v1查询输入为窗口局部归一化观测，历史成员key基于训练集标准化数值；编码器不显式对齐两者。
构造例子已发现已知匹配成员可能排在另一成员后面，不能声明H可靠或只匹配形状。
训练记忆仅来自训练集，评估/校准只读；全缺目标patch不能使用H。审计数值见 `evidence/H_audit_ETTh1_sr015_seed2023.json`。
本次发布不静默修复H，避免把新结构与已有R2成绩混用。

底层实现来自同仓库既有代码及其第三方依赖，遵循仓库根目录 `THIRD_PARTY_NOTICES.md`，不将通用注意力或历史检索本身声明为创新。
