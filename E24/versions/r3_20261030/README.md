# R3 最终定版 · 2026-10-30

用户确定的最终模型：**原完整 R2 + H train-global 坐标对齐**，运行内部标签 `H_GLOBAL`，CLI 别名 `R3`。继承 R2 发布 `46b6a01d7c67add15ea4d3144f20849d60016f88`。模型代码 `aligned_h_model.py` / `aligned_h_candidates.py` 与原 R3 实验逐字节一致。不是去 Confidence、消息去 z0、H_GLOBAL_VISIBLE 或后续 T/V 变体。

保留 CNN、Signature、基础 Decoder、原 T/V/H 候选、八头点级读取、学习式 Confidence、单支64校准与后端邻点。patch16/stride8，主干64，H top4、阈值0.7、legacy_v1 库更新规则不变。H 可见查询恢复到训练全局标准化尺度，辅助监督使用相同尺度；返回的历史数值仍换回目标窗口局部坐标。

## 多服务器使用

```bash
git clone --branch release/r3-final-20261030 https://github.com/Scorpionlailai/-.git paper-r3
cd paper-r3
# 正式结果必须记录 git rev-parse HEAD；后续服务器按给定完整 commit checkout --detach。
cd E24/versions/r3_20261030
# Python 3.10/3.11；先按 GPU 安装 PyTorch，再安装其余依赖。
# 此服务器使用 torch2.7.1+cu128，RTX5090；不要为5090安装旧cu124 wheel。
pip install -r requirements.txt
python selfcheck.py --dataset ETTh1 --data-root /srv/paper --assets /srv/r3-check/ETTh1

# 先做小流程；两阶段各2轮、每轮1batch，结果只用于验收。
python run_scenes.py --dataset ETTh1 --scenes sr015_bl8 --device cuda:0 \
  --data-root /srv/paper --output-root /srv/r3-smoke --smoke-batches 1

# 验收后正式运行四场景，默认单seed2023；本次发布不自动启动正式训练。
python run_scenes.py --dataset ETTh1 --device cuda:0 \
  --data-root /srv/paper --output-root /srv/r3-results
```

每服务器可分配一个数据集；更换 `--dataset` 即可。四场景为 `sr010_bl8 sr015_bl8 sr020_bl8 sr010_bl16`。sr表示块起点采样率，不是实际缺失比例。必须在每台服务器逐数据集 preflight/smoke 后再跑完整预算。

| CLI数据集 | 数据根目录下必要资产 | 既有协议 |
|---|---|---|
| ETTh1 | data/raw/ETTh1.csv | 0:8640/8640:11520/11520:14400；验证/测试2688窗口 |
| Weather | data/raw/weather.csv | 52696行，36887/42157/52696分界 |
| Exchange-Rate | data/raw/Exchange-Rate/exchange_rate.txt | 7588×8；5311/6070/7588分界；真实数值原txt、行索引时间 |
| BeijingAirQuality | data/processed/beijing_pm25_12stations.csv 及其 natural_missing.npz | 原12站点 PM2.5 和有效观测口径 |
| PAMAP2 | data/processed/pamap2_10hz_recordwise/，含manifest与分片 | 原record-local/受试者划分与原生有效mask |
| PhysioNet2012 | data/processed/physionet_15min_recordwise/，含manifest与分片 | 原患者划分/15分钟处理与原生有效mask |

原始数据、处理后分片、历史库和 checkpoint **不随本次代码发布上传**。它们应另行同步，尤其 PhysioNet 本机没有处理后资产，不能把入口可用解释为全部六数据集已验收。准备脚本/参考协议来自父发布和原数据归档；不要自行重做预处理后直接拼接旧基线成绩。Exchange 新入口按已登记协议接入，发布核验会对原mask逐值比对。每次结果保留 source、split、scaler、mask、处理manifest哈希；跨服务器先比对再汇总。

## 固定训练与选模

- 主干100轮；校准100轮。batch128、Adam lr0.001、weight_decay0，无本轮新增早停/搜索。开发默认模型seed2023；正式多seed的数量另行确定，不挑最优seed。
- 训练mask和顺序seed为2023+epoch；20组测试位置不是20次独立训练。
- 主干损失：hidden MSE + recent32均值残差平方 + 0.1检索辅助；按验证full96 MSE选主干。
- **校准冻结完整粗模型与历史库，包括level等全部粗阶段参数**。校准损失：hidden MAE + 0.25 recent32 MAE；按验证full96 MAE选最终checkpoint，最低MSE权重只作敏感性记录。
- 上述recent32项是原R3训练目标，保留定版，不在重新运行时暗改成新模型。
- 本发布仅修复训练和评价；**不运行预测Stage2**。未来预测必须冻结完整修复器、使用相同预测头和纯MSE协议，另设输出目录。
- 历史库仅训练集构建；原生缺失不进入监督真值；隐藏值不得进入输入查询。校准/评价只读库。
- 用同一选定模型同时提取校准前后结果，20组固定测试mask平均，并保留逐mask值；训练seed统计与mask波动分别报告。

## 结果与资源

正式输出在 `runs/<dataset>/<scene>/seed_2023/H_GLOBAL/{main,calibration}`，评价和导出在 `evaluation/<dataset>/<scene>/seed_2023/H_GLOBAL`。两阶段各保存curve、binding、资源和checkpoint。完整训练目录可恢复；代码/数据/协议变化必须新输出目录，绑定检查禁止静默混用。

`complete.json.seconds` 是训练循环（含轮内验证/保存）的累计时间，**不等于数据准备在内的全流程时间**。`run_scenes.py` 额外在 wallclock/ 保存包含数据准备、两阶段训练、验证/保存、20mask评价与导出的本次端到端墙钟 attempts.jsonl（恢复运行是本次尝试时间，不冒充从头总时间）。汇报须同时保留外部墙钟时间，峰值allocated/reserved显存，GPU型号、Torch/CUDA、batch与训练更新次数。此发布没有声称取得新性能结果；已有责任分离实验在 evidence 中，历史本机绝对路径仅为来源记录。

本版的科学身份由 `model_identity.json`、`release_lock.json` 与 `vendor_lock.json` 固定；旧实验checkpoint严格绑定旧源码/协议，不能直接放到新runs跳过检查。原checkpoint回放通过独立核验记录说明，不修改旧权重。模型模块与阶段含义见 evidence 最终报告；通用依赖与第三方代码遵循仓库根 THIRD_PARTY_NOTICES.md。
