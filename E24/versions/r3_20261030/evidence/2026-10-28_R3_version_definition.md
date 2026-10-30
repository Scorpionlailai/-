# R3 版本定义与后续实验基线

用户于 2026-10-28 指定：H 坐标对齐版正式标记为 **R3**；点级上下文与 Patch 证据职责分离实验以 R3 为基线。

## 唯一版本身份

R3 = 原完整 R2 + H 查询及检索辅助监督的 train-global 坐标对齐，对应原运行标识 `H_GLOBAL`。

- H 查询：可见窗口局部值先恢复至训练集全局标准化尺度，再编码；隐藏位置不进入查询。
- H 辅助监督：masked query 与 clean 训练目标统一使用 train-global 尺度。
- 历史库资格、时间边界、成员重编码、EMA 发布、top-k=4、阈值0.7及返回候选的局部坐标转换沿用原版。
- CNN、Signature、基础预测、Query、T/V、八头读取、学习式 Confidence、单支64校准与后端邻点均保留。

本次仅建立正式版本名，不修改已训练包、权重内容、运行目录或旧实验标签。旧记录中的 `H_GLOBAL` 按此定义对应 R3。不是 `H_GLOBAL_VISIBLE`，不含其他后续候选改动。

## 可复核来源

源实现目录：`/home/xyw/r2_H_coordinate_20261027/E24/versions/r2_20261026`。

已有 ETTh1 sr015_bl8、seed2023 主干与校准权重位于：
`/home/xyw/paper/E23/artifacts/performance/r2_H_coordinate_20261027/runs/ETTh1/sr015_bl8/seed_2023/H_GLOBAL`。

代码及上述权重的 SHA256 见同目录 `R3_model_identity.json`。已有结果不因版本更名产生新的性能结论，不将历史中断的种子视为已完成。

## 新实验命名

后续审计、融合三方案及候选编码去条件化实验均以 R3 为基线。三训练种子、20组测试位置及两阶段实验要求沿用本轮用户要求；先做实际代码审计，再设计修改。本文件不表示新实验已经启动或完成。

> 后续记录（2026-10-28）：R3点级上下文与Patch证据职责分离实验已完成。最终结论、三阶段结果与成本见 [最终报告](2026-10-28_R3_responsibility_separation_final_report.md)。原R3默认版本保持不变。
