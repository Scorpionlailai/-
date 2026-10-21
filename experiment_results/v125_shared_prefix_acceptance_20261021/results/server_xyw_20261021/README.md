# 原服务器PAMAP2有限验收结果

状态：PASS。

来源为交接提交392fc08425371c5a7aad7db47b89440c9548ae7c，9份资产已核验。完整报告见report.md，原始结果/日志及四张CSV见run/。无test评分或Stage2，诊断权重不晋级正式模型。大型权重与验证数组仅保留服务器，路径/SHA详见result_manifest.json。

所有失败、资源门槛未通过及缺测项均保留；没有自动重跑或扩大预算。A4独立实验和experiment_results/v125_ablation未被本验收覆盖。

最终复核：132项严格一致性检查与864条验证指标已核对，更新段减少47.1857%，训练allocated峰值增加0.0461%，原推理路径最大中位回退0.0112%，全部通过登记门槛。A4两seed已从第57轮断点恢复原两卡队列。完整run.log、调度与恢复日志、post_run_verification.json均已归档。仅支持当前A0/seed2023两轮范围，未扩大训练。
