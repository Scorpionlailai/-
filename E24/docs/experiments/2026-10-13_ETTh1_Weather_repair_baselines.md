# ETTh1 / Weather 修复基线最终结果

TimesNet、ImputeFormer、BRITS、CSDI在ETTh1与Weather的四种块缺失场景均已完成：32项修复训练与评估、32项冻结修复器的公共PatchTST预测，每项采用20组测试缺失掩码。所有修复输出在本地封存，预测直接复用；640组测试修复输出、1408个缓存文件与选中权重身份校验通过。

## 查看结果

- [六个模型同表：新增基线、v125与T1](../../artifacts/repair_reproduction_20261010/report.md)
- [新增32项结果完整精度、标准差与配置CSV](../../artifacts/repair_reproduction_20261010/results.csv)
- [与v125/T1的48行对照报告](../../artifacts/repair_reproduction_20261010/comparisons/final_repair_forecast_vs_v125_t1_20261013.md)
- [48行对照CSV与来源校验和](../../artifacts/repair_reproduction_20261010/comparisons/final_repair_forecast_vs_v125_t1_20261013.csv)
- [完整性与冻结审核](../../artifacts/repair_reproduction_20261010/review_status.json)
- [完整修复输出文件、权重状态与统计核验](../../artifacts/repair_reproduction_20261010/final_saved_output_audit.json)
- [实际配置、执行与交接记录](2026-10-10_remote_repair_reproduction_execution.md)
- [复现入口](../../integrations/repair_baselines/run.py)

ETTh1新增基线使用用户确认的独立重建掩码，v125/T1旧记录只作数值参考。Weather测试掩码资产已核验，但新增基线预测训练采用固定掩码缓存，与旧参考训练协议不同。旧Weather v125的20%×8及10%×16预测尚无完整本地参考记录，明确留空。

TimesNet使用作者目标数据集配置；ImputeFormer、BRITS、CSDI使用获用户允许的作者基础配置，来源见各运行`plan.json`。后续CSDI采用100轮训练、75/90轮学习率衰减、50扩散步、评价5样本中位数；早期Weather前两项保留200轮及评价10样本。预测训练缓存使用1个CSDI样本，公共PatchTST训练100轮。

仓库保存结果、逐组指标、选模记录、配置、来源与缓存校验和、复现和诊断代码。修复序列数组与模型权重保留在执行机器，按照仓库规则不纳入Git；缓存目录中的JSON记录可核对对应输出的身份。

用户已于2026-10-13明确授权将当前结果同步GitHub。实际推送状态及提交身份见[发布记录](../../artifacts/repair_reproduction_20261010/github_publication_status.json)。
