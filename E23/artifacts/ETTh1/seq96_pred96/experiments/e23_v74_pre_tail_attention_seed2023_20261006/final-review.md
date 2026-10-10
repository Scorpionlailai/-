# E23-v74 训练后独立最终复核

按 requesting-code-review 流程聚焦本轮最终产物。只读核对冻结文件、检查点、原始分数、资源、诊断及批准时间，执行CPU严格加载；未启动训练/GPU推理、重跑全套或修改现有源码/输出。唯一写入为本文件；原实现复核保持不变。

## 发现与结果

- **Critical：无。**
- **Important：无。**
- **Minor：无残留。** 初次检查发现MD长段诊断说明误写“v72与v73”，实际比较为v72/v74。根任务已修正报告及外部helper，独立复查标签和当前MD哈希链通过，数值CSV/raw未改。
- **裁定正确：拒绝v74，保留v72。** 固定验证MSE/MAE为0.9936831381315547/0.6850495911509542，相对v72增加0.04813579961556158/0.020871221884037672，未通过固定验证升级门槛。报告/audit只用验证决定保留版本，测试未参与checkpoint选择或晋级。T1目标未完成。

## 独立核对的证据

1. **冻结与身份。** 526源码、3079保护文件、129参考项及controller/plan去重3609路径全部SHA匹配，526源码归档逐项一致。两个Stage2 manifest/checkpoint SHA与选模锁及evaluation锁一致，均通过原loader身份检查、placement检查及CPU模型strict加载。独立构造确认repair503086参数、原预测器35427参数及seed2023初始化SHA与plan一致。
2. **单次轨迹与冻结。** Formal complete/exit0，控制台恰有一次Stage1和一次Stage2；两阶段日志均连续100轮。Stage1历史100轮/6600更新，按原global hidden MSE选epoch96；Stage2保持100轮、每轮66batch的原轨迹。MAE主/MSE次都选epoch99，全部318个model-state张量相同；各243个repair状态张量与Stage1逐项bitwise相同，memory/retrieval亦相同。双验证锁均在首次开发测试启动之前写出。
3. **批准时序透明。** approval记录的mailbox批准为02:00:25 UTC，formal启动为02:00:57.253677 UTC；实现复核文件mtime为02:01:00.852445 UTC，批准JSON记录为02:01:37.483130 UTC。approval中的review SHA与现有实现复核一致，明确document_saved_after_launch=true，并披露首次文档helper失败、后续shell行继续启动及没有重启。此前独立审查的实质通过消息先于formal，因此不要求补写文件先于启动，也不将首次文档gate误称成功。文件后补和未fail-fast的情况是已披露的执行历史，不改变单次训练或授权结论。
4. **统计与配对。** 检查100条raw draw（主/次各20，缓存v72/v69/T1各20），NPZ SHA、同20个mask seed/SHA、2688个target_start及目标元素数匹配。重算均值、ddof=1位置std及60条配对差；五CSV行数4/80/60/2/2，所有统计与选择值匹配。NPZ逐窗口double与原FP32批次汇总最大归约差1.563e-8。
5. **资源无重复。** repair阶段670.224152748473秒，forecast308.1597427809611秒，准备/保存等2.3449219008907676秒，总980.7288174303249秒（16.3455分钟）。峰值allocated/reserved为5.340568/9.400391 GiB，阶段峰值取max。真实路径去重只有一条Stage1和一条Stage2资源记录，次selector引用同一记录/成本；缓存基线未新增训练。训练时间排除最终测试及额外诊断，allocator峰值不是整卡显存。
6. **诊断与MD链。** generic修复JSON的来源manifest/checkpoint SHA正确，Stage1/Stage2 metrics完全相同、bank_state_unchanged为真。长段CSV逐项匹配raw JSON，v72/v74同38518个≥16段/1387503坐标及相同分区域计数；固定验证、全局训练标准化尺度、重叠窗口坐标出现次数，与测试预测指标分开。已选Stage1实际eval/inference来源不使用forecast_target，不追加T1推理。report-audit旧MD SHA等于long-gap-diagnostic-audit.report_before_long_gap_sha256，当前MD剥离追加诊断可重建旧SHA。标签修正后的最终MD SHA为dd4efd7d531d35fe005cd66d6e1faa62c5e7c0c0fe6c50457b8559ec535b4525；raw/CSV及最终链匹配。

## 结果与限制

主/次开发测试MSE/MAE均为 **0.5863106491884678/0.5143275963746502**。相对v72差+0.02474449729703181/+0.015545100419699736；相对T1差+0.058951239521000655/+0.03827276846178536。与v72、v69、T1比较，两指标均0/20胜出。不能声称前置读出改善预测或已超过T1。

长段整体修复指标改善，而长段最近32点MSE由v72的0.469959升至0.492846；这些诊断不证明预测收益或因果机制，也不支持推翻失败裁定。单seed、同一未来区间的已曝光20开发位置及位置std，不衡量训练seed波动或跨数据集泛化；T1的seed、batch、精度及Stage2配方不同。

未重新执行修复/预测推理，诊断数值追溯至实际推理保存的JSON、来源代码和检查点；测试统计由保存draw/NPZ独立重算。没有运行新20确认位置、publisher或delivery verifier。v75实现与确认位置全流程不属于本复核。

**最终就绪性：** 无残留Critical/Important/Minor，v74按失败实验归档；保留v72继续自主迭代，整体目标仍active，不能标记完成。
