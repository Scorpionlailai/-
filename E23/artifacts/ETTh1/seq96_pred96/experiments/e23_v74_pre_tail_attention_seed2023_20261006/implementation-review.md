# E23-v74 前置注意力独立实现复核

依照 requesting-code-review 流程，仅复核本轮placement改动、identity及相应测试/控制证据；旧模块以已锁定v73源码和已有复核为依据。未改冻结代码、启动训练、GPU工作或重跑全套；CPU操作为前向、无优化器更新的backward、旧检查点加载及哈希审计。唯一写入为本复核文件。

## 发现与就绪性

- **Critical：无。**
- **Important：无。**
- **Minor：无新增发现。**
- **就绪性：** 符合批准的有界设计，可启动唯一正式seed2023轨迹。复制来源为v73实现，验证晋级父版本仍为保留的v72；本结论不采用v73失败权重，也不预测v74是否优于T1。整体目标仍未完成。

## 已独立确认

1. **改动范围与旧模式。** 父源码524份校验一致，原文件仅model/config/train三处改变；SegmentAttentionReadout及tail/mean本身逐字节未改，新增为配置和7项行为测试。新模式使用同一注意力类，仅在point_refiner前调用一次；旧segment_attention_v1/direct_point_v1仍在原位置。独立forward hooks确认旧顺序tail→attention→mean，新顺序attention→tail→mean；每个模块返回的explicit对象原样交给下一个模块，原tail输出也确实进入mean。
2. **初始化与参数。** 新/旧模式继续由fork_rng构造同一读出；独立测试确认零输出时既有参数、初始化RNG、预测器初始SHA及eval预测结果保持。修复503086、原预测器35427参数，与v73相同，零新增参数。CUDA预检的原预测器初始化检查与上述计数通过。
3. **范围与隔离。** 注意力仍直接只写入原mask/validity划出的长度≥16隐藏段，global残差单位和小std转换不变；独立前向检查覆盖前64长段、短段、无效点分段、原观测及coverage。额外完整pre-tail链backward probe使用非零读出权重，将隐藏raw值及无效observed/hidden输入替换为NaN/Inf：输出与参数梯度有限，被排除raw输入梯度精确零，attention梯度非零；没有optimizer更新。空观察有效窗口及原all-invalid拒绝契约保持。
4. **严格加载。** 新identity在原mode、维度、scope和单位字段之外绑定point_context_placement=before_point_refiner，并进入manifest及checkpoint的严格验证路径；旧mode identity未加入新字段。独立CPU加载真实v73已训练检查点及模型strict state加载成功，运行模式仍为旧placement；同一检查点在v74新配置下被拒绝（配置SHA mismatch）。归档新行为测试另包含两种选择的完整冻结roundtrip和篡改placement拒绝，未重复其训练步骤。
5. **测试证据。** 独立非训练前向测试9通过、4个梯度/训练checkpoint测试未重跑；独立controller四项通过。新7项RED→GREEN、旧6项GREEN证据齐备。完整回归日志1166通过/9失败/1跳过，9个失败名称与锁定v73原回归逐项相同；没有新增失败。controller记录为适配后的GREEN，不声称本轮重新取得controller RED证据。
6. **CUDA与冻结。** 读取实际填充训练历史库的CUDA两步预检：有限loss，52个长段，第二步attention梯度0.261446、position梯度0.212062；Stage2全部repair冻结且参数哈希不变，未评分测试。526源码、3079保护文件、129参考项及controller/plan去重为3609路径，全部SHA匹配；526源码归档逐项一致。controller绑定新mode/placement、seed2023、100轮、原full-frozen Stage2 recipe及先锁定后开发测试。

## 检查限制

注意力直接门控范围不等同于完整组合模型的最终变化范围：后续既有tail/mean仍按原规则修改其eligible hidden点，这是本轮设计的下游处理。零初始化检查只保证初始状态与eval函数，不保证改变调用顺序后各dropout的训练随机轨迹相同。

GPU和完整回归结论来自原始归档证据，未由复核者重跑；本轮正式权重、243个repair张量冻结、最终资源、验证/开发分数和诊断尚待训练后复核。没有运行新20确认位置或将开发20位置表述为盲测；本复核不判断跨轮确认流程。

**结论：** 无残留Critical/Important/Minor，实现复核完成，批准范围内可继续正式实验；晋级只依据固定验证，是否达到T1目标另行判断。
