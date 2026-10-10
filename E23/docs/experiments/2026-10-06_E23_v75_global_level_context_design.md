# E23-v75 可见绝对水平进入修复主干

## 证据与假设
现有robust_local_v1前端5特征只提供中心化形状；signature_plain_pos_v3忽略传入mean/std，V calibration关闭，V消息和数值候选在局部坐标中。读取训练后v72、4个固定验证窗口的CPU信息探查：其他变量global mean +1（保持归一化可见形状/mask/std和目标mean）时，目标变量修复/预测变化精确为0。这是信息访问能力探查，不是实际因果识别。T1当前Stage2修复MSE0.603826但预测更优，进一步提醒总修复MSE不是最终判据。

## 单一改动
front_level_context=global_level_v1。在原window CNN的5输入后增加2通道：真实可见值在全局训练标准化尺度的数值，以及仅从这些可见值汇总的均值（广播到valid窗口点）。缺失/无效值在乘加之前遮掉，完全无观测变量的两个新增通道为0。后续变量signature传递这些输入形成的表示。原point证据接口仍返回5特征，readout/router维度不改。128新增修复参数，v72基础总421101；原预测器35427。

## 初始化与契约
先构造完整原CNN/signature，再在fork_rng内增加零初始化Conv1d(2,64,1,bias=False)，与原5通道卷积结果相加后进入原GELU。数学上等价于7通道输入投影；保留原5通道算子，避免更换GEMM尺寸造成零启动数值舍入变化。旧模块初始化/全局RNG/最终预测器保持。仅global_level_v1采用该新路径，none默认行为逐字节兼容。identity绑定策略、CNN总输入7/证据5、extra128及globalvisible来源，checkpoint严格拒绝重标记。允许的首轮配置为robust_local_v1/window_cnn_v1/signature_plain_pos_v3，已有三专家/候选/tail/mean/损失保留。

## 实验与条件
当前先准备实现，v74完成后若仍未达T1双指标目标才运行。本候选运行配置从v72（attention=none）单独添加2通道；若v74通过验证，则先将其作为更强验证对照，不把v74的注意力同时并入本配置。正式晋级依据运行前锁定的最新保留父版本，不能依据测试改变。Stage1/2原100轮、batch128/FP32、seed2023、损失和Stage2完整冻结、MAE主/MSE次选保持。T1与旧结果缓存。新20确认位置不评分。

## 取舍
方案1是2个全局数值通道，解决被丢弃的信息来源且参数最少，本轮采用。方案2是新增时间/变量axial读出，有更强表达但参数更多、同时改变聚合方式。方案3是Stage1未来辅助监督，保留既有最终预测头但改变训练目标，若数值来源无效再考虑。
