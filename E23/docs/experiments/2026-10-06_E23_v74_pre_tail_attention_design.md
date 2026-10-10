# E23-v74 前置全窗修复注意力

## 依据
固定验证诊断：v73使≥32长段修复MSE从v72的0.567302降至0.521888，但last32从0.458737升至0.495508，预测验证MSE/MAE从0.945547/0.664178恶化到0.964824/0.677103；不采用v73。不能据此声称末端误差是唯一原因。

## 设计
本轮只把同一个SegmentAttentionReadout移到原point_refiner之前，保留所有模块、损失、参数维度和初始化。新模式segment_attention_pre_tail_v2；旧模式segment_attention_v1和direct_point_v1行为完全保留。原三专家/候选/置信度折叠→全窗长段注意力→原末端point_refiner→原recent_mean_projector→原预测器。注意力输出仍零初始化，长段长度≥16与全局单位、真实mask/validity完全相同。预测器35427参数，修复503086；与v73相比零新增参数。identity添加point_context_placement=before_point_refiner，避免旧配置重标记。

## 固定实验
ETTh1 96→96、起点率0.10/块长16、原20位置；seed2023、batch128/FP32、Stage1/2各100轮6600更新。原Stage1 globalMSE+.1retrieval+recent_mean_v2+long_gap_moments_v1。Stage2完整修复器冻结，原MSE+.2MAE、uniform、lr1e-4与padding。验证MAE主选、MSE次选，同轨迹先锁定再测试。仅验证双指标改进才能替换v72；T1与旧结果只读取缓存。新20确认位置保持未评分。

## 其他方向及取舍
跨变量全局读出可能改善长段证据，但会同时增加参数和改变信息来源。本轮先检验处理顺序这一更小的可诊断改动；若失败，再依据验证诊断决定跨变量路径或受约束残差。
