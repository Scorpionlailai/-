# E23-v78 历史键训练/查询/库坐标对齐

## 依据
当前legacy_v1：_retrieval_loss用窗口归一化query/clean训练keyencoder，prepare_candidates也以窗口归一化query编码；历史库_stage_observed/_publish_epoch则用原始train标准化patch编码。坐标不一致可以从代码直接确认，但不能据此断言具体性能损失。旧v57global_diverse_v1同时加了var/cov正则和每step EMA，其负结果不独立否定尺度对齐。

## 单一算法改变
新historical_key_policy=global_aligned_v2。仅把训练query/clean与推理query转换成和库一样的train standardized global坐标；hidden/invalid先mask再做values*std+mean，query缺失位置0；clean过去真值只作训练loss teacher，未来不进入此loss、forward、历史库。原mask/候选因果资格/top4/threshold.7/本轮冻结keyencoder/候选quality与message保持。整个缺失patch依旧无H资格，不能声称新增全缺失覆盖。

损失保留原masked pair+0.5*contrastive/log(count)、权重.1，不增加diversity损失。EMA依旧epoch结束一次，历史库仍只stage完整原始可见patch、按同epochEMA重新编码。训练/query尺度与原型库对齐是本轮变化；legacy/diverse两旧mode完整保留，新增0参数，模型各权重/RNG/最终head初始化不变。空库初始函数相同；有库时query坐标本来就会改变，不能声称其初始检索结果一样。

## 条件父版与评估
父版须由v77固定验证决定：v77通过才继承global_level_mean_v2/repair424238，否则沿用v76mean_v1/repair420973；保留forecast_mse_v1 .05/warmup10/seed2022/discarded。正式前锁定；测试不决定继承。原Stage1其余loss、100轮/batch128；原Stage2head35427/seed2023/freeze/MSE+.2MAE/uniform/lr1e-4/zero_invalid及100轮保持。Stage1global修复MSE选、Stage2valMAE主/MSE次先锁再20开发tests。T1缓存不重训，新20确认保持未评分。资源全量计入，不宣称已有效。

## 检查计划
七项新增RED→GREEN：oldweights/RNG/head与空库初函数；训练query-clean/推理query实际输入均是global；相同global信号换局部mean/std得到同querykeys；隐藏NaNInf不进入编码；future不改变retrievalloss且有finitekeygradient；epochEMA次数与原bank成员只来自可见；严格checkpointmode/坐标/EMAidentity篡改拒绝。之后真实CUDA +5controller、冻结/archive、独立review、formal及资源/配对报告。
