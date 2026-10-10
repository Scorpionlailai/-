# 历史检索坐标待隔离检验

当前v72/v76 legacy_v1：train.py _retrieval_loss以normalized_observed/normalized_target训练query-clean keys；historical_candidates.py prepare_candidates以normalized patch values进行frozen query；_stage_observed及_publish_epoch中的库原型keys来自raw train-standardized patches。这是代码可见的训练/query与库编码输入尺度差异。若window mean/std变化，两类同一原值的编码输入不一样。不能仅据代码断言性能影响或唯一瓶颈。

旧global_diverse_v1(v57)同时改坐标、variance/covariance losses、每stepEMA，负结果不能单独否定坐标对齐。若v77仍未满足目标，可隔离候选global_aligned_v2：仅将训练masked query/clean和推理query改用raw global train standardized，历史bank本就raw不变；保留legacy pair+contrastive loss/.1/epochEMA、候选top4/.7/资格、旧mask与Stage2。新mode严格身份绑定，oldlegacy bitwise保留，原head不变。优先级暂在v77预测验证之后；未实现/未训练/未声明有效。
