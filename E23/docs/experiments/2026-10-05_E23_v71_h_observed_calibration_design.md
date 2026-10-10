# E23-v71：H可见点水平校准与留一置信度

本记录整理已在对话中宣布的局部方案。配置及源码在正式训练前锁定，见实验根plan.json/frozen-definition.json；本说明于训练进行中补充，不作为事后调参依据。

## 动机和范围

仅训练回溯诊断1024窗口：H原始候选按现有权重混合的hidden MSE为1.068811；四个虚拟零偏移观测的ridge水平校准在保持原权重时降为0.767438。它是原始候选数值诊断，不是完整模型修复或预测MSE。无效槽位最小误差的初版相关性已排除；校正诊断保留，candidate MSE完全一致。

以固定v69为底，仅在原H候选准备完成后对齐局部水平；不采用v70的全缺失查询路径。原top4、阈值0.7、group/member查询、候选时间资格、实际bank/raw成员、mask不改变。

## 固定校准

对每个候选，r为真实有效可见坐标的query-candidate残差，n为坐标数。偏移b=sum(r)/(n+4)，限制到全局训练尺度的±2*sqrt(query_std²+0.1²)，平移整个候选后进入原H context/message/point readout。原始与对齐候选单独存provenance。

置信度逐个排除被评分可见点，重新以(n+3)拟合偏移并施加同一上限；被留出点的平均绝对误差代替旧visible_error进入原权重与quality。仅对偏移拟合做留一，窗口原始可见统计、候选集合和编码器不重新拟合。n=1时评分偏移为零，不能靠拟合该点提高自身评分。原全缺失H回退及无效槽位保持。

不增加训练参数，原修复器420973参数、预测器35427参数及初始化/RNG保持。Stage1原global hidden MSE+0.1retrieval+recent_mean_v2，无新辅助损失，无未来预测标签参与校准或评分。Stage2整个修复器冻结，原PatchTST及MSE+0.2MAE/uniform项不变。

## 唯一正式实验

ETTh1 96→96，P16/S8，约72.9252%场景起点率0.15/块8；seed2023，batch128/FP32，Stage1/2各100轮、6600updates，lr0.001/0.0001、weight_decay0。

Stage1固定验证全局hidden MSE选模；Stage2验证MAE主选择，另存同轨迹MSE次选择。两者在测试前锁定。验证预测MSE严格低于同场景固定v69且MAE不差才升级；测试不决定checkpoint或晋级。

共享已有20组测试位置，均值与ddof1位置标准差、配对差/胜出次数、阶段及完整训练耗时、allocated/reserved峰值均保存。v69与作者配方T1为缓存参考，不重训；不同训练seed/batch/精度及Stage2策略注明。单训练seed，不把20组位置当作20次独立训练。

## 验证与归档

7项新增行为测试RED→GREEN，4项controller检查；全套1139通过、21项与原517-source基线完全同名的既有失败、1项跳过。真实CUDA预检使用实际原可见训练patch构成的临时库，两次梯度有限、head初始SHA相同、Stage2优化器步后修复hash不变。正式训练另行从头初始化。

独立审查无Critical/Important/Minor问题，7项CPU检查及96个scalar ridge/LOO算例通过。521源码、1358受保护文件、controller/plan已锁定；源码归档于独立目录。不会覆盖v69或合入主源码。

工作区：/home/xyw/paper/.e23-v71-h-observed-calibration/E23

实验根：/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v71_h_observed_calibration_seed2023_20261005
