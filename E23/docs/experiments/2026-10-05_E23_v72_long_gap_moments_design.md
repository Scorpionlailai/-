# E23-v72：Stage1长连续缺失段监督

用户批准：优先在长缺失下超过T1，先实验Stage1长段重建监督，预测头与Stage2保持。以下固定方案在唯一正式训练前保存；源码、plan和controller已冻结于实验根。

## 唯一变化

以固定v69为底，保留全部H/V/T、CNN/signature位置前端、原历史库、top4、P16/S8及完全缺失H回退，修复器420973参数、预测器35427参数与原初始化/RNG保持。没有v70 TV查询或v71水平校准。

Stage1原global hidden MSE+0.1检索项+recent_mean_v2仍保留。额外从原始观察/有效mask识别隐藏连续段，L至少16；段在96点窗口边界处截断，观察或无效点将其分开。只监督原训练回溯缺失真值，不使用未来预测目标，不增加库成员或把修复值变成观测。

全局训练标准化残差e=(prediction-target)*detach(query_std)。每段u=linspace(-1,1,L)，a=mean(e)，b=sum(u*e)/sum(u²)。损失为0.2*sum(L*a²)/sum(L)+0.2*sum(L*b²)/sum(L)，分母只计L>=16的段。无长段时额外损失及梯度为零。它约束误差的低频成分，不强制真实信号线性或平滑。排除段和坐标在相减前清零，避免0*NaN/Inf污染；mask和std不接收额外梯度。

## 固定实验

仅已有sr010_bl16，ETTh1 96→96，起点率0.10/块长16允许重叠，20组测试点缺失比例平均约81.94%。seed2023，batch128/FP32，100轮Stage1和100轮Stage2，各6600更新，lr0.001/0.0001、weight_decay0。

Stage1仍以固定验证全局缺失MSE选模，不按新增损失选择。Stage2完整修复器冻结，原PatchTST及MSE+0.2MAE/uniform配方保持；验证MAE主选择，同轨迹额存MSE次选择，两者在测试前锁定。固定验证预测MSE低于同场景v69且MAE不变差才晋级；测试用于透明汇报，不用于改配置、seed或选checkpoint。

共享旧20个测试mask，v69/T1准确性和资源记录均缓存，不重训基线。记录平均MSE/MAE、ddof1位置标准差、配对胜出、完整和分阶段耗时、allocated/reserved显存，以及全窗口、近期和长段内部修复诊断。20位置不是20训练种子，不能证明跨种子/时间段泛化。T1与E23作者配方、种子、batch/精度/Stage2更新策略差异继续列出。

## 验证与隔离

只改integrations/e23/config.py和train.py；新增loss helper、两份行为测试和配置。9行为测试及4controller RED→GREEN；完整1153通过、9既有失败名称、1跳过，详见test-audit.json。原基线21失败中的12报告检查因当前隔离目录可读取已有缓存而通过，不重写旧基线记录。

审查发现被排除短段的NaN/Inf污染，补独立RED回归后改为eligible-point operands mask；重跑完整回归和CUDA预检并重新冻结源码。初版definition/archive/preflight均保留为initial-*。正式训练使用更正后的冻结版。

真实CUDA预检：原35427参数预测头初始化SHA一致；实际训练mask触发52长段，连续两步有限，Stage2优化后修复器hash不变。预检临时模型不用于正式训练，正式另行从头初始化。

工作区：/home/xyw/paper/.e23-v72-long-gap-moments/E23

实验根：/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v72_long_gap_moments_seed2023_20261005
