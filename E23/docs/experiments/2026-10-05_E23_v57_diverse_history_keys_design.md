# E23 v57：有区分度、同尺度的历史检索键

状态：两阶段100轮及20组测试完成；14项检查、GPU预检查和最终审计通过。

## 依据与预先锁定

基于v54。v55及v56纯MSE双选模都未通过v54验证门槛。v57方案在v56测试之前声明，继承其验证保留父版本，不按新测试选模型或修改方案。

对v54选中的Stage1训练历史库做只读诊断：不同片段同变量的冻结键平均余弦0.993763–0.995857，坐标std均值0.00723–0.00929；在线键相似度0.999292–0.999541，std约0.00234–0.00286。键明显接近同一方向，难以区分历史形态。该诊断只读训练bank，未取测试误差或未来标签。

现有原型bank键来自全局训练尺度raw patch，查询及配对却来自各窗口局部标准化值。原EMA decay0.99每epoch更新一次，选中epoch92时初始参数系数0.396678。这是待比较的设计，不把文档说明不同直接当作已证实性能原因。

## 新H键政策：global_diverse_v1

1. 键训练clean/raw及可见query、推理query统一使用训练全局标准化尺度。先屏蔽原隐藏值；bank原raw内容保持其原口径。
2. 只把damaged但有原真实可见点、且全patch有真实有效坐标的patch用于clean/masked键配对。空patch不能读取H，padding不当完整clean正例。
3. 保留cosine alignment及0.5*contrastive/log(count)，温度固定0.1。clean键另加按变量的方差约束：std=sqrt(mean(centered²)+1e-6)，惩罚mean((relu(0.1-std)/0.1)²)，权重0.5；clean键非对角协方差平方和/维度的变量均值权重0.05。正则直接对clean分支反传，不能只靠mask模式实现内容区分。
4. EMA仍0.99，但每次optimizer.step后跟随；query view在epoch内继续冻结，epoch结束对bank重编码/发布，Stage2全bank/key冻结规则保持。new模式结束时不再额外做一次EMA更新。

这是耦合的键策略，未单独证明其中每项作用。不增加参数，总修复259628。H原始候选top4/阈值0.7、因果资格、V/T、窗口CNN、v54稳健相对修复坐标、point/tail读出、原预测头均保持。没有v55变量条件。旧模型/记录始终用legacy政策重放。

## 固定实验与审核

ETTh1 96→96；起点率0.15、块8、重叠，旧20位置平均实际72.9252%；seed502/固定validation2021。从头Stage1/2各100轮，batch128。Stage1原全局hidden MSE+0.1key loss、曝光0.2；Stage2继续原MSE+0.2MAE、epoch11起0.05uniform保留、CNN/Hkey/library冻结、其他修复lr1e-5、原预测器lr1e-4。

按验证MAE选checkpoint，验证MSE低于v54且MAE不差才升级，决定在原20组测试前保存。每版报告完整两阶段时间、显存allocated/reserved峰值、平均MSE/MAE和位置标准差。原PatchTST35427参数、源码、配置、初始化保持；Stage2函数/损失AST与v56兼容实现相同，优化配方未采用v56纯MSE。

14项检查：抗近常量键正则及梯度；原参数/RNG/head不变；推理读取全局可见值；空源有限回退；键目标不读取forecast_target；每optimizer step一次EMA而epoch只发布一次；Stage1/Stage2加载与key冻结；身份篡改与未知政策拒绝；旧前端冻结/纯MSE兼容。GPU预检查两步真实训练样本修复/键损失有限，正式全部重新初始化。

代码：`/home/xyw/paper/.e23-v57-diverse-history-keys/E23`。
记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v57_diverse_history_keys_sr015_bl8_seed502_20261005`。
