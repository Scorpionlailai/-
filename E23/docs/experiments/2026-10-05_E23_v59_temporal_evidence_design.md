# E23 v59：可见时间证据的多头重复读取

状态：两阶段100轮及20组测试完成；19项检查、GPU预检查和最终审计通过。

本方案在v58评分之前声明，按照其测试前保存的验证决定，以v58为父版本。v58新增近期均值校正及Stage1均值监督通过原验证门槛；本轮继承它们，仅增加修复器中的时间证据读取器。

## 新模块

三专家、原八头候选读出、重叠融合及tail修复保持。在tail修复之后、近期均值校正之前，最后32点作为query，用宽64/8头/2层cross-attention重复读取同变量96点的真实可见source。query来自原CNN特征、路由潜表示、当前修复值、原可见特征及固定变量身份；source来自只读原可见输入的CNN特征、原可见特征及固定变量身份。修复值只用于query，不改变source原mask。每个头8维，拼接后64维。

距离先验尺度为8/12/16/24/32/48/64/96，另学习每头191个相对位置偏置。原真实不可见source权重严格为0，附加零值null保障全空source有限；每层有0.1残差的attention与FFN。最后数值读出零初始化，保证原参数/RNG/函数和预测头初始化相同。增量以稳健相对坐标限幅±2，只应用到最后32点中的原真实缺失值；原观测、前64点、mask/validity/coverage保持。新增83969修复参数，整修复356846。

只读取当下96点回溯窗口，不读取未来输入或标签；Stage1损失保持父版全局hidden MSE、0.1键损失、近期均值MSE权重1、专家曝光0.2。其修复选模继续使用固定验证global hidden MSE。

## 固定实验

ETTh1 96→96，起点率0.15、块8/重叠，已有20组平均实际缺失72.9252%，seed502/validation2021。Stage1/2从头各100轮/batch128；Stage2原训练函数和objective AST一致，CNN/Hkey/library冻结，其他修复lr1e-5，原PatchTST lr1e-4，MSE+0.2MAE和epoch11起0.05均匀修复保留均不变。原预测头35427参数、源码/配置/seed502初始化保持；相对位置模块仅属于修复器。

按固定验证MAE选预测checkpoint；验证MSE比父版低且MAE不差才升级，决定保存后再测试旧20组缺失位置。报告MSE/MAE、位置标准差、完整两阶段时间和allocated/reserved峰值。若超过T1，锁定后另建同场景独立位置确认。

19项检查：零初始化下原参数/RNG/head/函数相同；真实可见source限定、全空null有限、非零只更新近期缺失、canary与微小std、全局loss梯度、checkpoint/身份校验、均值校正组合、旧前端冻结。GPU预检查两步真实训练样本，正式训练重新初始化。

代码：`/home/xyw/paper/.e23-v59-temporal-evidence/E23`。
记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v59_temporal_evidence_sr015_bl8_seed502_20261005`。
