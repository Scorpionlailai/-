# E23 v56：原预测头下的纯MSE训练目标

状态：Stage1复用、Stage2新100轮及两套20组测试完成；15项检查、GPU预检查和最终审计通过。

v55验证0.759591/0.588738未通过v54门槛，按测试前决定保留v54。v55开发测试0.427958/0.435557虽然MSE略低，也不用于改父版本。v56预声明在v55结束前保存。

本轮依据用户先前的纯MSE训练要求，复用v54精确同一份Stage1权重、H key与library及资源记录，单独重训Stage2 100轮，比较训练目标。原MSE+0.2MAE改为纯forecast MSE，取消epoch11起0.05uniform修复保留。原PatchTST源码、配置、35427参数、初始化SHA、输入padding、原RNG初始化位置、训练窗口/种子不变；优化器仍freeze_front_v1：全CNN与H key/library冻结，其他修复lr1e-5、预测头lr1e-4。

严格保存MAE主checkpoint和MSE副checkpoint。两个验证指标/checkpoint先锁定，再按预声明规则在测试之前确定父版本：若MAE版的验证MSE低于v54且MAE不差，采用MAE版；否则MSE版通过同一门槛才采用MSE版；均未通过则保留v54。之后两版均测原20组位置，不能用新测试改选模。

ETTh1 96→96；0.15起点率、块8、允许重叠；原20组实际72.9252%；训练seed502、固定validation2021；Stage1原100轮已经测量完整成本，新Stage2 batch128/FP32。模型完整时间统计原Stage1实测＋新Stage2实测，另外报实际增量训练成本；两行checkpoint来自同一次训练，成本不重复累计。报告两组平均MSE/MAE、位置std、allocated/reserved峰值和selected epoch。

代码审核确认原预测器及修复模型源码保持。Stage2 loss实现保持；仅放宽配方校验并扩展既有副checkpoint保存条件，训练epoch中的梯度和优化器更新代码一致。15项检查覆盖相同Stage1/预测器初始hash、纯MSE有梯度、前端保持不变、保存加载及副checkpoint资源关联、旧规则兼容；GPU预检查使用4个真实训练窗口进行两步检查，正式Stage2重新初始化。

隔离源码：`/home/xyw/paper/.e23-v56-pure-mse/E23`。
实验记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v56_pure_mse_sr015_bl8_seed502_20261005`。
