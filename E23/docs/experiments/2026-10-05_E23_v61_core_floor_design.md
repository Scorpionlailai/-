# E23 v61：整个修复路径的输入尺度下限

状态：两阶段100轮及20组测试完成；20项检查、GPU预检查和最终审计通过。

v60只对齐预测输入padding，未通过固定验证门槛，按它测试之前保存的决定继续以v58为父版本。v61在v60评分之前声明，不根据测试重新选择父版。v59开发测试领先的权重与结果保持。

## 改动

v54只在CNN/点级读取和tail数值读出中加稳健尺度，原base/H/V/T/decoder仍使用可见窗口std。本轮在修复器入口设置r=sqrt(original_visible_std²+0.1²)，把原真实可见中心值转换为r单位，整个原修复路径使用query_std=r，对应键配对监督也转换为同样单位。原window mean及真实观测mask不变，bank原raw空间和观测资格保持。

原点模块继续自身稳健公式，它们的有效数值尺度因此是sqrt(original_std²+2×0.1²)。修复结束后，按原真实时间点转换回原visible std坐标，原真实可见值逐位保持，按原16/8几何repatch。coverage/mask/validity保持。不加任何参数或未来标签；不声称此改动解决全部H原raw bank和查询的尺度差异。

原PatchTST源码、配置、35427参数、seed502初始化SHA保持，E23ForecastModel AST相同，预测接口继续原query_mean/std；Stage2训练函数/objective与v60兼容实现相同，优化配方及padding由验证保留父版继承，本轮仍原zero_invalid_v1。该改动改变修复函数，不要求初始修复输出相同；全部参数和随机初始化相同。

## 实验与审核

ETTh1 96→96；起点率0.15、块8/重叠、已有20位置实际缺失72.9252%，seed502/validation2021。从头Stage1/2各100轮/batch128。Stage1 global hidden MSE、0.1key loss、近期均值MSE权重1、专家曝光0.2保持，修复选模仍global hidden MSE。Stage2 MSE+0.2MAE、epoch11起0.05uniform保留，原CNN/Hkey/library冻结，其他修复lr1e-5、原预测器lr1e-4。

预测checkpoint按固定验证MAE选择；验证MSE比父版低且MAE不差才升级，在测试之前保存父版决定。之后旧20位置报告MSE/MAE及位置标准差、完整两阶段时间、训练allocated/reserved峰值。如果超过T1，再锁定后用新位置确认。

20项检查覆盖原参数/RNG/head无变化，内部输入/统计确实采用r，输出回原单位和原观测逐位保持，global loss梯度、微小std、canary、全空/全可见及validity、key配对同尺度且不读forecast labels、checkpoint及身份校验、与均值校正/padding组合、旧均值模块及前端冻结。GPU预检查两步真实训练样本，正式全部重新初始化。

代码：`/home/xyw/paper/.e23-v61-core-floor/E23`。
记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v61_core_floor_sr015_bl8_seed502_20261005`。
