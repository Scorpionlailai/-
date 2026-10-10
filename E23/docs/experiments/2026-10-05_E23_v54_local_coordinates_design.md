# E23 v54：可见窗口相对尺度修复

状态：两阶段100轮及20组测试完成；19项检查、GPU预检查和最终审计通过。

## 依据与唯一改动

v53的大前端训练修复误差约0.06，但末期固定验证约0.60，开发预测MSE0.497561；拒绝升级，保留v51验证父版本。训练mask按epoch与sample identity刷新已核实，不是静态mask错误。

本轮基于v51。其新增CNN及逐点证据读取全局训练标准化值，候选读出直接输出全局数值残差；原尾部refiner读取未加下限的局部标准化值。改变这些数值分支的坐标，测试绝对水平敏感性和小可见尺度是否妨碍泛化。

r=sqrt(visible_std²+0.1²)，每变量从原真实可见窗口计算。CNN feature0、逐点router的四个数值备选、overlap的当前候选和分歧、候选readout的数值query改为(original_global_value-visible_mean)/r。readout输出delta乘r得到全局残差，再除原query_std进入原显式接口。tail32 refiner把原local值乘query_std/r，输出乘r/query_std；范围仍为末32个真实缺失点。

H/V/T本体、bank检索坐标、质量和选择资格保留；未添加v52身份分支或v53变量注意力。没有新增参数，总修复259628。整体包含原H等分支，故不宣称完整模型对所有shift严格等变。尺度下限用于数值稳健，不能宣称严格scale invariance。

## 固定约束与预先判断

ETTh1 96→96，起点率0.15，块8，原20位置平均实际缺失72.9252%；seed502、固定validation2021。Stage1/2都从头100轮、batch128，原全局隐藏MSE＋0.1retrieval、曝光0.2；Stage2原MSE＋0.2MAE、epoch11起0.05均匀保留、整个CNN与H key/library冻结，修复lr1e-5、预测器1e-4。Stage2函数及损失AST与v47一致。

原PatchTST配置、代码、35427参数及seed502初始化SHA不变；预测器仍使用原visible-only query_mean/std。固定验证MAE选checkpoint，验证MSE降低且MAE不差于v51才升级；选择在20组测试之前保存。旧结果全部保存保护。每项报告平均MSE/MAE、mask标准差、完整两阶段时间、allocated/reserved GiB。

## 检查与保存

检查中心平移下局部分支特征不变、数值残差恢复到全局单位、1e-6尺度下梯度有限、隐藏NaN/Inf隔离、原可见值不变、原初始函数和RNG/预测头一致、checkpoint身份/加载、Stage2完整前端冻结。19项CPU检查及GPU预检查通过。

源码：`/home/xyw/paper/.e23-v54-local-coordinates/E23`。
记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v54_local_coordinates_sr015_bl8_seed502_20261005`。

若超过T1开发结果，再锁定权重，用独立新同场景测试位置确认；旧20组的反复开发结果不能单独证实论文优势。
