# E23-v100：Stage1 细化固定验证缓存验收

## 结果

ETTh1 sr015_bl8，seed2023/batch128，同v97参数与训练预算。细化100轮完整耗时 **306.624→263.969秒，减少13.91%**。缓存构建0.887秒，已计入本次完整耗时；CPU张量存储579.0MiB，21批/2688验证窗口。

完整100轮验证历史、选中epoch、选中的全部修复模型张量、历史库与key散列都与v97完全一致。按此等价性沿用原v97修复测试与对应预测结果，没有重复训练预测头或重测已有测试组。**v100是验证执行优化，修复结构和学习结果与v97相同。**

## 实现范围

修复细化阶段的专家骨干与历史库已冻结，只训练均值投影和dense refiner。首次固定验证时在 RecentMeanProjector 输入处捕获粗修复、router latents、窗口point features与标准化统计量，复制到CPU。随后每轮验证只执行当前的可训练后缀并计算原指标。训练动态缺失及全部训练forward不使用缓存。缓存只含验证输入与评分目标，未包含测试或未来预测目标。

模型/历史库冻结检查继续每轮执行；构建检测CPU与CUDA随机状态不变，验证重放检查骨干hash未变。所有计算均为FP32。仅适用于骨干完全冻结的细化阶段；粗修复阶段不能这样缓存。

## 资源口径

本表仅比较细化100轮，不能把264秒作为模型总训练耗时。原v97对应父粗修复100轮成本为571.690秒，修复两阶段合计为878.314秒；按原父模型计入成本，本次为835.660秒。Stage2本次未重跑，总预测训练资源继续引用原v97记录；不能宣称本次实测了一整套300轮新流程。GPU峰值是allocator allocated/reserved，CPU缓存单独报告；尚无完整多变量训练结果。

Stage2固定验证缓存此前在另一场景完成100轮权重等价性验收，可以在独立后续流程组合使用；这里没有测量两项缓存组合的全流程耗时。

## 可查验资料

- [资源CSV](/home/xyw/paper/E23/docs/experiments/E23_v100_stage1_validation_cache_20261008.csv)
- [扩展验收](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v100_stage1_validation_cache_20261008/extended_acceptance.json)：含100轮历史散列与完整模型/历史库一致性。
- [缓存实现](/home/xyw/paper/E23/versions/e23-v100-stage1-validation-cache-20261008/E23/integrations/e23/repair_validation_cache.py)
- 隔离源码、完整命令、训练日志、权重及源码归档：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v100_stage1_validation_cache_20261008`。

E24后台运行代码及封存v79未修改。
