# E23 v45：移除末端精修，独立比较修复粒度

状态：三组训练、双checkpoint的20组测试及修复诊断全部完成。

## 用户要求

不融合CNN/Transformer，仅尝试不同修复粒度。预测头保持。

## 预先固定设计

- 三组输出P/S为16/16、8/8、4/4，块数6、12、24。
- H/V/T保持64维、top-k=4，单轮T→V→H；末端point_refiner为None，没有新增CNN/attention。
- H检索采用16点父上下文，历史观测库16/8，历史候选裁切到子块。T/V物理时间跨度相同。
- 每组从头100轮Stage1+100轮Stage2，seed502、FP32、batch128。
- Stage1全局标准化缺失MSE+0.1检索；Stage2纯预测MSE、无repair retention，预测器LR1e-4，修复器LR1e-5。
- 原始PatchTST预测器，输入P16/S8、35427参数；源码与初始化hash不变，复制末值padding，训练RNG重置。
- 固定验证MAE选主checkpoint，另存验证MSE最低checkpoint；每组锁定后立即在同一20组测试缺失位置评估两版本。
- 测试结果用于开发诊断，不能替代独立最终测试；seed502为已有开发种子，并非论文最佳种子。
- 与v44相同粒度匹配对照，v41/T1等只读取已保存结果。
- 完整训练时间包含准备/两个阶段/验证/保存，最终测试与诊断另计。记录allocated/reserved峰值GiB。

## 产物

实验目录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v45_plain_granularity_seed502_20261004`

[主结果CSV](ETTh1_P4_E23_v45_plain_granularity_seed502_20261004_primary.csv)

## 完成结论

- 去掉末端融合支路后，三种粒度的测试MSE/MAE均低于各自v44 Hybrid。
- 本轮验证最优为8/8，测试最优为16/16；4/4没有带来提升。
- 未通过相对v41-slow的固定验证双指标门槛，保留旧开发参考。
- 52项测试与3组GPU预检查通过；3条独立训练轨迹，两个阶段各100轮，新增120个测试draw评估。
