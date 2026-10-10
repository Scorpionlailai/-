# T1与E23 v41-slow：高缺失场景对照

状态：六条独立训练轨迹及120个测试draw评估全部完成。

## 预先确定的范围

用户选择各场景从头训练。ETTh1回溯96、预测96；开发种子502，三种场景各训练T1和v41-slow。

| 起点率 | 块长 | 测试窗口实际平均缺失比例（20组） |
|---:|---:|---:|
| 0.15 | 8 | 72.9252% |
| 0.20 | 8 | 83.5060% |
| 0.10 | 16 | 81.9388% |

缺失生成规则保持独立变量、随机唯一块起点、允许块重叠。起点率并非实际点缺失率。两模型共用场景内的固定验证mask（seed2021）和20组测试mask（seed2026100401至2026100420）。训练mask由训练seed、epoch和sample_id确定，同一样本/epoch在不同batch配置下使用相同mask。

## 模型与训练

- v41-slow完整保留P16/S8、64维H/V/T单轮T→V→H、最近32点point residual修复支路，共174362修复参数；Stage1全局标准化缺失MSE+0.1检索，100轮、batch128、FP32、lr1e-3。
- v41-slow Stage2原MSE+0.2MAE；第11轮起加入0.05 uniform repair retention。100轮、batch128，预测器lr1e-4、修复器lr1e-5。冻结H检索key及库状态。
- T1保持作者ETTh1修复配置，max300轮、patience30、batch16、BF16、lr1e-3、隐藏坐标MSE选模；Stage2原共享预测扩展，100轮、batch128、FP32、纯预测MSE、联合lr1e-4。
- 两者原PatchTST预测头源码与配置不变，D16、35427参数、P16/S8，初始化SHA256为26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c。
- v41保留legacy padding与原训练RNG，T1采用已有共享预测接口。损失、修复学习率、padding/RNG不同，因此这是既定方案比较，不能只把差异归因于修复器结构。
- 各自按该场景固定验证MAE选预测checkpoint，锁定后立即测试同一20组缺失位置。

## 统计和保存

完整训练时间包含数据准备、两个训练阶段、每轮验证、checkpoint保存；最终20组测试另计。显存记录allocated/reserved峰值GiB，同一GPU1顺序运行。20组标准差为缺失位置波动，本轮单训练种子不能替代多种子正式结论。

新场景在隔离目录实现；原P4数据、CSV、split、训练标准化保持，原mask与旧权重不改。场景配置和mask经元数据与SHA256校验，训练前冻结源码与mask。

实验目录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/t1_v41_higher_missing_seed502_20261004`

[结果CSV](ETTh1_T1_v41_slow_higher_missing_seed502_20261004.csv)

## 完成情况

19项测试、6组GPU预检查通过。E23三组均100轮Stage1+100轮Stage2；T1三组Stage1按作者早停分别71、89、185轮，Stage2均100轮。三组高缺失场景T1均在20/20测试位置上同时改善MSE和MAE；v41-slow训练更快，但allocated显存更高。完整配对表与资源记录已保存。
