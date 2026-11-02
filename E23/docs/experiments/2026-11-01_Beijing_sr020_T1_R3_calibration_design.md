# Beijing sr020_bl8：T1＋同款R3校准

用户授权补齐T1／E6／R3 × 无校准／有校准的解释性比较。仅修复，不运行预测Stage2。

## 原T1资产

用户提供分支 `release/r3-final-20261030`，commit `7e6b3e8f06d9081b989f90e6406a7b466ed262d6`。从Git对象提取stage1/selected.pt，完整SHA256 `42b4503250affe8ab8ff95367a8d62ce2079918b1f9b5c5bec0e147e9f846dde`，与原选模记录相符。T1原训练seed2023、100轮19000更新，验证MSE选第90轮。权重为带core.前缀的93项state_dict。

本轮直接复用原T1，不重训、不重选。刚才准备的同配置重训方案未执行，已由原权重方案替代。上传的Stage2预测头不参与本轮。原T1代码SHA256 `3f87aac66745029a88907b7ce65766d3ce5f785a6d4190f6343f9649d4ad3de8`，本机副本与该commit逐值一致。模型配置封存为T1_settings.json，enc_in=12，保持原mask embedding=False等设置。

正式校准前必须严格加载原权重，回放固定验证集，与原第90轮MSE0.31760847363975614、MAE0.2803610719923891对齐（容许跨GPU FP32差5e-6）。不以测试筛选或修改T1。

## 唯一新增训练

- 数据、训练/验证/测试窗口、原生缺失、训练mask顺序及20组测试mask均与本轮R3 E消融一致。
- seed2023，batch128，100轮，19000更新。Adam .001、weight_decay0，FP32，无早停、无调度、无裁剪。
- 校准直接调用冻结R3快照的 `r2_variants.Calibration`：single64、局部卷积、一层Transformer、后端邻点guide，49880个训练参数，与R3同初始化。
- 损失：全96点隐藏且原有真值位置MAE＋0.25×近32点同口径MAE。按固定验证全96点MAE选择；同时保存MSE次选但不改主选择。
- T1所有参数和缓冲保持冻结、eval，每轮核验完整state_dict哈希，不更新T1 BN或任何其他状态。
- T1接受原本的全局训练标准化可见值和mask。不得先改成R3局部标准化输入再送T1。
- 仅为接入R3校准，将T1输出转换为R3相同的局部坐标，并构造ExplicitRepairBatch。可见值、mask、均值、标准差都只来自真实可见位置，fallback沿用R3。候选、R3 CNN/Signature、latent等完全不进入T1臂。
- 几何sequence_validity与E0一致；原生真值有效性在监督和计分中另行排除。避免把缺标签误当padding，同时不增加伪监督。

## 检查与评价

短跑只使用验证，测试保持封存。已检查零初始化校准与T1输出最大差4.768e-7（局部坐标往返FP32误差）、真实可见值逐值保留、隐藏输入值变化不影响输出、校准初始state_dict与R3完全一致、两轮反向及冻结、导出精确回放。首次依赖导入预检失败记录保留，修复为从原设置文件抽取同一配置JSON；未发生正式训练失败。

正式训练每轮输入fingerprint须与R3相同seed校准轮次相同；满预算且锁定验证MAE checkpoint后，再评价同一20组test mask，分别报告原T1及T1＋校准的full96/recent32 MSE、MAE、mask内配对差、资源和延迟。

E6与完整R3复用seed2023既有结果（不是三seed均值）。本轮只有T1校准头新增训练。不把20mask当作独立训练seed，不把不同臂的校准收益相加称为独立因果贡献。

实验根：`/home/xyw/paper/E23/artifacts/performance/Beijing_sr020_T1_R3_calibration_20261101`。
正式GPU0单任务运行，GPU1空闲。实现、T1源码、原权重及配置已记录source_lock.json，训练可从每轮checkpoint续跑。
