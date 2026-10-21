# 原服务器执行任务

## 冻结设计与预算

本轮只做 A0 / training seed2023，PAMAP2 sr015_bl8，seq96/pred96。先校验全部6权重以保留三seed来源，实际仅使用A0_2023自身coarse checkpoint。其他已训练权重复用，不训练A1或新seed。

- 参考：按原 `calibration_worker.py` 先64分支两轮，再256分支两轮。
- 候选：每个相同batch只执行一次冻结粗恢复，64/256各自更新，独立Adam与CPU/CUDA随机状态。
- 每支2轮、每轮85 batch、batch128，即170次更新/支/臂。
- 另从共享臂第1轮checkpoint恢复，第2轮重放85次/支。合计850次分支更新；不是完整100轮，不追加探索。
- epoch0完整验证/权重存档。选模只比较完成的epoch1/2，严格 `full96_validation_MAE < best`，分别为两支选权重；epoch0作为诊断不加入原规则选模。
- Adam lr0.001, wd0，无scheduler，FP32，原 full96 hidden-valid MAE +0.25 recent32 hidden-valid MAE，不裁剪/新增目标。
- 两臂来自相同coarse权重，`initialize_branch`保留原novel初始化，继承自身coarse水平参数，逐支seed_all(2023)。不得从训练完毕的calibration selected权重代替初始novel权重。
- 全部非校准参数、库、键、retrieval EMA冻结，原近期CNN冻结；每支完整原校准member训练，含其自己的水平网络。

## 数据与掩码

使用原 `data_for('PAMAP2','sr015_bl8',STUDY)`，原manifest SHA256 `da4825fe5bacbc0642eb246c647ffc080b8abcd9b801603bbd401ac3e830cbd6`；protocol SHA256 `cf3e694c10c5b4cd8256971bbd5a1783cd2b4abe32eca53caa6f74532273a110`。

原划分：subjects101–106训练、107验证、108/109测试；10Hz、40变量、按记录/活动边界切分，窗口stride16。必须由原manifest确认而非新建数据流程。训练数据顺序/动态mask seed2023，epoch参数0、1；验证mask2021，完整1306窗口。起点率0.15、块长8是生成参数，不等于最终缺失比例。保持原归一化和有效观测范围，无原生真值位置不作标签。不要用Weather的5175验证窗口。测试20组种子2026101801–2026101820仅是历史说明，本次禁止调用测试。

## 原服务器实现要求

在新的独立源码目录实现 `run_acceptance.py`，不可直接修改正式消融 worker。以归档 `calibration_worker.py` / `policy.py` 为参考，导入原 common/adapter/run。helper接口输入core及调用前缀的闭包，捕获的是原模型5参数校准接口；需保持 V125Adapter 的归一化、反归一化、观测值回填和原生缺失处理。不可以把归一化输出直接与原尺度truth计算损失。

1. 先验证原文件哈希/数据manifest/库hash；冻结父来源并写入本次source lock。把实际依赖及版本存档。若任何已知hash不符立即终止，不自动改锁。
2. 实现一个batch参考与共享前向/主MAE梯度对比，确认完全相同且每个branch主损失梯度非零、有限。主损失单独求导，不用总loss.backward成功代替。各支输入和输出均保留原observed值。
3. 保存两套优化器、独立Python/NumPy/Torch CPU及目标CUDA RNG。数据迭代器如消费随机流，须重放并检查输入hash，不能仅凭seed宣称一致。公共前缀不得消费随机数；检查前后状态。
4. 每轮使用原完整验证函数。分别恢复每支流进行验证，再保存流；若evaluate消耗随机数应保持与原参考一致，不假设其无随机性。分别记录选模并保留最佳分支权重，最终固定1:1融合。禁止使用平均两支MAE替代独立选模。
5. checkpoint须原子保存模型、两优化器、两随机流、库/EMA身份、epoch/batch游标、累计更新数、两个best/selected状态。断点恢复额外一轮与未中断同一终点逐值比较。
6. 共享计算只缓存当前batch。不得全量预计算后遗漏缓存生成/读取成本。两臂顺序运行单张空闲卡，GPU0映射由CUDA_VISIBLE_DEVICES控制，报告CPU线程4、TF32/cuDNN/determinism及版本。沿用原服务器参考设置；两臂完全一致。
7. 参考和共享均每轮全量验证，不启动正式100轮。训练每5分钟至少检查并记录进度/显存；进程异常保存原因，不自动无限重跑。

## 硬性验收门槛（先登记再看结果）

- 每步loss、首批主MAE梯度、最终权重tensor、Adam所有状态、各支RNG、每轮完整验证和各自selected epoch逐值相同；最大tensor差0。序列化文件SHA因格式不同不要求相同，用canonical tensor hashes/逐tensor对比。
- 恢复重放与未中断模型/Adam/RNG/选模逐值相同。
- 冻结参数/库/键/EMA全程不变；训练索引/观测mask hashes对应一致；observed数值不变。
- 更新段时间至少省15%；训练peak allocated增加≤5%；同协议推理中位耗时退步≤5%。推理不是预期收益，必须保留重复样本；若受波动影响，标记不确定，不事后放宽门槛。
- 原一般精度容忍MAE≤1%、MSE≤2%不能代替本次严格等价门槛。数值不等价即本轮失败，先分析，不扩训练。
- 明确记录逐臂总wall时间（准备、更新、验证、保存），与更新段分开。父粗训练成本引用历史；warm本轮成本单列。禁止把更新段节省称为完整Stage1节省。

## 需要返回

按templates四张CSV填写全部字段，缺失用空值和原因，不能填0假装测得。指标全96/前中后32、缺口1–8/9–16/17–32/33–96、逐变量；分层只解释，full96是主要判断。报告有效隐藏坐标计数、误差和与均值，空组明确标记。

独立计时：coarse、single64、single256、dual原推理，batch128，5warmup+20重复，CUDA同步；报告总参数/实际前缀调用次数、allocated/reserved峰值、median/p95/吞吐。训练和推理分别reset peak，标明输入是否已在GPU及GPU是否共享。新增验收计时不替代原历史资源。

返回result_manifest.json、运行命令、执行源码commit/patch、环境完整版本、source lock、每轮日志/损失/grad与冻结核验、权重与optimizer/RNG的哈希对照、失败差异（如有）、四张CSV。权重可留原服务器，但回传路径和SHA；不上传大型权重/原数据。新增结果独立GitHub目录/commit，不能覆盖 `experiment_results/v125_ablation`。

最后按四问汇报：支持了哪个判断？收益来自计算复用还是精度？限制？是否值得下一步扩大验收？本轮成功也不证明所有数据集或完整100轮的效果。
