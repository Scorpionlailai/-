# E23 v32：Stage1预测信息保持

用户已批准本会话提出的Stage1信息保持试验。目标是在三专家聚合的修复序列上保留预测信息，最终Stage2与现有T1均使用纯预测MSE。此为一次固定设置的可行性对照，不预先声称提升或新颖性。

## 固定矩阵与训练

- A：原v29-r1 Stage1全局修复MSE+0.1检索损失；Stage2纯预测MSE。
- B：相同Stage1基础目标，第21–100轮加0.1预测一致性MSE；Stage2与A相同。
- 两组各自从头Stage1/2各100轮，seed2021、batch128、FP32、Adam、lr0.001/0.0001、weight_decay0；修复宽度64、单轮iterative_tv_h_v1、原PatchTST宽度16。Stage1 specialist_exposure0.2、H/V/T设置沿v31 config字节相同副本。
- Stage1按原固定验证全局缺失修复MSE选权重；Stage2按固定验证MAE、最早平局选权重。报告选定轮次是否实际进入一致性训练，不用测试结果选择系数。
- B辅助预测器为原MaskAwareForecastModel/PatchTST结构，独立seed2053，在Stage1前20轮用完整训练历史预测训练未来，Adam lr0.001、wd0。修复器同时执行原训练；辅助构建与训练隔离torch RNG，不改变修复器随机序列。第20轮末固定辅助参数，eval且清空梯度，后续不再更新，也不按验证或测试选教师。
- 对完整历史和修复历史，统一使用原可见历史的query mean/std、mask与patch validity。辅助预测器输出处于训练集全局标准化尺度。完整目标分支no_grad；修复分支允许输入梯度，教师参数无梯度。loss=mean((g(repaired)-stopgrad(g(clean)))²)，保留基础修复和检索损失；不把未来标签作为修复器输入。
- 辅助模型只作Stage1训练监督，权重及身份独立保存供审计，不进入修复器state_dict。Stage2重新初始化正式预测器、重置RNG和新optimizer，两组初始预测器SHA相同，不继承辅助参数。最终推理参数不增加。
- Teacher预热20轮及一致性计算/保存/诊断均计入Stage1完整成本，分别记录新增累计计算时间、覆盖窗口、每轮基础/一致性损失、冻结SHA。辅助仅使用train批次；验证只诊断，不更新。

## 测试、诊断和判断

两组顺序完成各自两阶段后，立即验证重放、修复诊断并测试原固定三组缺失位置；seed2021/2026100201/2026100202、每组2688窗口/1806336预测坐标。Stage2纯MSE与T1一致，其Stage1原论文配置仍各自保留。

报告均值/总体std、完整训练时间及阶段时间、峰值allocated/reserved GiB、设备、阶段和辅助参数数、教师额外耗时与覆盖、Stage1与Stage2修复误差。补充冻结辅助在完整/修复验证历史上的预测MSE/MAE和信息差异，用于检查教师质量和信息约束是否奏效；不用于教师训练或新参数搜索。Stage1/Stage2各测一次并绑定checkpoint和mask SHA。

主要判断B相对A固定验证MSE严格改善且MAE不退步；全局候选相对现有v31-B验证0.7456886016863004/0.5817581012405776使用同样门槛。不能因测试均值更好改变此规则。T1、v31-B使用既有结果，不重跑。三组位置仅代表mask稳定性，不代表训练seed显著性。最终Stage2仍可能改变修复能力，必须报告最终诊断。

## 实现和保存

独立prediction_preservation.py实现辅助训练与损失；train_repair新增默认none的stage1_preservation参数，CLI接入clean_forecast_v1。阶段manifest/checkpoint/加载器校验模式、系数、预热、教师身份，旧调用兼容。专用runner/report及诊断脚本管理A/B全流程，禁止覆盖既有输出。

正式训练前冻结源/config/tests/script；现有dirty基础精确归档，修改保存相对基线patch；用户CSV与旧实验输出保留。起初沿已授权paper/e23-migration工作分支；发现另一会话并发修改后转入paper/e23-v32-preservation隔离工作副本，保留双方改动。按用户要求正式训练每5分钟检测。
