# E6、R3与T1：实现差异和贡献定位

本轮只读核对，没有启动新训练。

## 可核对来源

- T1论文：https://arxiv.org/html/2602.21043v1 （§3、§4.2.4、§4.2.5）。
- 官方模型：https://github.com/Oppenheimerdinger/T1/blob/main/models/T1.py 。本地对应 `/home/xyw/paper/T1/models/T1.py`。
- 官方ETTh1脚本：https://github.com/Oppenheimerdinger/T1/blob/main/scripts/imputation/T1_ETTh1.sh 。
- 官方参数入口：https://github.com/Oppenheimerdinger/T1/blob/main/run.py 。
- 我们的Signature：`/home/xyw/r3_release_20261030/E24/versions/r3_20261030/vendor/E24/integrations/v125_ablation/model/window_signature.py`，文件头明确记录借鉴T1的变量注意力几何。
- E6：`/home/xyw/paper/E23/artifacts/performance/R3_ETTh1_context_summary_ablation_20261031/implementation/variants.py` 的ContextCore/ContextAdapter。

## 核心事实

T1的CNN提取时间特征；每个CNN特征通道对应一个注意力头，Q/K/V形状B×H×M×L，单头变量注意力矩阵为M×M，沿变量轴融合整段时间签名。它不是“每个原始变量绑定一个头”。同一层同一头的矩阵不带目标时间索引，但卷积、不同通道、层级和重建头仍能形成时间位置差异，不能据此说T1不会局部建模、不能处理块缺失或没有自适应性。论文已经研究块缺失和观测模式变化。

我们的Signature沿用这一几何，但有自己的可见CNN输入、64通道、全96点特征、可见比例先验/无来源mask/null等差异。CNN、变量注意力、mask感知或Channel–Head几何不能写成独立原创。

E6不等同T1。它保留R3自己的CNN/Signature、基础Decoder、Patch16/stride8、Confidence和单支64校准，用目标patch编码和199→385→64→1的上下文MLP替代候选路径；没有T1的完整分层Stem/T1Block/PixelShuffle重建链，也没有完全相同的损失/优化/训练阶段。E6是R3内部容量匹配控制，不是T1复现或T1的等价替代。

T1推理主要处理当前输入窗口，没有本方法的显式同变量T片段、跨变量V片段及外部只读训练历史成员H的候选检索链。T1当然会通过参数学习训练集规律，不能称其“没有历史知识”。两者最终都恢复数值，差异不在“显式输出vs隐式输出”，而在输入证据是否按来源/位置/资格显式组织。

R3的候选是带来源记录、包含目标与来源上下文的编码消息，并非未混合的原始数值直接复制。真实历史成员检索、质量和点级条件读取是设计差异；当前消融不支持把所有这些都写成不可替代的贡献。

## 当前证据能支持什么

E0/E6有效参数353248/352968，差约0.0793%，同前后端、同预算。用强上下文替代候选后，四场景最终MSE相对E0增加0.88%、4.59%、3.12%、5.47%；12个配对MSE有11个、MAE全部12个支持E0。E6辅助项作用于自己的局部context_encoder，而非R3历史键目标，所以这是完整候选机制相对本轮替代方案的条件收益，不是严格单独推理算子的因果效应。

T/V的来源删除结果支持额外收益；H有明确场景依赖，sr015_bl8去H最终MSE三个seed更好。E4/E5支持当前专门筛选相对本轮随机/统一数值余弦检索的平均收益，但不能推广到所有检索器。E8显示逐点读取的优势依场景而变，低缺失共享读取反而更好，不能将逐点读取描述为普遍必要。按用户要求，本讨论不纳入E7。

因此，当前最合适的主张是：在已有强时间/变量上下文表示之外，来源可追溯、受可见性与资格约束的片段证据提供额外恢复信息。T/V是更稳定的证据，H为条件性补充。不能仅以“三专家命名”或注意力热图声称新颖/因果贡献。

## 对标T1的优先后续

1. 补齐T1原输出与T1+同款冻结修复校准，单独度量后端价值；这指修复内部校准，不是预测Stage2。
2. 在同一个固定T1特征接口上，比较等参数上下文修正和候选证据修正；共用同一冻结T1 checkpoint、同样的适配/重建接口、校准、损失和预算。两臂差异才更直接说明候选增量能否超出T1特征。接口需要另行设计，不能宣称现有R3已完成T1插件实验。
3. 对锁定模型按缺口长度、来源可见性和关系变化分层回放，报告误差差值，检验收益来自何处。注意力响应不代替误差/干预证据。

## 具体复现核对点

T1论文§3描述显式mask输入通道；当前官方ETTh1脚本没有传 `--imputation_use_mask_embedding`，官方run.py默认False。本机旧T1_SETTINGS及Exchange已有selection.json同样为False。mask仍用于观测归一化，不能称该模型完全不感知mask。

当前配置与公开脚本默认口径一致；若论文比较指向论文完整Mask-Aware Embedding结构，建议补开启该通道的敏感性检查并标注版本。不要把论文/代码口径差异当成我们的方法贡献，也不要未经核实直接宣布旧复现错误。
