# 分组式 clean–mask 原型记忆设计

## 1. 背景与问题

当前历史原型模块能够进入前向传播并改变实验结果，但它在验证集上的收益不能稳定迁移到测试集。进一步诊断表明，旧方案存在两个主要问题：原型按照包含目标中心的完整历史片段形成，而实际检索只能使用缺失 patch 的剩余信息；扩大候选池和 Top-K 后，模型只是聚合了更多噪声原型，并未提高候选的真实性。

本轮调整不再增加旧 Matrix Profile 原型的边数，而是借鉴 S4M 的有限记忆与增量分组思想，建立固定的 clean–mask 配对检索空间。完整 clean patch 负责提供可恢复内容，masked patch 负责在同一检索空间中查找历史相似模式。

## 2. 目标与非目标

本设计的目标如下：

- 只使用训练集构建历史原型库，严格隔离验证集和测试集；
- 每个变量独立建库，不跨变量共享具体原型；
- 让 clean patch 与其 masked 版本在同一个固定检索空间中可比较；
- 先定位相似组，再从组内选择具体原型；
- 最多使用 4 个、真实余弦相似度不低于 0.7 的历史原型；
- 将历史消息与时间边、变量边联合用于缺失 latent 恢复；
- 保持 PatchTST Encoder 和预测头不变，完整模型仍从头训练；
- 通过严格对照实验度量历史模块的净贡献。

本轮不追求以下目标：

- 不让历史原型库访问验证集或测试集；
- 不使用 clean checkpoint 初始化 `ours` 的 PatchTST；
- 不恢复 Matrix Profile 作为主检索方法；
- 不让历史检索空间在 `ours` 训练期间继续漂移；
- 不增加新的预测头，也不调整 PatchTST 原论文的预测超参数；
- 暂不迁移到 Weather，先完成 ETTh1 验证。

## 3. 总体流程

完整流程分为三个相互隔离的阶段。

第一阶段训练标准 clean PatchTST。该阶段沿用现有 clean 复现流程，不加入历史模块，不改变预测损失和模型结构。

第二阶段读取 clean checkpoint 和训练集，训练 mask-aware 查询编码器并建立分组式原型库。该阶段产出的检索编码器、组中心、组成员和 clean patch 全部冻结。

第三阶段训练 `ours`。PatchTST、恢复网络、时间边和变量边从头训练；冻结的历史库只负责检索，选中的原始 clean patch 则通过当前 `ours` 的 patch projection 转换到当前恢复空间。

因此系统中存在两个职责不同的空间：固定检索空间只判断历史模式是否相似，当前恢复空间负责将选中的 clean 内容用于缺失恢复。两个空间通过原始 clean patch 连接，不要求共享参数。

## 4. 固定 clean–mask 检索空间

### 4.1 输入维度

设 patch 长度为 $P=16$。clean patch、masked patch 和二值 mask 分别为：

$$
x^{\mathrm{clean}}\in\mathbb R^{16},\qquad
\widetilde{x}\in\mathbb R^{16},\qquad
m\in\{0,1\}^{16}.
$$

缺失不会缩短 patch。缺失位置使用统一占位值，并同时提供 mask：

$$
\widetilde{x}=m\odot x^{\mathrm{clean}}+(1-m)\odot 0.
$$

clean 侧等价使用全 1 mask。因此 clean 与 masked 输入都具有固定格式，masked 程度只改变 mask 中 0 的数量，不改变张量维度。

### 4.2 共享归一化坐标

历史检索统一使用训练集按变量计算的均值和标准差。clean patch、masked patch 和候选原型均在该固定坐标中编码，缺失占位 0 对应训练均值。检索阶段不使用验证集或测试集统计量，也不使用每个查询单独估计的统计量。

原型库同时保存恢复原始 clean patch 所需的信息。候选被选中后，先转换回其 clean 数值，再使用当前输入窗口的共享 observed-only 统计量重新归一化，最后送入当前 `ours` 的 patch projection。这样检索空间固定，而恢复值与当前 PatchTST latent 坐标一致。

### 4.3 编码与配对监督

冻结的 clean patch projection 与固定全 1 mask 共同定义 clean 侧编码。接口上 clean 与 masked 两侧都接收“数值加 mask”的固定格式，clean 侧内部只需要把数值送入已经训练好的 patch projection：

$$
k_i=\operatorname{Normalize}\left(E_{\mathrm{clean}}([x_i,\mathbf 1])\right).
$$

mask-aware 查询编码器读取 masked 数值和 mask：

$$
q_i=\operatorname{Normalize}\left(E_{\mathrm{mask}}([\widetilde{x}_i,m_i])\right).
$$

同一个训练 patch 的 clean key 与 masked query 构成正配对，基础对齐损失为：

$$
\mathcal L_{\mathrm{pair}}=1-\cos(q_i,k_i).
$$

训练时还约束 masked query 在同变量 clean key 集合中的相对邻域，使其优先接近对应 clean key，而不是不相似的其他 patch。clean key 始终停止梯度，避免检索目标随 masked 编码器共同漂移。

mask-aware 编码器只使用训练集和当前 variable-missing 生成器训练。第一版固定训练 20 个 epoch，不使用早停；每个 epoch 动态生成训练 mask。训练完成后，编码器与检索空间一起冻结。

## 5. 增量分组与有限记忆

### 5.1 唯一 patch 定义

每个训练 patch 使用 `(variable_id, absolute_start)` 标识。同一绝对 patch 即使出现在多个滑动窗口中，也只参与一次入库扫描。扫描步长为 1，以保留所有可能的 patch 相位；容量控制由分组记忆完成，而不是通过稀疏扫描提前丢弃模式。

### 5.2 分组规则

每个变量独立按照绝对时间顺序扫描。设新 clean key 为 $k_i$，现有组中心为 $c_g$，首先寻找最相似的组：

$$
g^*=\arg\max_g\cos(k_i,c_g).
$$

若最高相似度不低于 0.8，则新 patch 加入该组；否则建立新组。分组只使用一条明确边界，不设置 S4M 原实现中的灰色区间。

组中心维护所有曾加入该组成员的累计均值。第 $n$ 个成员加入后，先更新未归一化累计均值：

$$
\bar c_g^{(n)}=
\frac{n-1}{n}\bar c_g^{(n-1)}+
\frac{1}{n}k_n.
$$

用于余弦检索的组中心再做单位范数归一化：

$$
c_g^{(n)}=\operatorname{Normalize}(\bar c_g^{(n)}).
$$

### 5.3 容量与淘汰

每个组最多保存 16 个具体原型，每个变量最多保存 128 个组。因此单变量最多保存 2048 个具体原型，ETTh1 七个变量最多保存 14336 个具体原型。

组内容量已满时，移除最早保存的原型，再加入新原型。变量的组数已满且出现不属于任何现有组的新模式时，移除最早建立的整个组，再建立新组。淘汰遵循 S4M 式先进先出策略，第一版不引入额外的重要性评分。

被淘汰的具体原型不再参与细粒度检索；组中心仍由该组曾接收的全部成员累计得到。整个组被淘汰后，其中心、成员和累计统计全部删除。

## 6. 缺失 patch 的两级历史检索

### 6.1 可用性门控

历史检索至少需要 4 个可见点：

$$
\sum_{t=1}^{16}m_t\ge4.
$$

若条件不满足，历史消息直接置零，由时间边和变量边负责恢复。该规则防止完全缺失或极少可见信息的 patch 仅凭 mask 形状产生虚假高相似度。

### 6.2 组级粗检索

masked patch 通过冻结编码器得到查询 $q$。查询只与相同变量的组中心比较，并选择相似度最高的一个组：

$$
g^*=\arg\max_g\cos(q,c_g).
$$

组中心只负责缩小搜索范围，不直接作为恢复内容。若该变量没有可用组，则历史消息为零。

### 6.3 同掩码细粒度检索

选定组最多包含 16 个原型，因此第二层比较组内全部成员，不再设置额外候选池。对每个 clean 原型 $x_j$ 施加当前查询完全相同的 mask：

$$
\widetilde{x}_j^{(m)}=m\odot x_j.
$$

使用同一个冻结 mask-aware 编码器得到候选表示：

$$
k_j^{(m)}=E_{\mathrm{mask}}([\widetilde{x}_j^{(m)},m]).
$$

查询与候选的真实余弦相似度为：

$$
s_j=\cos(q,k_j^{(m)}).
$$

最终只接受相似度不低于 0.7 的候选，并在其中选择最多 4 个：

$$
\mathcal N_h(q)=
\operatorname{TopK}_{j}\{s_j\mid s_j\ge0.7\},
\qquad K\le4.
$$

不足 4 个时按实际数量使用；没有候选达标时历史消息为零。历史边不再经过当前可学习 Filter，0.7 表示实际余弦相似度，而不是可被模型整体抬高的置信概率。

### 6.4 历史消息

达标原型按照相似度进行温度 softmax 加权：

$$
\alpha_j=
\frac{\exp(s_j/\tau)}
{\sum_{l\in\mathcal N_h(q)}\exp(s_l/\tau)}.
$$

选中的原始 clean patch 使用当前 `ours` 的 patch projection 编码，得到历史消息：

$$
h_{\mathrm{historical}}
=\sum_{j\in\mathcal N_h(q)}
\alpha_jW_P(x_j).
$$

softmax 温度由统一配置文件管理，第一版初始值为 1.0。

## 7. 三类信息联合恢复

时间边和变量边继续使用现有候选构造与 Filter 机制。历史边替换为本设计的固定两级检索，不使用历史 Filter。

恢复网络接收受损初始 latent、时间消息、变量消息、历史消息和观测比例：

$$
h_{\mathrm{rec}}=
F_{\mathrm{recovery}}
\left(
h_{\mathrm{damaged}},
h_{\mathrm{temporal}},
h_{\mathrm{variable}},
h_{\mathrm{historical}},
\rho
\right).
$$

受损 patch 使用恢复结果，完整 patch 保持身份路径。随后全部 patch latent 进入现有整体 refinement，再送入原版 PatchTST Encoder 和预测头。PatchTST 的预测头、预测长度和论文超参数不因本轮历史模块调整而改变。

## 8. 训练损失与梯度边界

`ours` 的 PatchTST、恢复网络、时间边和变量边从头联合训练。历史检索编码器、clean key、组中心和组成员全部冻结。

第一版总损失为：

$$
\mathcal L=
\mathcal L_{\mathrm{forecast}}
+\lambda_{\mathrm{mse}}\mathcal L_{\mathrm{imp\text{-}mse}}
+\lambda_{\mathrm{mae}}\mathcal L_{\mathrm{imp\text{-}mae}}
+\lambda_{\mathrm{keep}}\mathcal L_{\mathrm{temporal/variable}}.
$$

imputation MSE 和 MAE 只监督训练时人为遮挡的位置。时间边和变量边保留现有 Filter keep loss；历史边没有 Filter keep loss。

第一版继续关闭 latent recovery loss。这样 `ours` 不会被强行拉向 clean PatchTST 的 latent 空间，clean checkpoint 只负责建立固定检索坐标。梯度可以通过被选中的 clean patch、当前 `W_P` 和恢复网络进入当前模型，但不能进入冻结的历史检索系统。

## 9. 防止训练自检索泄露

预构建历史库包含训练集 clean patch。若训练时允许 masked patch 检索其自身 clean 原型，模型将直接获得恢复答案，导致训练和测试条件不一致。

设当前训练查询起点为 $t_q$，候选原型起点为 $t_j$。同变量下所有发生时间重叠的原型均不得成为训练候选：

$$
|t_j-t_q|<16.
$$

这同时排除了查询自身和所有与查询共享数值位置的邻近 patch。该排除规则适用于训练、验证和测试的所有查询；当验证窗口为了回溯上下文而延伸到训练区间时，同样不能检索其重叠 clean 原型。测试查询通常与训练库完全分离，此时规则自然不会排除额外候选。

原型库构建、masked 编码器训练和统计量计算严格限定在训练边界内。任何数据源哈希、训练边界、变量顺序或归一化统计不匹配都必须终止加载。

## 10. 公平实验协议

第一轮只在 ETTh1 上进行，固定以下条件：

- 回溯窗口 96；
- 预测长度 96；
- `start_rate=0.1`；
- `block_len=5`；
- seed 为 2021；
- 训练 mask 动态生成；
- 验证和测试 mask 固定；
- 指标只在训练集标准化空间计算；
- PatchTST 预测超参数保持原论文配置。

训练 mask 由 `(seed, epoch, sample_id)` 确定，不依赖 batch 调用次数或模型内部执行路径。这样普通 missing、无历史恢复模型和完整模型在相同 epoch、相同样本上看到完全一致的缺失模式。

主要对照包含：

1. clean PatchTST；
2. 普通 missing PatchTST；
3. `ours-without-history`；
4. `ours-with-history`。

`missing_pretrained` 保留为附加对照，但不作为新历史模块成功与否的唯一判断基准。所有缺失模型必须使用相同验证 mask、测试 mask、数据源和划分协议。

## 11. 成功标准

完整模型首先需要同时优于普通 missing PatchTST：

$$
\mathrm{MSE}_{\mathrm{full}}
<\mathrm{MSE}_{\mathrm{missing}},
\qquad
\mathrm{MAE}_{\mathrm{full}}
<\mathrm{MAE}_{\mathrm{missing}}.
$$

同时需要比较历史模块的净贡献：

$$
\Delta_{\mathrm{hist}}^{\mathrm{MSE}}
=\mathrm{MSE}_{\mathrm{without\ history}}
-\mathrm{MSE}_{\mathrm{with\ history}},
$$

MAE 使用同样定义。两项均为正时，说明历史模块在当前测试集上产生了实际收益。

若历史模块只改善验证集而损害测试集，或者只改善 MSE 却明显损害 MAE，则不能认定历史策略成功。低历史使用率本身不是失败，只要严格阈值下被使用的历史消息具有正向净贡献。

## 12. 诊断与可追溯性

原型库构建阶段需要记录：

- 每个变量最终组数；
- 每组保留原型数和累计接收次数；
- 新建组、加入组、组内淘汰和整组淘汰次数；
- 分组相似度分布；
- 原型绝对时间位置分布。

训练与评估阶段需要记录：

- 因可见点少于 4 而跳过历史检索的比例；
- 组级最高相似度分布；
- 通过 0.7 阈值的比例；
- 最终选中 0、1、2、3、4 个原型的比例；
- 各变量历史消息使用率；
- 被选原型在缺失位置上的离线 MSE 和 MAE；
- `ours-with-history` 相对 `ours-without-history` 的逐变量变化；
- clean checkpoint、历史库、检索编码器、数据源、配置和 mask 的哈希。

这些信息必须写入运行清单，避免再次只根据平均置信度判断历史模块是否有效。

## 13. 兼容性与错误处理

新的分组式原型库使用独立格式版本，不兼容旧 Matrix Profile 原型库。加载时必须校验：

- 数据集名称；
- 训练起止边界；
- 变量数量、名称及顺序；
- patch 长度和 patch 步长；
- 检索维度；
- 训练集归一化统计；
- clean checkpoint 哈希；
- mask 填充值和编码规则；
- 分组阈值、容量与格式版本。

任一字段不匹配时必须给出明确错误并要求重建原型库，不允许自动截断、填充或回退到旧历史算法。

没有历史组、可见点不足、组内没有候选达到 0.7，均属于正常运行状态，历史消息置零，不视为异常。

## 14. 验证范围

实现完成后至少验证以下行为：

- clean 与 masked 输入维度固定一致；
- 同一个绝对 patch 不重复入库；
- 分组阈值 0.8 的边界行为确定；
- 每组最多 16 个原型；
- 每变量最多 128 个组；
- 组内和组间先进先出顺序正确；
- 组中心按照累计成员更新；
- 少于 4 个可见点时历史消息为零；
- 候选使用与查询相同的 mask；
- 相似度低于 0.7 时严格拒绝；
- 最多选择 4 个具体原型；
- 训练时排除自身与重叠原型；
- 验证和测试只能读取训练集原型库；
- 历史分支关闭时保持现有无历史行为；
- 历史检索系统无梯度，当前 `W_P` 和恢复网络有梯度；
- 元数据或哈希不一致时拒绝加载；
- 相同 seed、epoch 和 sample id 生成相同训练 mask；
- 所有模型共享相同固定验证和测试 mask。

## 15. 配置原则

所有新增选择均由统一配置文件管理。第一版固定默认值为：组加入阈值 0.8、原型使用阈值 0.7、每组容量 16、每变量最大组数 128、最少可见点数 4、历史 Top-K 为 4、扫描步长 1、mask-aware 编码器训练 20 个 epoch、历史 softmax 温度 1.0。

这些参数后续可以单独消融，但第一轮实验不得同时改变多个参数。每次迭代只验证一个明确假设，并保留相同数据划分和固定测试 mask。
