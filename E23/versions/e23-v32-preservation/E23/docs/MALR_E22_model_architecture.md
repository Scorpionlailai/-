# 当前模型的模块、层结构与融合机制

本文依据当前 dynamic_context_v3（内部版本 E22）的活动前向与训练代码，描述各模块的层组成、层间输入输出、参数共享关系和模块融合方式。预测端以可替换预测器及其输入接口表示，不绑定具体模型。本文不包含具体数据集或实验结果。

## 1. 整体组成与数据流

模型由输入处理、历史原型恢复 H、有向变量关系恢复 V、H/V 基础融合、动态时序补偿 T、显式序列恢复、缺失感知预测接口组成。

H 和 V 并行读取受损输入，分别生成目标 patch 的关系消息。共享残差网络将两类消息分别转换为候选恢复表示，H/V/none 融合器在隐空间进行加权组合。线性解码器将融合表示转换为数值 patch。T 以该表示和解码结果为条件，从当前回溯窗口内的同变量观测中聚合局部与全局上下文，输出缺失坐标上的数值残差。

H/V 基础数值恢复与 T 残差相加后，通过重叠 patch 合并与真实观测保留，形成显式恢复序列。恢复数值与原始观测掩码共同进入输入适配器，再由预测器输出未来序列。

Stage 1 训练恢复前端；Stage 2 加载其参数与历史状态，建立冻结的 H/V 候选评分教师，使用未来预测损失微调学生恢复路径和预测端。Stage 2 每次前向都重新恢复当前输入，阶段之间不传递固定插补数据集。

## 2. 张量约定与输入层

### 2.1 符号与形状

| 符号 | 含义 | 形状或取值 |
|---|---|---|
| \(B,L,C,F\) | 批量大小、回溯长度、变量数、预测长度 | 正整数 |
| \(P,S,N,D\) | patch 长度、步长、patch 数、恢复隐维度 | 正整数 |
| \(D_k\) | H 检索空间维度 | 当前公共配置取 \(D_k=D\) |
| \(X^s,M\) | 训练统计量标准化后的历史序列、原始观测掩码 | \([B,L,C]\) |
| \(U\) | 按当前窗口可见点归一化后的受损输入 | \([B,L,C]\) |
| \(p,m,v\) | patch 数值、有效观测掩码、坐标有效性 | \([B,C,N,P]\) |
| \(\rho\) | 有效坐标中的可见比例 | \([B,C,N]\) |
| \(z_0,z_H,z_V,z_{HV}\) | 初始、H 恢复、V 恢复、融合表示 | \([B,C,N,D]\) |
| \(p_{HV},\Delta_T\) | 基础恢复 patch、T 数值残差 | \([B,C,N,P]\) |
| \(\widehat U,\widehat X^s\) | 窗口归一化空间、训练标准化空间的恢复序列 | \([B,L,C]\) |
| \(\widehat Y^s\) | 训练标准化空间的预测 | 当前内部排列为 \([B,C,F]\) |

\(M=1\) 表示真实观测，\(M=0\) 表示缺失。\(v=0\) 表示 padding 等无效坐标，与真实时间位置上的缺失不同。

下文 Linear(a,b) 表示输入 a 维、输出 b 维的全连接层，除特别说明外均带偏置。拼接和维度变换作用于最后一维。GELU、Softplus、Tanh 和 softmax 是无独立可训练参数的非线性运算。

### 2.2 归一化与 patch 切分

第一层标准化使用训练集的每变量均值与标准差。第二层针对当前窗口、当前变量，仅使用有效观测估计 \(\mu_{bc},\sigma_{bc}\)：

\[
U_{btc}=\begin{cases}
(X^s_{btc}-\mu_{bc})/\sigma_{bc},&M_{btc}=1,\\
0,&M_{btc}=0.
\end{cases}
\]

整段变量窗口无观测时使用训练统计量回退。窗口统计量停止梯度，恢复和预测共享这组统计量；后续不根据生成值重新估计归一化参数。当前 observed-only 入口没有可学习 affine 变换。

切片长度为 \(P\)，步长为 \(S\)，末端复制 \(S\) 个数值位置：

\[
N=1+\left\lfloor\frac{L+S-P}{S}\right\rfloor.
\]

复制位置的有效性为 0，不计入观测比例、重建监督和序列合并。可靠度为 \(\rho=\sum m/\sum v\)，patch 状态分为完整观测、部分缺失和完全缺失。

### 2.3 修复输入投影

修复入口是一层 Linear(P,D)：

\[
z_0=W_rp+b_r.
\]

H/V 前没有额外的时序编码骨干。目标 patch、供 V 使用的源 patch，以及转换到当前窗口归一化空间后的 H 原型数值，共用这套修复投影。H 检索编码器和预测输入投影使用不同参数。

## 3. H：历史原型检索与消息生成

### 3.1 层组成

记关系网络隐藏宽度 \(h_H=\max(D,11)\)，检索隐藏宽度 \(h_k\) 当前配置为 \(2D_k\)。

| 子结构 | 输入 | 层序列 | 输出 |
|---|---|---|---|
| 缺失感知检索编码器 | 遮挡后数值与 mask 拼接，\(2P\) 维 | Linear(2P,h_k) → GELU → Linear(h_k,D_k) → L2Normalize | 检索 query/key |
| 原型数值投影 | 当前窗口归一化空间的原型，\(P\) 维 | 与修复入口共享的 Linear(P,D) | 候选 latent |
| 候选消息头 | 候选 latent、11 维关系特征 | Linear(D+11,h_H) → GELU → Linear(h_H,D) | 候选消息 |
| 候选收益头 | 目标 latent、候选消息、11 维关系特征 | Linear(2D+11,h_H) → GELU → Linear(h_H,1) | 标量收益 \(g_j^H\) |
| 候选惩罚头 | 与收益头相同的输入 | 独立的 Linear(2D+11,h_H) → GELU → Linear(h_H,1) → Softplus，再加 \(10^{-6}\) | 非负惩罚 \(u_j^H\) |

消息头先输出消息，收益头与惩罚头再读取该消息及目标表示。三个头互不共享参数。L2 归一化只用于检索 query/key 空间，不是修复 latent 的归一化。

### 3.2 历史库与检索路径

历史库按变量保存有限分组原型。成员包含变量标识、训练时间起点、训练标准化空间的数值 patch 和检索 key。只有训练边界内、当前掩码下完整可见的 patch 才可写入。原型根据 key 相似度加入已有组或新建组，成员和组数受容量约束，按先入先出淘汰。

检索编码器有三个同结构实例：可训练编码器、EMA 编码器、epoch 级冻结查询快照。Stage 1 用可训练编码器计算检索损失；实际检索使用固定快照。EMA 跟随训练编码器更新，epoch 边界同步查询快照、更新记忆 key 并发布原型。Stage 2 和推理不更新这些状态。

受损目标 patch 的检索过程为：

1. 将可见值转换到训练标准化空间，与 mask 拼接，编码为 query。
2. 在同变量历史库中选取具有可用历史成员的最相似组，再按成员 key 相似度排序取有限候选。
3. 过滤低相似度成员；成员结束时间必须不晚于目标 patch 起点。
4. 将原型数值按目标窗口的 \(\mu,\sigma\) 重新归一化，经 \(W_r\) 投影。
5. 候选 latent 与关系特征进入消息头，再由目标 latent、候选消息与特征共同进入评分头。

当前 H 特征宽度为 11。实际赋值槽位包括检索余弦相似度、两个固定为 1 的槽位、目标可靠度、目标 mask 在全部 patch 坐标上的均值、归一化变量索引，其余槽位为 0。这些是构造特征，不是独立可学习 embedding。

### 3.3 候选聚合

合法且预测收益 \(g_j^H>0\) 的候选按 \(g_j^H-u_j^H\) 排序保留 Top-K，然后在选中集合内计算：

\[
\alpha_j^H=\operatorname{MaskedSoftmax}_j(g_j^H-u_j^H),
\qquad m_H=\sum_j\alpha_j^Hm_j^H.
\]

分支收益与惩罚为候选收益和惩罚的同权重加权和。输出包含聚合消息、收益、可用性及候选计划。没有候选通过时消息为零，H 不可用。惩罚头在代码中称为 uncertainty，但没有独立的概率校准目标，不是统计置信区间。

## 4. V：有向跨变量关系消息

### 4.1 候选池与输入特征

对目标 \((c,i)\)，V 枚举当前窗口其他变量的所有 patch \((c',j)\)，其中 \(c'\ne c\)。每个目标有 \((C-1)N\) 个候选槽位，覆盖同步与正负时滞。源必须具有可见信息，源 latent 来自原始受损输入的 \(z_0\)，不使用当前前向生成的恢复值。

11 维候选特征依次为：源可见比例、目标可见比例、mask 重叠、归一化源变量索引、归一化目标变量索引、归一化有符号时滞、方向、log1p patch 距离、同步标记、归一化距离带、共同可见坐标上的形状相关性。相关性只是网络输入特征，不直接决定筛选结果。

### 4.2 层组成

设绝对距离带数量为 \(Q\)，有符号时滞桶数为 \(J=2Q+1\)，隐藏宽度 \(h_V=\max(D,11)\)。当前距离边界为 \((2,4,8,16)\)，对应 \(Q=5,J=11\)。

| 子结构 | 输入 | 层序列 | 输出 |
|---|---|---|---|
| 源变量身份嵌入 | 源变量索引 | Embedding(C,D) | \(e_{src}\) |
| 目标变量身份嵌入 | 目标变量索引 | 独立的 Embedding(C,D) | \(e_{tgt}\) |
| 时滞嵌入 | 有符号时滞桶索引 | Embedding(J,D) | \(e_{lag}\) |
| 有向消息转换器 | 源 latent、三个 embedding、11 维候选特征 | Linear(4D+11,h_V) → GELU → Linear(h_V,D) | 候选消息 \(m_j^V\) |
| 收益头 | 目标 latent、候选消息、11 维特征、3 维统计量 | Linear(2D+14,h_V) → GELU → Linear(h_V,1) | \(g_j^V\) |
| 惩罚头 | 与收益头相同的输入 | 独立的 Linear(2D+14,h_V) → GELU → Linear(h_V,1) → Softplus，再加浮点 epsilon | \(u_j^V\) |

\[
m_j^V=\operatorname{MLP}_V([z_{c',j},e_{src}(c'),e_{tgt}(c),e_{lag}(j-i),f_j^V]).
\]

源与目标 embedding 不共享，使反向变量对可以学习不同映射。这里的方向是信息传递方向，不表示已经识别出因果关系。

### 4.3 统计先验与聚合

统计表按“源变量、目标变量、有符号时滞桶”索引，保存训练候选实际收益的 EMA 均值、二阶矩和计数。查询得到的均值、标准差及样本是否足够的标记构成 3 维特征，只进入评分头，不进入消息转换器。统计表是非梯度更新的缓冲状态。

V 先做正收益准入，再按 \(g_j^V-u_j^V\) 为每个源变量保留最高分时滞候选，最后跨源变量 Top-K，经 masked softmax 加权形成 \(m_V\)。没有合格候选时消息为零、分支不可用。

当前代码先为完整候选池构造特征和消息，再筛选，并不是在消息计算前完成稀疏化。

## 5. H/V 消息到基础恢复

### 5.1 共享残差恢复层

H/V 消息分别进入同一个残差 MLP：Linear(2D,D) → GELU → Linear(D,D)。

对受损且相应分支可用的目标：

\[
z_{none}=z_0,\qquad
z_H=z_0+\operatorname{MLP}_{rec}([z_0,m_H]),\qquad
z_V=z_0+\operatorname{MLP}_{rec}([z_0,m_V]).
\]

完整观测目标或不可用分支的关系残差置零。H、V 和候选收益监督中的单候选反事实恢复共享这套 MLP。当前有效恢复是一次消息聚合和一次残差更新，没有多层迭代图传播。

### 5.2 H/V/none 软融合层

融合 MLP 为 Linear(3D+1,D) → GELU → Linear(D,3)。输入是三个候选表示与一维有效缺失坐标比例：

\[
d=P^{-1}\sum_{k=1}^{P}\mathbf 1[\neg m_k\land v_k].
\]

分母为 \(P\)，不是有效坐标数；含 padding 时 \(d\) 不一定等于 \(1-\rho\)。不可用分支表示在 MLP 输入中置零，然后计算：

\[
\ell=\operatorname{MLP}_{fuse}([z_{none},z_H,z_V,d]),\quad
\pi=[0,g_H,g_V],
\]
\[
\omega=\operatorname{MaskedSoftmax}(\ell+0.25\operatorname{stopgrad}(\pi)),\quad
z_{HV}=\omega_{none}z_{none}+\omega_Hz_H+\omega_Vz_V.
\]

最终 softmax 屏蔽不可用分支，none 始终可用，三个权重之和为 1。H 与 V 可以同时贡献，不是硬选择一个分支。Stage 1 的先验来自当前 H/V 算子；Stage 2 来自冻结教师。先验停止梯度，融合 MLP 仍由主任务训练。

none 表示不添加 H/V 关系残差，仍经过解码和 T 补偿，不等于不插补。该融合层从 Stage 1 即存在，Stage 2 沿用预训练参数。

### 5.3 数值解码层

一层带偏置、无额外输出激活的 Linear(D,P) 将融合 latent 解码：

\[
p_{HV}=W_{dec}z_{HV}+b_{dec}.
\]

输出位于窗口归一化空间。真实观测保留由后续显式恢复模块执行。

## 6. T：动态局部与全窗口时序补偿

### 6.1 五组 MLP

隐藏宽度 \(h_T=\max(32,2D)\)，时序关系特征宽度为 12。

| 子结构 | 拼接输入 | 层序列 | 输出 |
|---|---|---|---|
| Query 编码器 | \(z_{HV}\)、\(p_{HV}\odot m_{eff}\)、有效观测指示均值 | Linear(D+P+1,h_T) → GELU → Linear(h_T,D) | 目标 query |
| Source 编码器 | 原始源数值乘有效观测 mask，再拼接该 mask | Linear(2P,h_T) → GELU → Linear(h_T,D) | 候选源表示 |
| 候选评分头 | query、源表示、12 维关系特征 | Linear(2D+12,h_T) → GELU → Linear(h_T,1) | 候选分数 |
| 上下文门控头 | query、local context、global context、两个可用性标记 | Linear(3D+2,h_T) → GELU → Linear(h_T,3) | none/local/global logits |
| 共享残差解码头 | query、单路 context、\(p_{HV}\odot m_{eff}\) | Linear(2D+P,h_T) → GELU → Linear(h_T,P) | 单路数值残差，随后经 Tanh |

\(m_{eff}=m\land v\)。五组 MLP 相互独立；local/global 两路共用残差头。T source 编码器与 H 检索编码器不共享权重，输入数值空间与训练用途不同，T source 输出不执行 H 检索使用的 L2 归一化。

### 6.2 H/V 与 T 的交互

目标 query 为：

\[
q_T=\operatorname{MLP}_q([z_{HV},p_{HV}\odot m_{eff},\operatorname{mean}_P(m_{eff})]).
\]

因此 H/V 通过隐表示和解码数值共同影响 T。这里的 \(p_{HV}\odot m_{eff}\) 是解码器在观测坐标上的输出，不是已替换回真实观测的数值。

源来自当前窗口内同变量、非自身的原始受损 patch：

\[
h_j=\operatorname{MLP}_{src}([p_j\odot m_{eff,j},m_{eff,j}]).
\]

源不读取 H/V 或 T 生成的恢复值。目标条件来自基础恢复，候选内容来自原始观测，两条路径在评分头与残差头相遇。

### 6.3 评分与上下文聚合

12 维时序特征依次为：有符号时间差、方向、归一化绝对时间差、log1p patch 距离、源可见比例、目标可见比例、mask 重叠、源相对位置、目标相对位置、局部标记、归一化距离带、共同可见坐标上的形状相关性。

评分头输出再加源可靠度偏置：

\[
s_{ij}=\operatorname{MLP}_{score}([q_{T,i},h_j,f^T_{ij}])
+\log(\max(\rho_j,\varepsilon)).
\]

候选必须含有效观测，目标必须受损；部分可见源也可参与。local 集合在最短距离带内分别取左、右最近的一个可用源，最多两个锚点；当前最短距离带是两个 patch 间隔以内。global 集合为剩余全部可用源，包括远处源和未被 local 选中的近处源，不使用固定 Top-K。

两类集合分别 masked softmax，再形成：

\[
c_L=\sum_{j\in\mathcal S_L}\alpha^L_{ij}h_j,\qquad
c_G=\sum_{j\in\mathcal S_G}\alpha^G_{ij}h_j.
\]

空集合返回零上下文与不可用标记。右侧候选仍属于给定历史窗口，不涉及未来预测区间。

### 6.4 门控、解码与有界残差

门控读取 query、两个上下文及可用性：

\[
[a_{none},a_L,a_G]=\operatorname{MaskedSoftmax}
(\operatorname{MLP}_{gate}([q_T,c_L,c_G,A_L,A_G])).
\]

两个上下文分别经过同一个残差头：

\[
r_L=\operatorname{MLP}_{res}([q_T,c_L,p_{HV}\odot m_{eff}]),\quad
r_G=\operatorname{MLP}_{res}([q_T,c_G,p_{HV}\odot m_{eff}]).
\]
\[
\Delta_T=\mathbf 1[\neg m\land v]\odot
R\big(a_L\tanh(r_L)+a_G\tanh(r_G)\big),\qquad R=2.
\]

none 对应零残差；无可用源时回退 none。真实观测和 padding 上残差恒为零。local/global 是先分别解码、限幅，再门控加权，不是先混合上下文后只解码一次。

评分头、门控头、残差头的末层均零初始化，query/source 编码器不是零初始化。全新模型的 T 残差为零；Stage 2 加载 Stage 1 学到的 T 参数，不再次置零。

## 7. 显式序列恢复

该模块没有可训练层，由条件选择、求和、覆盖计数、除法、尺度变换和切片组成。

先在 patch 层面组合：

\[
p_{cand}=\operatorname{where}(m,p,p_{HV}+\Delta_T).
\]

再将 patch 映射回原始时间轴，对有效覆盖值求和并除以覆盖次数，形成每个时间点唯一的恢复值。padding 不参与计数。原始观测也单独合并，随后执行观测保留：

\[
\widehat U=M\odot U+(1-M)\odot\operatorname{OverlapAverage}(p_{cand},v),
\qquad \widehat X^s=\widehat U\odot\sigma+\mu.
\]

正常输入由统一序列 mask 切分，重叠处来源标记一致。\(\widehat U\) 进入当前预测路径，\(\widehat X^s\) 用于训练标准化空间的恢复监督或输出。真实观测不由恢复分支改写；缺失坐标的梯度可穿过重叠平均，回到解码器、T 和 H/V。

重新切 patch 时保留已恢复的非零数值，同时保留原始 observation mask。恢复后的缺失位置不会改成真实观测标记。

## 8. 缺失感知输入适配与预测接口

恢复端交付显式序列 \(\widehat U\)、原始观测 mask \(M\) 与有效性标记，保留原始时间轴和变量轴，不要求预测器消费 H/V/T 专用 latent。

当前适配器有两条投影路径：

| 路径 | 层结构 | 功能 |
|---|---|---|
| 恢复数值 | 重切 patch 后 Linear(P,D)，带偏置 | 编码完整恢复数值 |
| 缺失来源 | 缺失指示经过 Linear(P,D)，无偏置、零初始化 | 编码哪些坐标来自恢复 |
| 两路连接 | 逐元素相加 | 得到预测器输入表示 |

\[
r_{miss}=\mathbf 1[\neg m_{rep}\land v_{rep}],\qquad
z_F=W_f\widehat p+b_f+W_mr_{miss}.
\]

\(W_f\) 与修复投影 \(W_r\) 是独立参数，Stage 2 初始化时复制初值，之后分别更新。mask 路径不把恢复值重新清零，也不是置信度评分。

令 \(\mathcal A_\psi\) 为输入适配器，\(f_\theta\) 为可替换预测器：

\[
\widehat Y^{norm}=f_\theta(\mathcal A_\psi(\widehat U,M)),\qquad
\widehat Y^s=\widehat Y^{norm}\odot\sigma+\mu.
\]

预测器的内部层结构由接入模型决定，不属于恢复机制的固定组成。更换时需匹配输入布局、mask 注入、输出形状和归一化约定；当前重切 patch 与双投影相加是一个已实现的适配方式，不是所有预测器的必要结构。联合微调要求预测路径允许梯度回传。

## 9. 冻结教师与学生的交互

Stage 2 的教师复制 Stage 1 修复投影、H 消息头及评分头、V 的三个 embedding、转换器及评分头，参数冻结且始终处于评估模式。

教师对每个当前输入重新生成候选消息、收益与惩罚，输出候选选择、聚合系数、分支可用性和收益先验。冻结的是参数，不是对所有输入都相同的一张关系图；教师也不提供未来预测蒸馏目标。

学生用自己的修复投影和消息网络生成候选消息，按教师计划聚合：

\[
m_H^{student}=\sum_j\operatorname{stopgrad}(\alpha_j^{H,teacher})m_j^{H,student},
\quad
m_V^{student}=\sum_j\operatorname{stopgrad}(\alpha_j^{V,teacher})m_j^{V,student}.
\]

学生消息再进入共享恢复 MLP 和可训练 H/V 融合层；教师收益作为停止梯度先验。学生评分头被这条前向绕过，没有有效预测梯度。T 独立计算 query、上下文、门控与残差，不使用教师的旧时序算子。

历史库、检索快照和 V 统计在 Stage 2 保持只读。学生消息、基础融合、解码器、T 和预测器仍可由预测损失调整。

## 10. 两阶段训练及损失连接

### 10.1 Stage 1：自监督恢复

完整历史值只作为监督，不作为受损恢复分支的输入。最终重建在训练标准化空间、唯一序列的有效缺失集合 \(\Omega\) 上计算：

\[
\mathcal L_{repair}=\frac{1}{|\Omega|}\sum_{\Omega}
\operatorname{SmoothL1}(\widehat X^s-X^s).
\]

H/V 单候选消息经过共享残差恢复层和数值解码器，以目标缺失坐标的 Huber 误差生成收益标签：

\[
g_j^*=\operatorname{clip}\left(\frac{E_{none}-E_j}{E_{none}+\varepsilon},-1,1\right).
\]

标签停止梯度，收益头使用 SmoothL1 回归，先在各分支内部平均，再在 H/V 分支间平均。T 不使用这类收益头。

检索损失使受损 query 接近同一 patch 的完整 key，完整 key 停止梯度：

\[
\mathcal L_{memory}=\mathcal L_{pair}
+\frac{0.5\mathcal L_{contrastive}}{\max(\log(\max(n,2)),1)}.
\]

其中 pair 是余弦配对损失，contrastive 是批内对比交叉熵，\(n\) 是参与配对的受损 patch 数。当前公共配置总目标为：

\[
\mathcal L_1=\mathcal L_{repair}+0.20\mathcal L_{gain,HV}+0.05\mathcal L_{memory}.
\]

没有独立路由分类损失、惩罚头校准损失或未来预测损失。当前前 20 个 epoch 为 bootstrap，H 在基础融合中被屏蔽，历史库累积和检索辅助学习仍在进行；之后 H 按可用性进入融合，T 从开始即存在。

### 10.2 Stage 2：预测监督微调

阶段交接加载最佳恢复参数、历史库、检索快照和 V 统计，建立教师、独立预测器与缺失来源投影。预测路径未在 Stage 1 接受未来预测训练。

\[
\mathcal L_2=\operatorname{MSE}(\widehat Y^s,Y^s).
\]

Stage 2 每个 batch 重新恢复输入，预测梯度穿过适配器、显式恢复、T 与 H/V 学生路径。没有额外修复、路由、检索或教师蒸馏损失。当前恢复路径学习率为预测路径的 0.25 倍。

| 组件 | Stage 1 | Stage 2 |
|---|---|---|
| 修复投影、H 消息头、V embedding 与转换器 | 学习恢复 | 预测损失微调 |
| 共享恢复 MLP、H/V 融合、数值解码、T | 学习恢复 | 预测损失微调 |
| H/V 学生评分头 | 收益监督及消息聚合路径 | 活动前向由教师评分替代 |
| H 检索编码器、EMA、查询快照、原型库 | 表征学习与状态累积 | 固定、只读 |
| V 收益统计表 | 根据训练收益更新 | 固定、只读 |
| 冻结教师 | 尚未建立 | 按当前输入生成计划，不接受梯度 |
| 预测器与输入适配器 | 未参与预测任务 | 由预测损失训练 |

## 11. 参数共享与融合位置汇总

| 位置 | 组合方式 | 参数关系 |
|---|---|---|
| H 候选内部 | 评分筛选后对消息加权求和 | 候选共用 H 消息头，消息头与评分头独立 |
| V 候选内部 | 每源变量选一个时滞，再 Top-K 与加权求和 | 变量对共用转换器，身份 embedding 区分方向 |
| 消息到分支表示 | 目标与消息拼接，经 MLP 生成残差再相加 | H/V 和反事实恢复共享恢复 MLP |
| H/V/none 之间 | 候选 latent 按 softmax 权重求和 | 独立融合 MLP，读取停止梯度收益先验 |
| H/V 与 T 之间 | H/V 条件化 T，T 数值残差加到 H/V 解码结果 | 不共享检索、消息或评分网络 |
| T 的 local/global 之间 | 分别聚合、解码限幅，再门控加权 | 共用源编码器、候选评分头和残差头 |
| 重叠 patch 之间 | 有效覆盖数值平均，保留真实观测 | 无可训练层 |
| 恢复数值与缺失标记之间 | 双投影逐元素相加 | 两个不同线性层 |
| 教师与学生之间 | 教师计划控制学生消息聚合 | 参数初始复制，之后不共享可训练权重 |

兼容代码仍保留旧 temporal operator、多路由中间张量和 hard route 诊断字段。当前 E22 活动前向不调用旧 temporal operator，最终恢复使用 H/V/none 软融合与独立动态 T；hard route 诊断不替代软融合。

## 12. 论文来源与实现边界

保留项目已有设计记录中的借鉴来源：

| 来源 | 对应思想 | 当前实现边界 |
|---|---|---|
| RevIN，ICLR 2022 | 输入归一化与输出逆变换配对 | 当前仅依赖可见点统计，使用共享窗口统计量，不是直接调用未经修改的标准 RevIN 层 |
| S4M，ICLR 2025 | 有限分组历史记忆、原型帮助处理缺失、mask 参与预测 | 当前 H/V/T 组织、显式恢复与两阶段训练不等同于该方法，预测端不绑定其架构 |

1. Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, Jaegul Choo. **Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.** ICLR 2022. [作者项目页](https://seharanul17.github.io/RevIN/)，[论文页面](https://openreview.net/forum?id=cGDAkQo1C0p)。
2. Jing Peng, Meiqi Yang, Qiong Zhang, Xiaoxiao Li. **S4M: S4 for Multivariate Time Series Forecasting with Missing Values.** ICLR 2025. [论文 PDF](https://openreview.net/pdf?id=BkftcwIVmR)，[官方代码](https://github.com/WINTERWEEL/S4M)。

V 的有向转换器、训练收益统计、H/V 软融合、冻结计划回放、T 的 local/global/none 残差组织以项目代码为定义。现有记录不足以将这些细节归为某一外部方法的直接复现，也不构成文献独创性的证明。

## 13. 代码索引

以下路径相对当前 checkout 根目录。

| 内容 | 文件 |
|---|---|
| 主前向、两阶段目标与活动参数 | src/malr/models/dynamic_context_model.py |
| H 消息与评分层 | src/malr/models/historical_relation.py |
| H 候选检索与原型重投影 | src/malr/models/online_history.py |
| 历史库与检索编码器 | src/malr/prototypes/online_bank.py、src/malr/prototypes/retrieval.py |
| V embedding、转换器、评分和统计 | src/malr/models/variable_relation.py |
| T/V 候选池与特征 | src/malr/models/candidates_v2.py |
| 共享残差恢复层 | src/malr/models/branch_recovery.py |
| H/V/none 软融合 | src/malr/models/forecast_fusion.py |
| T 的五组 MLP 与门控 | src/malr/models/dynamic_temporal_context.py |
| 显式序列恢复 | src/malr/models/explicit_repair.py |
| 冻结教师与计划 | src/malr/models/route_teacher.py、src/malr/models/relation_outputs.py |
| 收益标签与恢复基础类 | src/malr/models/edge_gain.py、src/malr/models/online_stage1_v2.py |
| 归一化与切片 | src/malr/data/normalization.py、src/malr/data/patching.py |
| 阶段交接与优化器 | src/malr/cli.py、src/malr/training/online_trainer.py |
| 当前公共配置 | configs/datasets/_e22_common.yaml |
