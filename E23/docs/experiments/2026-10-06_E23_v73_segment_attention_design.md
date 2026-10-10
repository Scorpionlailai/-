# E23-v73 长缺失段全窗注意力修复读出

用户授权自行迭代，目标为sr010_bl16场景优于固定T1；不改预测头。此次作为已有point_context_readout接口的局部新策略推进。仅一个seed2023、两层/四头、宽64的固定配置，从头100+100轮。T1和历史版本保持缓存。

## 诊断依据与选择
v72长度≥32的修复MSE .565398→.567302，近期长段改善但整段不足；≥32占固定验证缺失坐标65.99%。已有尾部逐点修复32点、HVT patch读出及全窗CNN/跨变量signature，不能把已有全局信息或距离特征描述为首次新增。v70增强H可用性、v72长段监督未解决该场景预测差距。

候选方向有整段水平/趋势投影、全96点上下文读出、历史原型再检索。此轮选全96点注意力读出，因为已有整段监督和历史查询试验，且能直接学习长段内部变化。其他方向待本轮验证后决定，不并入同一个对照。

## 新读出
保留全部v72模块，在既有point_context_readout位置（原tail refiner后、recent mean前）新增segment_attention_v1。每变量每原始时间点输入143维：CNN point64、按原有效patch覆盖折叠的HVT latent64、全局尺度coarse值1、原可见值/mask/位置/左右距离5、变量onehot7、原始隐藏连续段长度及段内phase2。原可见值在相减/缩放前mask，段遇观测/无效点分开并在窗口处截断。

投影143→64，两个pre-norm时间self-attention+FFN block，4头、FFN128、dropout.1、学习时间位置。读取全部有效96个修复feature token；这些token始终是模型特征，修复值不改写观测mask。每层额外零null key/value处理读出层内空valid上下文；原patch管线仍拒绝整窗无有效点的输入。末端global数值残差零初始化、无强制线性/平滑/幅度截断，只写入原始连续隐藏长度≥16的坐标；global残差除原query_std转回原预测接口坐标。原观测、短段、无效mask和coverage保留；新增82113参数，修复器合计503086参数。

新增参数初始化在fork_rng中，原模型参数与预测器初始SHA保持。只读取回溯可见值、合法库及模型已有修复结果，无未来标签。全局数值残差独立于原稳健局部尺度，避免输出由小局部标准差限制。

## 实验与证据
Stage1保留v72 global hidden MSE、0.1检索、recent_mean_v2、long_gap_moments_v1 (.2/.2)；按原固定验证global修复MSE选择。Stage2完整冻结修复器，原35427参数PatchTST、初始化、MSE+.2MAE、lr1e-4和padding不改。主验证MAE/次MSE同轨迹锁定，再报告原20共享位置。基准v72按固定验证MSE下降且MAE不差才更新；目标还要求两个验证与测试指标优于T1。测试的反复曝光是开发过程，最终另锁新位置做一次确认，不能将开发测试描述为盲测。

验证覆盖零初始化函数/RNG、真实长短段门控与global单位、原观测及hidden NaN/Inf隔离、全缺/无效窗口、训练梯度、双checkpoint严格加载和identity拒绝。全套记录既有失败名称；CUDA实际mask与库预检后冻结source/plan/controller。保留每轮配置、失败、完整训练时间、显存allocated/reserved与长段修复诊断。
