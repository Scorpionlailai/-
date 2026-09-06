# 共享归一化与缺失时间点重建监督实施计划

> 执行要求：按任务顺序使用测试驱动开发；每项先建立失败测试，再做最小实现，最后运行相关回归测试。执行前同时阅读对应规格文档。

**目标：** 在保持 PatchTST Encoder 与原预测头不变的条件下，使用训练阶段人工遮挡前的完整回溯窗口，直接监督缺失时间点的数值恢复。

**总体结构：** missing 输入与 clean 目标共享由可观测值计算的窗口归一化统计量。最终恢复 latent 通过独立线性重建头得到重叠 patch，聚合回 96 点时间轴后，仅在人工遮挡位置计算标准化空间 MSE 与 MAE；未来预测仍经过原 PatchTST 路径。

**技术范围：** Python、PyTorch、NumPy、YAML、pytest；沿用现有 ETTh1 数据协议、动态训练掩码、固定验证测试掩码和检查点协议。

**对应规格：** `docs/superpowers/specs/2026-09-06-shared-normalization-imputation-supervision-design.md`

## 全局约束

- PatchTST Encoder 和原预测头结构不得修改。
- ours 继续从头联合训练，不使用 clean checkpoint 初始化。
- 完整回溯值只能进入训练目标，不能进入 query、key、边筛选、消息聚合或预测前向。
- clean 目标必须使用 missing 可观测值产生的窗口均值与标准差。
- 重建损失只覆盖人工遮挡且有效的时间点，不覆盖可观测点和 padding。
- 原 latent recovery loss 继续关闭。
- 第一轮不改变三类边 top-k、阈值、候选池及当前 Filter 优化设置。
- 新配置独立保存，旧配置和旧检查点行为必须保持兼容。
- 所有指标只在训练集标准化后的空间中计算。
- 不运行真实长训练；代码验证完成后由用户在本机 GPU 上执行 ETTh1。

---

## 文件职责

- `src/malr/data/normalization.py`：产生共享坐标下的 missing 输入与完整数值目标。
- `src/malr/training/data.py`：只在需要重建监督的训练批次中携带完整标准化目标。
- `src/malr/training/trainer.py`：扩展批次协议、传递损失权重、聚合并输出训练损失。
- `src/malr/models/reconstruction.py`：负责 latent 到 patch 的线性重建，以及重叠 patch 到 96 点时间轴的有效平均。
- `src/malr/models/recovery.py`：在启用时持有独立重建头，不改变现有恢复与预测输出。
- `src/malr/models/stage1.py`：仅在训练目标中调用重建分支，并保证完整目标不进入前向恢复。
- `src/malr/training/losses.py`：定义缺失时间点 MSE、MAE 及新的总损失组合。
- `src/malr/config.py`：解析并验证两项重建权重，保持零权重配置的旧哈希兼容。
- `src/malr/cli.py`：按配置启用训练目标，确保验证和测试不构造重建监督。
- `configs/stage1_sr010_replacement_filter_imputation.yaml`：保存第一轮 ETTh1 重建监督实验设置。
- `tests/unit/test_normalization_and_patching.py`：验证共享坐标与目标构造。
- `tests/unit/test_reconstruction.py`：验证线性重建形状、重叠平均和 padding 排除。
- `tests/unit/test_teacher_and_losses.py`：验证缺失位置损失和总损失权重。
- `tests/integration/test_gradient_flow.py`：验证重建梯度到达恢复侧但不经过原预测头。
- `tests/integration/test_leakage_guards.py`：验证隐藏完整值只影响训练目标。
- `tests/integration/test_training_smoke.py`：验证训练日志、检查点和旧配置兼容。

---

### 任务 1：构造共享坐标下的完整训练目标

**涉及文件：**

- 修改 `src/malr/data/normalization.py`
- 修改 `src/malr/training/data.py`
- 修改 `src/malr/training/trainer.py`
- 测试 `tests/unit/test_normalization_and_patching.py`
- 测试 `tests/integration/test_leakage_guards.py`

**输入：** 完整窗口、观测掩码、仅由训练集拟合的变量回退统计量。

**输出：** missing 标准化输入、共享均值、共享标准差、有效观测掩码，以及可选的完整标准化训练目标。

- [x] 新增失败测试：同一窗口的 missing 输入和 clean 目标必须使用完全相同的均值、标准差。
- [x] 新增失败测试：改变固定掩码下的隐藏完整值，不得改变 missing 输入、均值和标准差。
- [x] 新增失败测试：上述隐藏值变化必须改变完整标准化目标。
- [x] 新增失败测试：变量整窗无观测时，missing 与 target 都使用训练集回退统计量。
- [x] 执行新增测试，确认失败原因是尚未返回共享坐标目标，而不是测试构造错误。
- [x] 扩展归一化结果，使完整目标直接由现有观测统计量变换得到；均值和标准差保持停止梯度。
- [x] 扩展训练批次，使完整目标为可选字段；批次迁移设备时正确保留空值。
- [x] 训练数据仅在 ours 且重建权重大于零时携带目标；验证、测试和其他变体不携带目标。
- [x] 运行归一化、数据窗口和泄露防护测试，确认全部通过。
- [x] 单独提交该任务涉及的数据协议与测试，不包含无关文件。

---

### 任务 2：实现 latent 数值重建与重叠聚合

**涉及文件：**

- 新建 `src/malr/models/reconstruction.py`
- 修改 `src/malr/models/__init__.py`
- 新建 `tests/unit/test_reconstruction.py`

**输入：** 形状为批次、变量、patch、latent 宽度的恢复结果；patch 长度 16、步长 8、目标窗口长度 96。

**输出：** 形状为批次、时间、变量的标准化重建窗口，以及每个时间点的有效覆盖计数。

- [x] 新增失败测试：线性重建头将每个 latent 映射为长度 16 的 patch，批次、变量和 patch 轴保持不变。
- [x] 新增失败测试：两个 patch 对同一时间点给出不同值时，时间轴输出必须取有效平均。
- [x] 新增失败测试：末端超出 96 点的复制 padding 不进入输出或覆盖计数。
- [x] 新增失败测试：每个原始时间点覆盖计数为正，输出不存在非有限值。
- [x] 执行新增测试，确认因重建模块尚不存在而失败。
- [x] 实现单层线性重建头，使用正常可训练初始化，不与 PatchTST 原预测头共享参数。
- [x] 实现按 patch 起点回填并按有效覆盖次数求平均的 96 点聚合；超出窗口长度的预测直接排除。
- [x] 验证重建过程保持梯度，且没有额外深层网络或隐式预测路径。
- [x] 运行重建单元测试，确认全部通过。
- [x] 单独提交重建模块和对应测试。

---

### 任务 3：定义缺失时间点 MSE、MAE 与联合目标

**涉及文件：**

- 修改 `src/malr/training/losses.py`
- 修改 `tests/unit/test_teacher_and_losses.py`

**输入：** 96 点标准化重建窗口、完整标准化目标、原始时间点观测掩码、两项损失权重。

**输出：** 缺失位置重建 MSE、缺失位置重建 MAE、加权项及总损失。

- [x] 新增失败测试：只改变可观测位置的重建值时，两项缺失损失保持不变。
- [x] 新增失败测试：缺失位置的手工样例得到预先计算的 MSE 与 MAE。
- [x] 新增失败测试：没有缺失位置时返回与计算图相连的有限零值。
- [x] 新增失败测试：总损失严格满足下式：

$$
L_{total}=L_{forecast}+\lambda_{mse}L_{imp}^{MSE}+\lambda_{mae}L_{imp}^{MAE}+\lambda_{keep}L_{keep}.
$$

- [x] 执行新增测试，确认因损失项尚不存在而失败。
- [x] 实现只在未观测且有效位置进行归约的 MSE 与 MAE。
- [x] 扩展训练损失结果，分别保存原始值与加权值；旧调用在两项权重为零时保持原结果。
- [x] 运行损失单元测试，确认新旧损失路径全部通过。
- [x] 单独提交损失实现和对应测试。

---

### 任务 4：将重建监督接入 ours 训练路径

**涉及文件：**

- 修改 `src/malr/models/recovery.py`
- 修改 `src/malr/models/stage1.py`
- 修改 `tests/integration/test_gradient_flow.py`
- 修改 `tests/integration/test_leakage_guards.py`

**输入：** 最终恢复 latent、可选完整标准化目标、原始观测掩码、两项重建权重。

**输出：** 保持不变的未来预测，以及仅训练期存在的重建窗口和重建损失。

- [x] 新增失败测试：权重大于零时，ours 创建重建头并产生正确形状的训练期重建窗口。
- [x] 新增失败测试：权重为零时不创建重建头，旧 ours 参数键和前向行为保持不变。
- [x] 新增失败测试：完整目标缺失但重建权重大于零时立即报出明确错误。
- [x] 新增失败测试：固定 missing 输入时，仅改变完整目标不会改变 forecast、恢复 latent 或边选择。
- [x] 新增失败测试：同样的目标变化会改变重建损失。
- [x] 新增失败测试：只对重建损失反向传播时，重建头、替换式融合和 Filter 获得非零有限梯度，PatchTST 原预测头没有梯度。
- [x] 执行新增测试，确认失败位置与新增训练分支一致。
- [x] 让恢复模块按需持有重建头，但不在普通 forward 中执行重建，避免验证和推理额外开销。
- [x] 在训练目标中对最终恢复 latent 执行重建、时间轴聚合与缺失损失计算。
- [x] 保证预测路径仍从同一恢复 latent 进入原 PatchTST Encoder 和原预测头。
- [x] 运行梯度、泄露、恢复和端到端测试，确认全部通过。
- [x] 单独提交训练接线和防泄露测试。

---

### 任务 5：增加配置、日志与实验隔离

**涉及文件：**

- 修改 `src/malr/config.py`
- 修改 `src/malr/cli.py`
- 修改 `src/malr/training/trainer.py`
- 新建 `configs/stage1_sr010_replacement_filter_imputation.yaml`
- 修改 `tests/unit/test_config.py`
- 修改 `tests/integration/test_training_smoke.py`
- 修改 `tests/integration/test_checkpoint_protocol.py`

**输入：** 两项非负重建权重，第一轮均为 0.1。

**输出：** 可复现的新实验配置、训练日志、检查点元数据和运行清单。

- [x] 新增失败测试：负重建权重必须被配置验证拒绝。
- [x] 新增失败测试：旧配置缺省两项权重时均为零，旧配置哈希保持与现有工件一致。
- [x] 新增失败测试：新配置明确设置两项权重为 0.1，并继续关闭 latent recovery loss。
- [x] 新增失败测试：训练日志包含重建 MSE、重建 MAE、两项加权值以及已有预测与 keep 指标。
- [x] 新增失败测试：配置哈希不同的重建检查点不能被旧配置加载。
- [x] 执行新增测试，确认配置与日志字段尚未实现。
- [x] 增加两项默认值为零的配置字段及非负验证；旧配置哈希序列化时省略默认零字段。
- [x] CLI 仅为启用重建监督的 ours 训练批次请求完整目标，并把权重传给训练器。
- [x] 训练日志按批次平均记录原始与加权重建损失；验证阶段不计算这些训练辅助损失。
- [x] 新配置继承当前候选池 32、独立 Filter 学习率、keep-loss、top-k 和阈值，只新增重建监督权重。
- [x] 运行配置、训练烟雾和检查点协议测试，确认全部通过。
- [x] 单独提交配置、日志和实验文件。

---

### 任务 6：完整回归与用户训练交接

**涉及范围：** 全项目测试、最终差异、训练命令说明，不新增功能。

- [x] 运行全部 pytest 测试，确认零失败；官方 S4M 未启动的既有跳过项可以保留。
- [x] 检查文本差异，确认不存在空白错误、调试输出、占位项或无关文件修改。
- [x] 检查新模型参数分组，确认 PatchTST 原预测头仍属于原基础学习率组，重建头不与其共享参数。
- [x] 通过微型训练批次确认预测、重建、Filter 梯度和检查点保存均可运行。
- [x] 输出 PowerShell 执行顺序：生成固定掩码、构建训练集原型库、训练同配置 missing、训练启用重建监督的 ours。
- [x] 明确要求先比较验证 MSE、验证 MAE和重建损失下降趋势；未通过验证门槛时不运行测试集。
- [x] 保留当前功能分支，不执行合并。

## 验收标准

- hidden clean 值对前向预测严格无影响，对重建损失有确定影响。
- 重建目标与 missing 输入处于完全相同的窗口归一化坐标。
- MSE 与 MAE 仅覆盖人工遮挡的原始时间点，每个时间点只计一次。
- 重建损失能够训练恢复模块和 Filter，不直接训练 PatchTST 原预测头。
- 旧配置权重为零时不创建重建参数，旧检查点继续严格加载。
- 新实验验证结果至少同时优于同配置 missing；优选目标为 MSE 小于 0.7915073 且 MAE 小于 0.6034579。
