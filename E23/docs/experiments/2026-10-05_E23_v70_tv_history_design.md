# E23-v70：TV上下文驱动H的全缺失patch检索

用户已确认局部改动：先由T、V提供信息，再交给H检索。本轮仅检验H查询改动，预测器结构、配置及Stage2配方保留v69。

## 数据流

原可见窗口→原CNN/signature位置前端→T预览更新→V预览更新→TVHistoricalQuery→原历史库top4检索→原H读取/路由/候选读出→原显式恢复接口→原PatchTST。

有可见值的patch保留原检索，只有有效坐标全部缺失的patch启用TV查询。完全没有真实可见值的整个窗口拒绝此路径。原mask不改变，TV生成的潜变量不成为真实观测或历史库成员。

查询映射：LayerNorm(64)，拼接query_mean、log(query_std)、本变量可见比例，Linear(67,64,bias=False)后单位化。新增4416参数，修复器425389参数；额外初始化隔离RNG，原35427参数预测器及原修复权重初始值保持一致。

历史库、因果候选资格、四个名额、0.7相似度阈值保持；不足阈值时H可拒绝读取。仅更新完全缺失patch的缓存，其他patch候选不变。候选键是训练集全局标准化尺度，查询监督对齐同一冻结历史编码器。硬候选检索使用detached query；辅助损失为查询提供梯度。

## 训练与选模

Stage1保留global hidden MSE、0.1原检索项、原recent_mean_v2监督；额外加入0.1*cosine查询对齐损失。teacher是本epoch冻结编码器对训练回溯真值patch的编码，只参与损失，不进入推理输入、不使用未来预测目标、不增加库成员。query、T、V接收该梯度，teacher不更新。

single-round TV→H；specialist exposure仍0.2，H全缺失潜在资格在TV前确定，实际候选在TV后检查。专家禁用/单专家预览遮罩保持，候选不强制获得路由权重。

固定ETTh1 96→96、P16/S8、repairD64、seed2023、batch128、FP32；各场景100+100轮，lr0.001/0.0001、weight_decay0。Stage1全局验证修复MSE选模；Stage2完全冻结修复器，原MSE+0.2MAE/uniform损失与原PatchTST保持。验证MAE主选择，另存同轨迹MSE选择，两者在测试前锁定。

先起点率0.10/块16（实际约81.94%），再0.15/块8（约72.93%）。每场景从头训练，复用各场景共享20组测试位置和T1/v69参考，不重训基线。主选择若固定验证预测MSE未下降或MAE变差则保留该场景v69；测试用于透明报告，不选配置/seed/checkpoint。

## 验证与记录

原始基线：1132通过、21项已有失败、1项已有跳过，详见baseline-isolated-tests.log。首次未设置PYTHONPATH的collection检查无效，不作为基线；正式命令明确PYTHONPATH=src:.。

新增行为测试先RED后GREEN：TV改变候选且排除未来；隐藏NaN不改变查询；部分可见patch候选不变；无真实观测窗口拒绝；对齐梯度能到查询/T/V而teacher冻结；原权重及预测器初始化/RNG相同；H查询发生在V预览之后。当前相关32项已通过；完整回归和真实GPU预检后再训练。

记录20组均值/ddof1位置标准差、配对胜出、两阶段完整耗时、allocated/reserved峰值、mask/源码/权重SHA、H全缺失覆盖和修复诊断；保存Stage1/2逐位冻结审计。继承旧失败按名称披露。v69工作区及权重不覆盖，不合入主源码。
