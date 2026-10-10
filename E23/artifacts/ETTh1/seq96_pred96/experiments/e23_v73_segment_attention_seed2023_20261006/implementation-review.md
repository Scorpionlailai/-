# E23-v73 新模块独立实现复核

依照 requesting-code-review 流程，聚焦新增 `segment_attention.py`、model/config/train接入差异、新测试及controller/preflight。未重新审阅旧版本全部源码或历史报告，未改冻结代码、启动训练或GPU工作；仅执行CPU前向、反向数学检查和只读哈希审计，写入本复核文件。

## 严重程度与就绪性

- **Critical：无。**
- **Important：无。**
- **Minor：无新增发现。**
- **就绪性：** 本轮新增实现满足已批准的有界设计，修正后的controller、完整回归和CUDA预检证据齐备，冻结定义与归档一致，可启动唯一正式seed2023轨迹。此结论不预测训练效果，也不证明已优于T1。

## 已独立检查

1. **几何、范围与单位。** 对28条96点序列用独立逐段标量算法复算length/phase，窗口边界、全隐藏/全可见、单点、观察分割及短段均一致。门控使用原 `sequence_missing & sequence_validity`，仅长度≥16坐标写入；global残差除原正且有限query_std回到接口坐标。独立前向测试确认小std时仍写入固定global幅度，前64长段可被修改，短段/无效分割段及原观测保持，coverage不变。
2. **注意力与数据路径。** 143维输入由已有CNN/HVT latent、coarse repair、原可见值、原mask/位置/距离、变量标识和隐藏段几何构成；只读原96点回溯，无future target输入。2层4头pre-norm attention/FFN的null key/value经bias-free attention保持零。独立检查全无valid上下文的block输出精确零、输入梯度零、参数梯度有限。全无观测但valid的窗口保持有限；原patch管线对all-invalid patch/window的拒绝仍保留。
3. **排除值与梯度。** 在非零输出权重下，将隐藏raw observed值及无效sequence/point/features分别换成NaN与Inf，valid修复值与有限参考逐位相同。独立backward确认这些被排除坐标梯度精确零，全部新模块梯度有限，attention梯度非零；单独folding检查也确认all-invalid latent patch不污染输出或梯度。这些检查没有优化器更新或训练。原已选参数、库与模型产生的合法修复feature才进入全窗注意力，修复值不改写原观察mask。
4. **接入、初始化与身份。** 新读出在原tail refiner之后、recent mean之前；无新增预测器接入。输出Linear的weight/bias均零初始，初始化在fork_rng内。独立新/旧模型测试确认原body状态、初始化后的RNG、预测器参数及初始forecast逐位相同。新模块实际82113参数；实际配置CUDA预检总修复503086、原预测器35427参数。manifest与checkpoint均经原严格identity验证路径，绑定143维、96范围、16门槛、2层4头、64宽、dropout.1、global单位、null0及原mask等字段；归档行为测试包含双选择Stage2 roundtrip、完整冻结及scope篡改拒绝。
5. **实验控制。** controller锁定seed2023、100轮、full frozen、原MAE主/MSE次同轨迹、原loss/lr/padding，配置SHA及新mode身份；保留开发测试已曝光声明。预检使用实际原观测训练patch填充throwaway bank，两步有限，52个长段；第二步attention梯度0.261545、原position梯度0.212116，Stage2一步后完整修复器未改变，不评分测试。
6. **证据与冻结。** 独立运行4项非训练前向行为测试：4通过（1.41秒）；4项controller：4通过（0.03秒）。读取6项新行为RED→GREEN及双checkpoint/训练梯度归档证据，未重复其训练步骤。完整回归日志1159通过/9失败/1跳过，9个失败名称与baseline1153通过/9失败/1跳过精确相同。524源码、2451保护文件、129参考项及controller/plan去重后2979路径全部SHA匹配；524源码归档逐项匹配。仅3个原文件改动及3个新文件，与声明一致。

## 检查限制

未执行正式训练、GPU预检或全套回归；GPU/完整回归结论来自已冻结的原始日志，数学、排除梯度、前向兼容及哈希为独立CPU复核。零初始化兼容验证的是初始化状态与eval函数，不代表加入dropout的新模块会保持后续训练的随机轨迹。原观察/短段在本读出内不被改写，后续既有recent mean仍按其原逻辑执行。此轮是否缩小长缺失修复和预测差距、严格训练后冻结、完整资源汇总及是否达到T1目标，须在正式训练后另作产物复核；20个已曝光开发位置不能表述为新盲测。
