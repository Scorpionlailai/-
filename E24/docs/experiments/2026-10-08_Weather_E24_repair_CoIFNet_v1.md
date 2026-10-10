# Weather：冻结 E24-v79 修复结果接入 CoIFNet v1

用户于2026-10-08授权做一版简单接入。先做 Weather 0.10/8、96→96、21变量，继承现有同场景修复权重；不重新训练 Stage1。

这是独立探索协议 `E24-weather-frozen-repair-coifnet-sr010-bl8-v1`，预测器和损失改为作者 CoIFNet，因此不并入原公共 PatchTST Stage2 控制实验。原模型、基线代码、结果及权重保持封存。

输入路径：真实可见历史 → 冻结的 E24-v79 完整修复器 → 全局训练标准化尺度的补全历史 → CoIFNet。可见值逐位保持；原缺失 mask 原样保留，修复值不标为真实观测。

CoIFNet 原 RevON 会把缺失坐标清零，直接调用 `impute(repaired, original_mask)` 无法接入修复信息，并会污染均值求和。因此适配器先将真实可见值和原 mask 交给作者 RevON 计算统计，再用其 `norm-fore` 接口归一化完整补全序列，传入未改动的 shared_model 和 aux_head，最后反归一化。该连接无新增可训练参数；不修改作者源码。时间特征保留作者小时和 day-of-month modulo7。

只训练 CoIFNet：seed2023、Adam lr0.001、weight decay0、FP32、batch512、上限100轮、验证MAE patience10；loss=0.2 future MAE + 0.8 observed-history reconstruction MAE。重建只监督真实可见历史，不把修复结果当真值；隐藏历史及未来标签仅供监督/离线评分，不作为输入。CoIFNet从头初始化，不暖启动已有原生CoIFNet权重。

修复器全部参数、缓冲、训练模式、检索编码器和历史库冻结。修复前向按128样本分块，保持CoIFNet batch512。训练缺失逐轮动态生成，每轮重新修复；固定验证修复可缓存，测试各mask分别计算。每轮及测试结束校验修复、检索与记忆状态身份。

复用原0.10/8数据manifest、固定验证5175窗口、20组测试mask，每组10444窗口。验证选定权重后才测试；这是已查看过的mask库上的探索/回归对照，不能称新的未见确认集或多训练种子结论。

验收：修复值扰动确实影响未来输出；观测统计不依赖修复值；中性填充值与原作者前向一致；替换隐藏历史/未来标签不改变前向；CoIFNet有限梯度；optimizer一步后修复/检索/库不变；验收权重丢弃。

入口：`PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m integrations.e24_coifnet.workflow launch`。等物理GPU1无计算进程后顺序训练/测试，不停止已有任务。supervisor使用独立session，stdin关闭，每300秒记录进展；产物目录保存 `console.log`、`launch.json`、`pid`、`status.json`、源码封存、验收、训练资源和对照CSV。

新训练时间只包含复用Stage1后的修复计算、CoIFNet训练、验证、保存；原Stage1时间另列，不能将新增训练时间称为从头完整模型成本。测试和预检时间分别记录。

## 本次启动与验收

2026-10-08 18:20（Asia/Shanghai）启动独立后台 supervisor，PID1713588、SID1713588、PPID1。目录：`artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_coifnet_v1_20261008_182013/`。

两个归一化/信息传递单元检查通过；真实Stage1权重的CPU验收通过：标签隔离、修复值影响预测、可见路径一致、有限梯度及一次更新后修复/检索/记忆状态不变。预检仅使用训练样本，权重废弃。

启动检查时状态为`waiting_for_gpu`；GPU1正在运行0.15/8的BiTGraph作者基线，本版训练尚未开始。supervisor每15秒检查GPU，连续三次空闲后自动从头训练CoIFNet，再用验证选中的唯一主权重计算20组测试结果。结果写入该目录的`evaluation/comparison.csv`、`evaluation/twenty-test-draws.json`。具体实时阶段以`status.json`和日志为准。

2026-10-08 23:02（Asia/Shanghai）状态查询发现原任务仍未开始训练，GPU1后续ChannelTokenFormer持续占用，而GPU0已空闲。已将等待中的supervisor结束并在原状态中登记`superseded`及替代路径；原日志/源码封存保留。使用`integrations.e24_coifnet.relocate`将完全相同的数值配置移到物理GPU0（RTX5090，UUID `GPU-382ce2ea-81a9-82ea-073f-729eaefd0268`），新独立supervisor PID1718981。新产物目录：`artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_coifnet_v1_20261008_230201_gpu0/`，原Stage1权重、模型结构、损失、batch、精度、seed与mask均不改变。资源对照需按新的实际GPU UUID记载。

## 初始训练诊断（尚无正式测试结果）

GPU0训练首批loss约548342，第6轮固定验证MSE2856.0005、MAE1.4930，明显异常；这些是验证/早期训练指标，不能作为20组测试结果。

在CPU只读重放seed2023的首个512样本训练batch，原生CoIFNet初始loss0.33916，接入修复后loss467626.53（CPU与GPU具体数值不同，但均严重放大）。冻结修复本身该batch隐藏坐标MSE0.08623；原生可见输入归一化最大绝对值7.11，接入补全后最大11776.54。一些变量可见局部std达到RevON的sqrt(1e-5)约0.003162下限，修复值在这些尺度上被极度放大，初始输出最大绝对值约3.26e9。诊断见新产物目录的`diagnose_inputs.py`、`diagnose_inputs.log`、独立后台launch记录。

v1仍按锁定原配置训练并保留负结果，没有将数值尺度修改混入本次轨迹。后续若对归一化下限/缺失表示做稳定化，应另立v2并从头初始化CoIFNet，不能直接覆盖或续训后继续标作v1。

## 用户暂停此接入方向

用户明确要求暂时不进行此类接入。处理请求时，GPU0上的v1训练及测试已于2026-10-08 23:22（Asia/Shanghai）结束，controller/worker均已退出；没有存活接入任务需要终止。保留已完成状态、代码、权重、结果和日志，暂停后续E24修复→CoIFNet接入开发、训练、测试及自动重启，直到用户明确恢复。暂停标记：`integrations/e24_coifnet/PAUSED_BY_USER.json`。
