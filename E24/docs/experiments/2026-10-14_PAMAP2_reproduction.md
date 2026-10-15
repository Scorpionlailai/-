# PAMAP2 六模型复现执行记录

用户确认采用10Hz方案：96点回溯、96点预测，各9.6秒，切窗步长16。六模型为TimesNet、ImputeFormer、BRITS、CSDI、v125、T1，四场景，共24项修复训练与24项公共PatchTST训练，测试20组缺失。双卡后台队列，先四个基线修复，再v125/T1修复；全部修复完成后自动进入预测阶段。

## 最终结果与审核（2026-10-15）

24项修复训练、24项公共PatchTST训练已全部完成；2026-10-15 13:30后台审核通过，发布前再次完整核验通过：1056个缓存文件SHA256、480组模型测试记录、24项权重/冻结状态、天然观测评分和受试者/记录边界。12项回归测试再次通过。用户随后明确授权“上传github”。

下表为四场景等权平均，每场景先对20组缺失测试取均值；每格为MSE / MAE，越低越好。预测使用验证MAE选模的主分支，验证MSE次选分支和缺失组标准差见完整CSV。

| 模型 | 全96点修复 | 近32点修复 | 预测 |
|---|---|---|---|
| TimesNet | 0.666293 / 0.374531 | 0.679302 / 0.380579 | 0.779285 / 0.434026 |
| ImputeFormer | 0.567941 / 0.313659 | 0.576479 / 0.318239 | 0.733077 / 0.415891 |
| BRITS | 0.787303 / 0.508045 | 0.805622 / 0.517011 | 0.748410 / 0.435661 |
| CSDI | 0.553809 / 0.297745 | 0.563637 / 0.302742 | 0.731859 / 0.415415 |
| v125 | 0.385473 / 0.259446 | 0.394197 / 0.264441 | 0.728827 / 0.415459 |
| T1 | 0.414998 / 0.271632 | 0.425054 / 0.277139 | 0.756621 / 0.423688 |

v125在四场景的全96点及近32点修复MSE/MAE均第一；预测MSE三场景第一、预测MAE两场景第一。T1在sr010_bl8预测MSE/MAE第一；CSDI在sr010_bl16预测MAE第一。单训练种子2023，20组缺失的标准差不代表多训练种子波动。

[逐场景完整结果](../../artifacts/pamap2_reproduction_20261014/report.md) · [全部结果CSV](../../artifacts/pamap2_reproduction_20261014/results.csv) · [最终审核](../../artifacts/pamap2_reproduction_20261014/final_saved_output_audit.json) · [发布状态](../../artifacts/pamap2_reproduction_20261014/github_publication_status.json)

本次同步结果、代码、测试、预处理元数据及84份小型共享掩码。约36.5GB修复数组、0.67GB模型权重、原始包和预处理数组保留在服务器；完整文件身份可由预处理manifest及缓存manifest核对。

## 数据及边界

UCI原始包688167640字节，SHA256为6c2282b5b52775cd0588d450d10a315260c94fa3228b9fe453f48c47fda106f4。21个发布分卷、完整ZIP和19个原始文件均校验通过，14份Protocol/Optional记录覆盖9人。101–106训练，107验证，108–109测试；同一人的全部会话始终同组。

保留心率及手腕/胸/踝各13个传感器字段，共40通道；时间戳、活动标签不作为特征，无效姿态列排除。保留原始两个量程加速度，作者README注明±6g可能饱和；本协议未按测试表现筛选传感器。

先按文件、活动变化、活动0区间、时间倒退/重置和超过0.015秒的原始间断分段，再逐段按右闭100ms区间取已有观测均值，输出时刻不早于任何输入样本。活动0按作者要求排除，删除后两侧不拼接。得到272614行、126段，其中124段可产生完整窗口。训练/验证/测试行数191344/23289/57981，窗口10958/1306/3349；每个窗口的历史96点和目标96点必须同段。

每段独立前向填充，开头缺失仅用训练观测中位数作有限数值占位；保留原始聚合观察掩码（25715个缺失坐标，约0.236%），模型不将天然缺失视为观测，监督和评分也排除这些坐标。标准化器仅拟合训练填充序列。PAMAP2只有相对秒数，无真实日历，TimesNet/ImputeFormer日历通道置零。

验证种子2021；测试20组2026101401–2026101420。每段独立生成块缺失，种子由组种子与段ID的SHA256派生，六模型共用封存矩阵。训练为每段EpochMaskManager，预测训练固定epoch0缓存。块起点率不等于最终缺失比例。

## 模型、训练与复用

作者参数及来源写入parameter_sources.json和各plan.json。TimesNet使用Weather插补基础配置（10轮，batch16，patience3）；ImputeFormer使用Air36基础配置（上限300轮、batch16、每轮最多300批、patience20），均非PAMAP2最佳参数声明。BRITS原网络隐藏108、原损失、TorchScript，上限200轮、batch64、patience20。CSDI100轮、75/90衰减、50扩散步，训练缓存1样本，验证及测试5样本取中位数。

v125封存结构保持共享粗修复100轮、64/256校准各100轮、固定1:1；batch128保持不变。40通道下原始全负样本矩阵显存不足，使用512行分块和梯度重算，保留所有负样本及同一交叉熵定义；值与梯度等价回归测试通过。反序列化时精确保留原float64原型键，避免旧加载器重复归一化改变最后位。实际batch128两步、非空历史库预检峰值allocated19.624GiB、reserved22.629GiB；运行时开启expandable_segments缓解碎片。T1原作者结构，输入40，100轮、batch128，实测峰值9.376GiB。

六模型×四场景GPU预检全部通过，包含原定训练批量优化、可见值保留、空变量/空样本、检查点往返和缓存接预测头。CSDI生产5样本推理也已验证。12项回归测试通过，包括因果聚合、跨边界保护、天然缺失排除、缓存只读、历史库精确恢复及对比损失梯度等价。

Stage1保存训练、验证和20组测试的FP32修复输出；Stage2只读这些缓存，缺失即报错，不能隐式生成修复。公共PatchTST100轮、batch128、FP32、Adam lr1e-4、纯预测MSE，验证MAE主选，验证MSE同轨迹另存；修复器完全冻结。最终自动核查24项全部缓存、真实权重状态、20组计分坐标、均值及ddof1标准差。单训练seed2023，位置标准差不代表训练种子标准差。

## 后台运行

队列采用start_new_session、stdin=DEVNULL、落盘日志与进程锁，脱离会话后继续运行。每卡一个任务，自动领取下一项；如失败会记录并阻止缺结果进入预测。队列本身不自动推送GitHub；本轮已在结果呈现后获得用户明确上传授权。

```bash
PYTHONPATH=/home/xyw/paper/E24/src:/home/xyw/paper/E24 \
/home/xyw/.local/share/mamba/envs/paper-timesnet/bin/python -u \
-m integrations.pamap2_baselines.parallel_queue \
--root /home/xyw/paper/E24/artifacts/pamap2_reproduction_20261014 --gpus 0 1
```

[当前报告](../../artifacts/pamap2_reproduction_20261014/report.md) · [队列状态](../../artifacts/pamap2_reproduction_20261014/queue_status.json) · [数据核查](../../artifacts/pamap2_reproduction_20261014/data_preparation_audit.json) · [双卡实测](../../artifacts/pamap2_reproduction_20261014/dual_gpu_verified.json)

数据来源：[UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)；[作者README](https://archive.ics.uci.edu/ml/machine-learning-databases/00231/readme.pdf)。
