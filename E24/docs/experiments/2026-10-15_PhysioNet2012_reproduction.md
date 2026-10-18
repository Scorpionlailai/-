# PhysioNet 2012 六模型复现执行记录

用户明确选择15分钟、96→96，延续全96点修复、近32点修复和预测评价。TimesNet、ImputeFormer、BRITS、CSDI、v125、T1各运行四种块缺失场景，先完成24组修复训练与缓存，再完成24组公共PatchTST训练，测试每组20份缺失掩码。双卡独立后台队列已完成；用户查看结果后明确授权同步GitHub。

## 最终结果与发布（2026-10-18）

全部24组修复训练和24组PatchTST训练于2026-10-16 12:57完成并通过后台审核。2026-10-18用户查看完整逐场景表格后明确授权“将这个结果上传到github上”。发布前重新运行27项回归检查并全部通过，完整复核24项状态、1056个缓存文件和480组模型测试记录通过；封存源文件15份未改变。

下表为四场景等权平均，每场景为20组缺失测试均值；每格为MSE / MAE。预测列使用验证集MAE选模主分支，完整CSV同时提供验证MSE次选分支及各缺失组标准差。

| 模型 | 全96点修复 | 近32点修复 | 预测 |
|---|---|---|---|
| TimesNet | 0.696960 / 0.523070 | 0.604819 / 0.503309 | 0.811045 / 0.611190 |
| ImputeFormer | 0.480302 / 0.407154 | 0.401482 / 0.389197 | 0.702142 / 0.568236 |
| BRITS | 0.611180 / 0.483590 | 0.513412 / 0.466429 | 0.714990 / 0.575224 |
| CSDI | 0.813843 / 0.572266 | 0.704187 / 0.542794 | 0.844644 / 0.627928 |
| v125 | 0.484710 / 0.407905 | 0.401988 / 0.387312 | 0.714526 / 0.568348 |
| T1 | 0.647613 / 0.496901 | 0.559791 / 0.475852 | 0.794801 / 0.600975 |

ImputeFormer在四场景平均的全96点修复MSE/MAE、近32点修复MSE和预测MSE/MAE领先；v125在四个场景的近32点修复MAE均第一。sr010_bl8的预测MSE由BRITS领先，sr020_bl8的预测MAE由v125领先。使用单次训练seed2023，微小差异不代表多训练种子统计显著性。

[全部逐场景结果](../../artifacts/physionet_reproduction_20261015/report.md) · [完整结果CSV](../../artifacts/physionet_reproduction_20261015/results.csv) · [最终审核](../../artifacts/physionet_reproduction_20261015/final_saved_output_audit.json) · [发布状态](../../artifacts/physionet_reproduction_20261015/github_publication_status.json)

本次同步文本结果、代码、测试、预处理manifest及共享掩码的种子/生成器/校验信息。约23.2GB修复数组、0.23GB模型权重、145MB共享掩码数组、原始数据和预处理数值数组保留在服务器，不纳入Git提交。共享掩码可在恢复原始数据、完成预处理后用封存生成器重建，并与已发布manifest中的SHA256核对。

从原始数据重建预处理容器及共享掩码（使用记录的依赖环境；原始文件须满足既有下载清单中的SHA256）：

```bash
export PYTHONPATH=/home/xyw/paper/E24/src:/home/xyw/paper/E24
/home/xyw/.local/share/mamba/envs/paper-timesnet/bin/python -m integrations.physionet_baselines.prepare
/home/xyw/.local/share/mamba/envs/paper-timesnet/bin/python - <<'PY'
from pathlib import Path
from integrations.physionet_baselines.data import ROOT, SCENES, SharedData
assets = ROOT / 'E24/artifacts/physionet_reproduction_20261015/shared_assets'
for scene in SCENES:
    data = SharedData('PhysioNet2012', scene, assets)
    print(scene, data.manifest_sha)
PY
```

## 数据口径

使用官方A/B/C集合全部12000份ICU记录；6个下载资产、12003个解压文件SHA256均与既有下载清单一致。按唯一去标识RecordID固定seed2023随机分为8400训练、1200验证、2400测试，不跨记录拼接、不跨记录借历史。RecordID代表一次ICU住院记录，不声称恢复了跨次住院的个人身份。

采用BRITS/CSDI作者的35个动态字段，顺序与作者代码一致；不使用静态描述、死亡结局或其他结局标签。只接受有限且非负的真实观测。每份48小时记录独立放入192个右闭15分钟区间，取区间最后有效观测；00:00归入第一区间，24:00属于历史末端，24:01属于未来首区间，48:00保留，48:01及以后排除。区间输出时刻不早于任何所用原始观测。

前24小时96点作为修复和预测输入，后24小时96点为预测目标；近32点指历史最后8小时。每记录一个窗口，步长192，不对连续记录切滑窗。天然缺失率94.294696%，原始无观测坐标始终不作为可见输入、训练真值或评分真值。13份没有所选动态观测、19份无历史观测、25份无未来观测均保留并记录；缺少真值的坐标权重为零，无监督批次使用有限的零损失。

每记录独立前向填充，开头仅用训练原始观测中位数作数值占位。标准化均值/标准差只拟合训练记录的原始有效观测，不使用填充值、验证或测试数据。空变量的逐窗归一化使用训练标准化空间0均值、1标准差回退。无真实日历时间，TimesNet/ImputeFormer日历协变量置零。

本轮是经用户确认的共同修复→预测协议。BRITS/CSDI作者原PhysioNet实现使用小时网格48步，CSDI使用set-A及五折记录划分；本轮15分钟96→96、全部A/B/C、固定70/10/20划分与其不同，不能将本轮分数称为作者原基准分数的直接复现。

## 参数与执行

TimesNet沿用作者Weather基础配置：batch16、最多10轮、patience3、Adam lr0.001。ImputeFormer沿用作者Air36基础配置：batch16、最多300轮、每轮最多300批、patience20、FP16、lr0.001及原调度器；二者均不声称是PhysioNet目标数据集最佳参数。

BRITS保留作者35维网络、hidden108和原重建/双向一致性损失，沿用此前授权的TorchScript执行、最多200轮、patience20、batch64。CSDI保留作者PhysioNet基础结构和50扩散步，使用此前用户指定的100轮、75/90学习率衰减、训练缓存1样本、修复评价/测试5样本取中位数，名称为CSDI。

v125沿用封存粗修复100轮、64/256两校准分支各100轮、固定1:1；batch128，全负样本对比损失按512行分块并重算梯度，精确保留历史float64键。T1为原结构、输入35变量、100轮、batch128。六个模型均在本数据上独立训练，不迁移其他数据集权重。

四场景为sr010_bl8、sr015_bl8、sr020_bl8、sr010_bl16。掩码逐RecordID生成，六模型共用封存资产；验证seed2021，测试20组2026101501–2026101520。块起点率不等于最终缺失比例。训练掩码逐轮动态变化；Stage2固定epoch0训练掩码并复用修复缓存。

Stage1完整保存训练、验证及20组测试FP32修复结果。Stage2禁止隐式补生成缓存，只训练公共PatchTST：100轮、batch128、FP32、Adam lr1e-4、纯预测MSE；验证MAE主选，验证MSE同轨迹次选，修复参数、缓冲和历史库冻结。单训练seed2023，20组缺失标准差不代表多训练种子波动。

## 验证与后台队列

新增15项及既有PAMAP2的12项检查合计27项通过，覆盖因果时间聚合、患者记录隔离、填充边界、仅训练有效观测统计、空监督零梯度、原始缺失排除、缓存只读、v125历史恢复和全负样本损失梯度一致性。

六模型×四场景GPU预检全部通过：原定真实训练批量参数更新、可见值保留、全空输入、权重往返、缓存接预测头；CSDI生产5样本推理通过。实测batch128时v125峰值allocated16.830GiB，T1峰值8.227GiB。源文件15份已封存，运行期间禁止修改，以保持缓存身份可核查。

队列以独立session、stdin=DEVNULL和文件日志启动，每卡一个任务、自动领取下一任务；全24组修复完成后进入预测。失败记录进入队列状态并阻止不完整结果进入预测。最后自动核验24项模型权重/冻结状态、1056个缓存文件、480组模型测试记录、真值坐标、均值/标准差及RecordID边界。

```bash
PYTHONPATH=/home/xyw/paper/E24/src:/home/xyw/paper/E24 \
/home/xyw/.local/share/mamba/envs/paper-timesnet/bin/python -u \
-m integrations.physionet_baselines.parallel_queue \
--root /home/xyw/paper/E24/artifacts/physionet_reproduction_20261015 --gpus 0 1
```

[实时报告](../../artifacts/physionet_reproduction_20261015/report.md) · [队列状态](../../artifacts/physionet_reproduction_20261015/queue_status.json) · [参数来源](../../artifacts/physionet_reproduction_20261015/parameter_sources.json) · [协议确认](../../artifacts/physionet_reproduction_20261015/protocol_accepted.json) · [数据审核](../../artifacts/physionet_reproduction_20261015/data_preparation_audit.json) · [回归验证](../../artifacts/physionet_reproduction_20261015/regression_verification.json) · [双卡训练实测](../../artifacts/physionet_reproduction_20261015/dual_gpu_verified.json)

数据来源：[PhysioNet官方说明](https://physionet.org/content/challenge-2012/1.0.0/)；动态字段及作者预处理来源：[CSDI dataset_physio.py](https://github.com/ermongroup/CSDI/blob/main/dataset_physio.py)、[BRITS input_process.py](https://github.com/caow13/BRITS/blob/master/input_process.py)。
