# Beijing Air Quality 基线复现执行记录

**已完成：24组修复、24组预测，2026-10-14 03:13（北京时间）通过最终核验。结果已同步GitHub（2026-10-14）。**

数据与协议已由用户确认：UCI Beijing Multi-Site 12站点PM2.5，35064小时、12变量。先TimesNet/ImputeFormer/BRITS/CSDI，再v125/T1，同一组四场景和20组测试缺失。用户要求确认两张GPU均实际训练后才结束会话。

原始压缩包及16个原始文件已恢复，SHA与此前下载记录一致。原生缺失8739坐标（2.0769%），原始观察掩码已恢复为无对象数组的NPZ。训练数值容器为因果前向填充，开头空缺用训练中位数；原生缺失仍作为缺失输入，不监督、不评分。训练标准化器仅拟合训练段填充序列。时间切分train [0,24544)、validation [24544,28050)、test [28050,35064)，窗口数24353/3411/6919，保留切分边界之前96点历史。

四个场景sr010_bl8、sr015_bl8、sr020_bl8、sr010_bl16；新的20组测试种子2026101301..2026101320，共用封存掩码。全96及近32评分只计原本有真实记录且被人工遮挡的坐标；预测只计有真实记录的目标。缺失率是块起点率，允许重叠，不能当最终缺失比例。训练seed统一2023；20组位置标准差不代表训练种子标准差。

代码独立放在E24/integrations/beijing_baselines，保留既有ETTh1/Weather源码与缓存绑定。六个模型各四场景的GPU预检均通过：真实梯度、空样本/空变量、可见值保留、检查点往返、公共头相同初始化和仅复用缓存的预测优化步骤。6个协议回归测试通过。

TimesNet用作者Weather插补基础参数（64/64宽度，10轮，patience3）；ImputeFormer用作者Air36基础配置（不是UCI Beijing12，batch16，上限300轮，patience20，每轮最多300批）；BRITS保留作者网络隐藏108、原损失及已授权TorchScript，上限200轮、patience20；CSDI100轮、75/90衰减、50步扩散、训练缓存1采样、验证/测试5采样中位数。均公开来源，不称前三者为目标数据最佳配置。v125来自封存20261009模型快照，共享粗100轮，64/256校准各100轮，固定1:1；原生缺失仅用于掩码，不作为检索教师或观察原型。T1原作者结构，输入12变量，100轮；两者从本数据重新训练，无跨数据集权重转移。

每项Stage1先保存训练、验证、20组测试全部FP32修复输出并封存权重/代码/数据/掩码SHA。Stage2只读缓存，缺失即报错，不允许调用修复器生成新输出。公共PatchTST100轮、batch128、FP32、Adam lr1e-4、纯预测MSE，验证MAE主选及MSE次选。所有修复阶段完成后自动进入预测阶段；最终自动重算计分支持、均值、ddof1标准差，核对实际修复状态、冻结状态及1056个缓存文件SHA。

队列通过start_new_session脱离会话，日志落盘；每张卡一个任务，自动承接下一项；锁防止重复队列。遇到失败会留下失败记录，不静默跳过、不伪造完成，预测屏障禁止缺修复结果继续。用户已于2026-10-14授权将当前结果同步GitHub；实际状态见[发布记录](../../artifacts/beijing_reproduction_20261013/github_publication_status.json)。权重和大型修复缓存保留本地，共用缺失掩码随结果上传。

运行入口：

```bash
PYTHONPATH=/home/xyw/paper/E24/src:/home/xyw/paper/E24 \
/home/xyw/.local/share/mamba/envs/paper-timesnet/bin/python -u \
-m integrations.beijing_baselines.parallel_queue \
--root /home/xyw/paper/E24/artifacts/beijing_reproduction_20261013 --gpus 0 1
```

[当前报告](../../artifacts/beijing_reproduction_20261013/report.md) · [队列状态](../../artifacts/beijing_reproduction_20261013/queue_status.json) · [确认协议](../../artifacts/beijing_reproduction_20261013/protocol_accepted.json) · [数据核查](../../artifacts/beijing_preparation_20261013/preparation_audit.json)
