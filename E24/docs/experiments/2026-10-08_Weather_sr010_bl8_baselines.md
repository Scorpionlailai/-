# Weather 0.10/8 作者配置基线：后台正式对照

## 任务与固定协议

2026-10-08 授权继续运行 T1 和其他基线。E24-v79 已完成的 Weather 实验作为固定参考，不重新训练、不更改其源码或权重。新增 Weather 适配全部位于 `integrations/weather_baselines/`。

- Weather 原始 52,696 行、21 变量；窗口96，预测96，全变量预测。
- 训练/验证/测试按时间70%/10%/20%；全局标准化器仅拟合训练行。
- 缺失起点概率0.10，块长8，变量独立，允许块重叠。0.10是起点概率，不是最终缺失比例；测试窗口实际缺失平均约57.01%。
- 训练缺失逐轮动态生成；验证固定seed2021；测试复用已封存20组seed2026100801—2026100820，所有模型完全相同。
- 每组保留全部10,444个测试窗口、验证全部5,175个窗口，包括最后不足整批的窗口；训练丢弃末尾不足整批的部分，作者batch差异明确记录。
- 数据、mask、配置身份：`artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json`及其SHA文件。入口逐次校验原CSV和所有mask校验和。
- 不使用测试指标选择参数或checkpoint。20组标准差仅反映缺失位置变化，不是20次训练或训练种子标准差。

## 两阶段插补比较

T1、SAITS、BRITS先按作者Stage1规则训练插补器，再接与E24完全相同的原始PatchTST预测头：D16、4heads、FF128、3层、P16/S8、dropout/fc_dropout0.3。

Stage2严格固定：修复器所有参数、缓冲和模式冻结，纯未来MSE，FP32，Adam，lr1e-4，weight_decay0，固定100轮，无早停，batch128，seed2023，每轮286次更新，总28,600次。预测头初始权重SHA256为 `fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561`，与E24一致。

主checkpoint按单组固定验证MAE选择；同一训练轨迹另保留验证MSE选模版本。两者分别测试20组并明确标注，不能把测试较好的版本重新定义为主结果。修复器末尾插补等内部细节沿用各自官方适配，公共预测头保持一致。此Stage2是本研究缺失回溯窗口预测任务的统一扩展，不宣称是作者论文中未公布的下游训练设置。

Stage1修复验证与测试修复都在全局训练标准化尺度上计分，避免局部近零标准差主导评分。最终测试修复指标来自冻结Stage1权重的显式输出；同时报告全96点及末32点的隐藏坐标MSE/MAE。可见坐标保持输入真实值，不计入修复评分。

## 作者训练配置

| 模型 | Stage1或原生训练 | 轮数上限/早停 | batch | lr/优化器 | 精度 | 选模 |
|---|---|---|---:|---|---|---|
| T1 | 官方Weather脚本；heads128、P2/S1、blocks[2,2]、large kernels[71,31]、small5、FF ratio1 | 300 / patience30 | 16 | 1e-4 / Adam | BF16 | 隐藏修复MSE |
| SAITS | 公开SAITS-base；groups2、D256、FF128、4heads、dk/dv64、drop0.1 | 10,000 / patience30 | 128 | 1e-3 / Adam | FP32 | 隐藏修复MAE |
| S4M | 官方Weather示例；D/FF256、8层、memory200、topK20、short32 | 10 / patience3 | 16 | .005 / SGD momentum.9, WD1e-5 | FP32 | 未来MSE |
| CRIB | 论文2层4heads、D32、P8；公开Weather训练脚本 | 10 / 无 | 16 | .001 / Adam | FP16 | 未来MAE |
| CoIFNet | 论文TableIII：hidden256、时间嵌入8/8、drop.1、lambda.2 | 100 / patience10 | 512 | .001 / Adam | FP32 | 未来MAE |
| BiTGraph | 论文blocks3、topk10；公开small模型channels8/8/16/32、node_dim3、drop.3 | 200 / 无 | 32 | .001 / Adam | FP32 | 未来MAE |
| ChannelTokenFormer | 公开regular模型默认D256、FF1024、8heads、2层、P24、drop.1、global tokens2、keep_prob1 | 10 / patience3 | 32 | 1e-4 / Adam | FP32 | 未来MSE |
| BRITS | 作者README例子hidden108、impute_weight.3、label_weight0 | 1000 / 无 | 64 | .001 / Adam | FP32 | 隐藏修复MAE |

T1修复训练seed502沿用此前锁定选择；其他模型训练seed2023。S4M数据桥配置seed2021仅用于核验固定数据协议，模型初始化和训练随机数使用单独记录的seed2023。未在Weather上搜索“最佳种子”。

T1使用Weather官方lr1e-4，不能套用ETTh1脚本的1e-3。SAITS、BRITS、BiTGraph、ChannelTokenFormer没有明确公开Weather最优配置，按用户授权采用公开基础配置/论文固定架构并标明版本。维度21的必要适配不改变原作者核心文件。

训练损失保留作者定义：T1隐藏修复MSE；SAITS observed reconstruction MAE + hidden imputation MAE；BRITS原始双向ORT求和及一致性权重；CRIB forecast MAE + consistency MSE + 1e-6 KL；CoIFNet .2 future MAE + .8 observed reconstruction MAE；BiTGraph future MAE；S4M/ChannelTokenFormer future MSE。各作者论文配置、公共Stage2配置和任务适配区别保存在机器可读profile，避免笼统声称所有训练完全相同。

## 原生预测模型

S4M、CRIB、CoIFNet、BiTGraph、ChannelTokenFormer直接采用作者原生预测架构，不附加第二个预测头，不与“两阶段插补器”混记。它们的作者原生内部训练照常保留；“冻结Stage2”要求适用于显式修复后接公共预测头的模型。

原生模型没有可核验显式修复输出时，修复指标记N/A。CoIFNet可报告其原生历史重建，但不会将其称为独立Stage1。S4M保留作者每轮首批bank warmup不更新优化器的逻辑，选模保存模型权重及完整CPU记忆库；验证/测试禁止记忆更新，并固定推理抽样随机数。环境没有编译结构化S4核，使用原作者PyTorch fallback，资源报告注明这一限制。ChannelTokenFormer按每个样本的整块可见模式分组，保证官方“按首样本删除缺失patch”的操作对整组均正确；这部分耗时计入训练。

## 资源记录和后台执行

每条正式训练命令记录完整墙钟时间（导入、数据准备、建模、所有训练、验证、保存）、训练阶段时间、PyTorch峰值allocated/reserved GiB、GPU名称/UUID、Torch/CUDA/精度和batch。总训练耗时为所需训练阶段之和；20组最终测试耗时单独列出。失败/重试耗时另列，不能隐藏。预检权重全部废弃，其时间不计入正式训练。

物理GPU1映射为虚拟CUDA0，UUID `GPU-1315e15c-d657-e61f-bde5-6e0b38e6e54b`。启动时存在其他会话的GPU训练任务；队列不停止它们，每个子进程启动与半小时状态记录其并发占用。因此这里的时间是实际完成耗时，后续论文严格速度对比应在空闲独占GPU复核。峰值显存统计只计本进程PyTorch分配，不把其他进程显存计入本模型。

顺序：T1 → SAITS → S4M → CRIB → CoIFNet → BiTGraph → ChannelTokenFormer → BRITS。独立会话后台进程，stdin关闭，日志落盘，关闭IDE或结束当前对话不影响队列。每半小时落盘检查一次进度；每个模型完成立即更新汇总CSV。单阶段失败保留日志，最多按同配置重试一次；已有完整epoch保存时恢复模型、优化器、随机数和更新计数，随后继续其他模型。

启动命令（已有存活队列会拒绝重复启动）：

```bash
cd /home/xyw/paper/E24
PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python -m integrations.weather_baselines.queue launch
```

状态：

```bash
cd /home/xyw/paper/E24
/home/xyw/miniforge3/envs/malr/bin/python -m integrations.weather_baselines.queue status
```

产物根目录：`artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/`。

- `plan.json`：执行协议、种子、环境、顺序与SHA。
- `source_seal.json`、`experiment-source.tar.gz`及SHA：正式运行前源码封存。
- `preflight/*/acceptance.json`：8模型前后向、标签泄漏、同头初始化/冻结验收；正式训练不用这些权重。
- `workflow_checks*/`：废弃小批训练及选中checkpoint保存/加载验证，明确不作为实验结果。
- `queue.pid`、`queue.log`、`status.json`、`journal.jsonl`：持久队列与每半小时记录。
- `<模型>/stage1|stage2|training/`：checkpoint、选模manifest、训练日志、资源sidecar、latest恢复点。
- `<模型>/evaluation/<选模指标>/twenty-test-draws.json`：全部20组分项及均值/标准差，输入/checkpoint身份。
- `docs/experiments/Weather_sr010_bl8_E24_baselines_20261008.csv`：自动更新汇总，原E24结果固定引用；缺失结果留空，不填推测值。

本轮无clean Weather对照，DRR留空；不套用ETTh1 clean结果。

## 参数来源

- T1本地作者脚本 `../T1/scripts/imputation/T1_Weather.sh`、原始 `exp/exp_imputation.py`。
- CRIB [论文Appendix D.1](https://arxiv.org/html/2509.23494)和本地 `../CRIB/bash/bash_train.sh`。
- CoIFNet [论文Table III](https://arxiv.org/html/2506.13064v2)和本地 `../CoIFNet/conf/`；论文lambda.2优先于公开代码.1。
- 其他来源和固定仓库revision逐模型见 `configs/baselines/weather_sr010_bl8_author_20261008.json`；Stage选模manifest保存作者源码摘要。
