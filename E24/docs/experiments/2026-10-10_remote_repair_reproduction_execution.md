# 新服务器：四个修复基线复现执行记录

用户确认范围：TimesNet、ImputeFormer、BRITS、CSDI；ETTh1、Weather；sr010_bl8、sr015_bl8、sr020_bl8、sr010_bl16。先完成32项修复比较，再冻结全部修复器并接入相同公共 PatchTST。最终报告包含全96点隐藏坐标修复、近32点隐藏坐标修复、未来96点预测的 MSE/MAE。用户审核本地报告后才上传 GitHub；执行器不包含上传行为。

新入口为 `integrations.repair_baselines`，不调用原服务器依赖旧权重的控制器。不覆盖既有记录。各作者模型保留独立环境，GPU1顺序执行。正式训练 seed=2023；20组掩码是位置重复，不是20次训练。

| 模型 | 作者参数来源 | Stage1预算与选模 |
|---|---|---|
| TimesNet | ETTh1 / Weather 专用 imputation shell脚本及 run.py 默认值 | 最多10轮，batch16，Adam lr0.001，type1，patience3，固定验证隐藏坐标MSE最小 |
| ImputeFormer | config/imputation/imputeformer_la.yaml 和官方训练入口；非目标数据集最佳配置 | 最多300轮，每轮最多300批，batch32，混合FP16，grad_clip5，magic调度，patience20，验证MAE最小 |
| BRITS | README 示例及 main.py 优化器；非目标数据集最佳配置 | 1000轮，batch64，hidden108，impute_weight0.3，label_weight0（无标签任务），Adam lr0.001，末轮权重 |
| CSDI | config/base.yaml 和 utils.py；非目标数据集最佳配置 | 200轮，batch16，Adam lr0.001，wd1e-6，150/180轮衰减，末轮权重；50扩散步、100样本中位数 |

用户已明确允许后三个模型使用作者基础配置并标注来源。不能称为作者 ETTh1/Weather 最佳配置。BRITS/CSDI 官方入口没有保存“验证最优权重”，故保留其末轮策略，不能自行按测试结果选模。CSDI 每20轮记录50扩散步的固定验证噪声损失。

必要适配：统一切分、训练集标准化、窗口索引和块缺失；修复器不加载未来目标。TimesNet 对整个变量无观测的窗口只将计数分母 clamp_min(1)，已验证非空情况下与作者原实现逐位一致。ImputeFormer 以每变量一个值通道加作者日周期sin/cos两通道输入，保留作者 Freg，FFT 用FP32处理96点窗口，其余训练保留混合FP16。BRITS仅调整时间长、变量数、设备及旧Python兼容，不改作者损失公式。CSDI训练噪声目标真值仅用于损失，不作为可见条件，插补阶段输入仅为真实可见值和掩码。

共享资产：Weather内容checksum与历史manifest一致。ETTh1原二进制资产未包含在迁移快照中；重建掩码的20组timeline/window缺失比例全部与历史记录一致，但npz文件SHA不同，原内容身份尚未验证。用户已确认此次四个模型共用重建后封存的ETTh1掩码，并以 `repair-shared-rebuilt-etth1-v1` 独立报告，不宣称与旧实验严格逐坐标配对。

Stage2保持原公共 PatchTST（P16/S8、D16、4 heads、FF128、3层、dropout0.3），100轮、batch128、FP32、Adam lr1e-4、无衰减、纯未来MSE。修复器完整冻结/eval、不入优化器；每轮和评估前后核对状态SHA。验证MAE主选和MSE次选分别保存。新基线共用原端点复制补齐接口。预检已验证四模型、两数据集、四场景共32例的真实观测保持、全缺失输入有限、训练损失/梯度、预测形状、冻结及同预测头初始化，跨环境初始化SHA一致。

执行目录：`E24/artifacts/repair_reproduction_20261010/`。

- `shared_assets/`：数据、scaler、窗口和各mask身份记录；实际mask二进制保留本地。
- `queue_status.json` / `monitor.jsonl`：当前任务及每300秒日志/进程检查。
- `<dataset>/<scene>/<model>/seed_2023/`：配置、源码封存、权重、逐轮曲线、逐mask指标、资源记录。
- `report.md` / `results.csv` / `review_status.json`：自动更新的本地报告，完整性未通过前标为进行中。
- `configuration_source_seal.json` / `environments/`：作者配置文件SHA、GPU UUID、依赖环境。

队列先完成所有Stage1及20组修复测试，全部成功后才启动Stage2。失败保留日志，不自动覆盖失败产物或跳过全修复完成的阶段屏障。CSDI预检用2样本仅验证接口，正式报告始终100样本；采样耗时计入修复推理，并与训练时间分开。

启动方式（由队列锁避免重复启动）：

```bash
PYTHONPATH=/home/xyw/paper/E24/src:/home/xyw/paper/E24 \
  /home/xyw/.local/share/mamba/envs/paper-timesnet/bin/python -u \
  -m integrations.repair_baselines.queue \
  --root /home/xyw/paper/E24/artifacts/repair_reproduction_20261010 --gpu 1
```

## 用户授权加速版修订

此前表格为初始作者配置记录。用户明确授权 BRITS/CSDI 加速版，当前执行以 acceleration_plan.json 为准。BRITS 保留原网络和损失，TorchScript 循环；上限1000→200轮，patience20，验证MAE选模。CSDI 保留200训练轮与50扩散步，100→10个采样，并行5个样本、中位数聚合。报告明确区分加速变体和作者基础配置；不预先声称精度不变。初始源码封存在 author_adapter_v1，修订源码封存在 accelerated_adapter_v2。

用户澄清预测阶段标准是测试20组缺失掩码。实现已核查，并增加20个不同种子的强制检查。逐组计算预测指标后汇总均值及样本标准差。

## 双GPU执行修订

用户授权“采用双卡并行”。当前入口为 `integrations.repair_baselines.parallel_queue --gpus 0 1`，每卡调度一个独立实验，使用不同输出目录；已完成结果跳过，原单卡正在执行的子进程由新调度器接管，无需重训。公共队列锁防止重复调度，监测两卡任务与失败状态。全部32组修复评估完成后才启动预测阶段，预测测试仍为20组缺失掩码。交接记录见 `dual_gpu_handoff.json`，授权记录见 `dual_gpu_authorization.json`，当前任务见 `queue_status.json`。

## 固定训练掩码与修复缓存（用户确认）

四个新增基线的Stage2统一固定EpochMaskManager第0轮、seed2023训练掩码，FP32缓存所有训练窗口；每轮打乱、batch128丢弃尾批，仍训练100轮。验证缓存一次，测试20组缓存供验证MAE/MSE两个预测权重共用。缓存绑定选中修复权重、实际状态、数据资产、窗口顺序、采样参数，拒绝旧缓存。此训练掩码协议不同于原动态训练协议，报告明确披露。

后续CSDI名称直接写CSDI：Stage1训练100轮，75/90轮LR衰减，50扩散步；Stage2训练缓存1次采样，修复/验证/最终预测评价5次采样取中位数，20组测试掩码保持。已启动两项Weather任务保留200轮、150/180衰减及10次评价采样的封存配置，预测阶段恢复对应采样数以核对修复指标。缓存预检记录区分确定性集成探针与真实未训练CSDI检查，不作为正式结果。

2026-10-10 22:13（北京时间）仅重载双卡调度器以更新报告生成器，旧supervisor320773退出、新supervisor344310接管；训练child336673、337497及/proc启动token原样保留，无重复启动或训练重启。接管记录：cached_protocol_dual_gpu_handoff.json、cached_protocol_supervisor_reload.json。

### Prediction cache reuse guard (2026-10-12)

Forecast test banks now require all 20 completed repair cache manifests before evaluation. Missing outputs fail explicitly instead of silently triggering repair sampling. Two regression tests passed: missing caches never call repair, and both forecast checkpoint selections reuse all 20 existing caches without repair calls. The six pending CSDI jobs were checked against real selected checkpoints, all 120 cache bindings and file hashes passed with repair forward forbidden. The two active legacy Weather workers remain running to finish storing previously unsaved outputs; they loaded the earlier runner before this guard. Existing sealed cache implementation files are unchanged. Evidence: `prediction_cache_read_only_preflight.json` and `source_snapshots/prediction_cache_guard_v4/verification.json`.

### 后续执行必须遵守：保存修复输出，直接接预测头（2026-10-12 用户明确要求）

用户原话：“那你后面一定要记住，修复记录保存直接接上预测头”。此要求适用于全部模型、两个数据集和四种场景，也适用于后续交接继续执行。

- 修复评估时同步保存逐窗口、逐时间点、逐变量的完整修复序列和观测掩码；不能只保存 MSE/MAE、日志或模型权重。20组测试掩码分别保存完整输出。
- 缓存封存修复权重、数据资产、窗口顺序、缺失掩码、实际采样参数与实现身份，并核验文件校验和。单组输出全部写完后才生成完成 manifest；部分写入不能算完成。
- 训练和验证所需的对应掩码修复结果各生成一次并缓存，预测训练各轮直接复用。预测测试直接读取修复评估保存的20组输出，MAE/MSE两种选模权重共用相同修复输入，不重新采样。
- 修复器保持冻结；PatchTST只训练预测头。缺失或身份不匹配的测试缓存必须明确报错，不能在预测测试阶段自动重跑修复。
- 进度汇报分别核查修复指标和可复用输出的完整性。预测交接前必须检查20/20组输出都已封存，避免把“指标已计算”当成“输出已保存”。

当前旧两项 Weather CSDI 漏存输出的补齐属于历史补救，不作为后续执行方式。新增要求的机器记录见 `repair_output_reuse_requirement_20261012.json`。

### 最终审核（2026-10-13）

32项修复评估和32项公共PatchTST预测均已完成，最后任务状态时间为北京时间05:14。每项封存训练、验证及20组测试缓存，共640组测试修复输出、1408个缓存文件，实际文件SHA和修复权重实际状态SHA全部通过。全部逐组均值、ddof=1标准差和误差总和/坐标数已核算。

报告原先用修复误差字典逐位相等判断冻结，12项因数值细微差异误报；原始指标最大绝对差4.4151806433134766e-08。修订后权重SHA、缓存状态、掩码身份、窗口及坐标数必须一致，两个预测分支的修复指标必须逐位相等；原评估与缓存评估的MSE/MAE允许2^-23绝对误差、零相对容差，并单独保留逐位一致性与最大差异。原始实验结果未改写。五个报告审核回归用例通过，覆盖实际全部结果、明显指标差异、掩码变化、修复父权重变化和缺少完整输出核验。

最终报告：report.md、results.csv、review_status.json；与v125/T1对照：comparisons/final_repair_forecast_vs_v125_t1_20261013.{md,csv}。旧Weather v125的20%×8及10%×16预测在本地参考资产中无完整20组结果，留空。ETTh1历史对照继续按独立掩码协议披露。证据：final_saved_output_audit.json、final_review_summary_20261013.json、source_snapshots/report_audit_v5。等待用户决定是否上传；未执行GitHub上传。
