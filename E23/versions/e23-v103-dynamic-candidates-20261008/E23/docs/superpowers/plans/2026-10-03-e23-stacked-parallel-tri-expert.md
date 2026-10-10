# E23 v30 Stacked Parallel Tri-Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现独立参数的多层、多组完整H/V/T修复网络，完成1层8组、4层1组、4层8组的训练与共同场景评估。

**Architecture:** 历史候选与key/原型库由一个共享前端提供。每层各组独立计算一次T→V→H，组维批量运算后按门控和拼接投影融合成64维，四层使用不同参数。复用现有显式修复及宽度16的PatchTST接口。

**Tech Stack:** 项目现有Python/PyTorch 2.7.1+cu128、pytest、YAML、NumPy、CSV；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-03-e23-stacked-parallel-tri-expert-design.md`，用户于2026-10-03确认书面设计。

## Global Constraints

- 工作目录`/home/xyw/paper/E23`，已批准分支`paper/e23-migration`。先捕获dirty基础，再保存任务增量；不能把先前工作整体混入提交。
- 用户已有`docs/experiments/E22_all_datasets_missing_scenarios.csv`保持SHA `530947f4a58aaba96f23d7a254cc0b279bc94fc78b70c228869fcef484721fa2`。
- ETTh1 P4，96→96，7变量，patch16/stride8，12patch；每组及融合输出64维，现有预测器宽度16。
- H one_group/top4/阈值0.7；V window_context_topk_v1/top4/offsets(-1,0,1)；T radius2/boundaries(2,4,8,16)。原mask、validity、rho、query mean/std固定。
- 每层/组独立拥有H消息、完整V/T及router参数；共享key/EMA/索引/原型库、原始输入、初始编码、decoder及预测器。
- G8融合：134→64→1的GELU组打分、独立组bias、65→1的no-change、9项softmax，512→64无bias投影；门控零影响初始化，投影八个64列块初始化为单位阵。G1省略新增融合。
- 新模式`stacked_parallel_tv_h_v1`；layers为1或4，groups为1或8，rounds固定1，activation_checkpointing默认false。旧模式与checkpoint保持兼容。
- 每模型从头Stage1/Stage2各100epoch，seed2021，batch128，FP32，lr0.001/0.0001，weight_decay0。Stage1仅最终层全局hidden MSE+0.1 retrieval，specialist exposure0.2；Stage2仅最终层forecast MSE+0.2 MAE，按固定验证MAE选权重。
- 正式训练顺序l1_g8→l4_g1→l4_g8；旧v29-r1/v28/T1不重新训练。按用户最新要求，每个模型完成自己的验证锁定并重放后，立即评估既有三组test mask一次，seed2021/2026100201/2026100202；后续配置和选模规则继续冻结。
- 优先物理GPU1，记录UUID与并发条件。完整时间含准备、训练、验证、保存，测试与诊断另记；报告训练峰值allocated/reserved GiB。
- root为`artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v30_stacked_parallel`。训练前冻结全部实现/config/test/script哈希，训练期间不修改。
- 下文命令默认cwd为项目根；Python/git路径先设`PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH`，`PYTHONPATH=src:.`，`OMP_NUM_THREADS=4`，`MKL_NUM_THREADS=4`。
- Tasks1–6各自GREEN后归档该任务增量并选择性提交已核对的新文件；先前dirty文件的整文件不得盲提交。正式训练前保存所有dirty基础与累计增量，保证可重建。

## Review Focus

1. 激活重计算重新执行forward时，历史检索/EMA/库写入不能重复；归Task5、Task7的副作用计数和梯度对照测试。
2. 某组或专家关闭后，经另一变量来源状态或另一层仍可能泄漏；归Task4的真实状态扰动和singleton测试。
3. 完整组拥有独立V嵌入，但批量路径误复用第一组top4；归Task2的受控不同来源选择与独立梯度测试。
4. 全无有效来源、单变量、padding、极小query_std导致空softmax或错误尺度；归Tasks2–5的有限性、精确回退和全局损失测试。
5. 相同形状的旧多头记录、错误层/组身份或过期诊断混入报告；归Tasks5–6的加载拒绝和checkpoint/config/hash绑定测试。

---

## File Structure

- `historical_candidates.py`：共享检索前端与raw候选缓存；`historical_messages.py`：单个H消息编码；旧`historical.py`保留公开类/导入和旧state_dict键。
- `stacked_group.py`：一组完整H/V/T与router参数所有权；`group_ops.py`：从独立模块参数构建带组维的linear/LayerNorm/MLP运算，保留autograd；`stacked_group_kernels.py`：批量H/V/T及router数学。
- `stacked_fusion.py`：组门控与512→64残差融合；`stacked.py`：独立组/层所有权、全局exposure、逐层控制与诊断类型。
- 既有`model.py`/`integrations/e23`：模式派发、配置、训练及身份；既有评价脚本：新模式标签/元数据。
- `profile_e23_v30.py`/`verify_e23_v30_compatibility.py`：成本及兼容验证；`diagnose_e23_v30_stack.py`：固定权重逐层/组诊断；`report_e23_v30_trial.py`/`run_e23_v30_trial.py`：矩阵、选择、报告及顺序执行。

### Task 1: 分离历史候选前端与独立H消息

**Files:** Create `src/malr/models/e23/historical_candidates.py`, `historical_messages.py`; Modify `historical.py`; Create `tests/unit/test_e23_historical_provider.py`。

**Interfaces:** `HistoricalCandidateProvider(patch_length:int,d_model:int,retrieval_dim:int,variable_count:int,*,topk:int=4,similarity_threshold:float=0.7,top_p:float|None=None,candidate_cap:int=8,retrieval_strategy:str='one_group')`拥有key_encoder/retrieval/index，提供`refresh_snapshot(snapshot)`、`prepare_candidates(inputs:ExpertInputs)->HistoricalCandidateCache`。`HistoricalMessageEncoder(patch_length:int,d_model:int,*,contextual:bool)`仅拥有消息/质量/状态匹配参数，`from_cache(inputs,cache,*,round_embedding:Tensor|None=None)->ExpertEvidence`。为保留旧键，采用共享的初始化/消息计算函数；旧HistoricalExpert不能把原顶层message_projector改注册到新子路径。

- [x] **Step 1:** 捕获当前tracked/untracked源代码、配置、测试、脚本的基线及SHA；存入root/implementation-evidence，排除artifacts、缓存及.git，另存用户CSV与git status。沿用已批准当前分支；读取using-git-worktrees技能并据现有授权判断是否需独立scratch，不能从HEAD丢失dirty基础。
- [x] **Step 2:** 写`test_provider_has_only_retrieval_parameters`、`test_provider_cache_matches_existing_historical_cache`、`test_message_encoder_matches_existing_message_and_gradients`：沿用`tests/unit/test_e23_historical._inputs/_memory`，复制同名消息权重，比较真实候选、消息/质量、权重与梯度；断言message encoder没有key/EMA/index。

核心断言：

```python
assert provider.state_dict()
assert all(name.startswith("key_encoder.") for name in provider.state_dict())
assert "key_encoder" not in dict(message.named_modules())
torch.testing.assert_close(actual.message, expected.message, rtol=0, atol=0)
```

- [x] **Step 3:** 运行`python -m pytest tests/unit/test_e23_historical_provider.py -q`，记录缺少新类的RED。
- [x] **Step 4:** 实现provider/消息分离，保留旧类导入、构造参数和state_dict键；缓存训练张量不来自detached provenance。
- [x] **Step 5:** 运行新测试与`tests/unit/test_e23_historical.py tests/unit/test_e23_historical_cache.py tests/integration/test_e23_model.py`；检查空库、未来成员排除、top-p与旧H政策不变，保存GREEN及任务增量。

### Task 2: 独立参数的组维批量专家计算

**Files:** Create `src/malr/models/e23/stacked_group.py`, `group_ops.py`, `stacked_group_kernels.py`; Create `tests/unit/test_e23_group_ops.py`, `test_e23_grouped_experts.py`。

**Interfaces:** `grouped_linear(x:Tensor,layers:Sequence[nn.Linear])->Tensor`、`grouped_layer_norm(x:Tensor,layers:Sequence[nn.LayerNorm])->Tensor`、`grouped_mlp(x:Tensor,layers:Sequence[nn.Sequential])->Tensor`，特征张量的倒数第二维为G、最后一维为feature；参数由原独立模块torch.stack构建，不能detach或重新包装成替代Parameter。

`TVHGroup(variable_count:int,patch_length:int,d_model:int,*,variable_topk:int=4,variable_offsets:tuple[int,...]=(-1,0,1),temporal_local_radius:int=2,temporal_boundaries:tuple[int,...]=(2,4,8,16))`在stacked_group.py中拥有historical:HistoricalMessageEncoder、variable:SparseVariableExpert、temporal:MultiScaleTemporalExpert、router:ExpertRouter及零初始化的`condition_embedding:nn.Embedding(3,D)`。各专家启用contextual，条件使用embedding第0项，整个3×D参数形状沿用v29便于数值映射；构造每组新网络，不复制随机初始化结果。

`GroupedExpertEvidence(message:Tensor,quality:Tensor,available:Tensor,provenance:Mapping[str,Tensor])`分别为`[B,C,N,G,D]`、同形、`[B,C,N,G]`。`GroupedRoundOutput(latents,weights,deltas,preview_gates,available)`形状分别为`[B,C,N,G,D]`、`[B,C,N,G,4]`、各专家修正、`[B,C,N,G,2]`、`[B,C,N,G,4]`。

`prepare_grouped_variable_sources(inputs:ExpertInputs,experts:Sequence[SparseVariableExpert],*,epoch:int,explore:bool)->tuple[VariableSourceCache,...]`保留每组独立选择与可微prior。`grouped_historical(inputs,states:Tensor,cache,encoders:Sequence[HistoricalMessageEncoder],embeddings:Tensor)->GroupedExpertEvidence`；`grouped_variable(inputs,states,experts:Sequence[SparseVariableExpert],caches:Sequence[VariableSourceCache],*,source_context_allowed:Tensor|None,embeddings:Tensor)->GroupedExpertEvidence`；`grouped_temporal(inputs,states,experts:Sequence[MultiScaleTemporalExpert],*,source_context_allowed:Tensor|None,embeddings:Tensor)->GroupedExpertEvidence`。states为`[B,C,N,G,D]`，embeddings为`[G,D]`；原H缓存共享，V来源每组独立选择，T每组独立编码。`grouped_tv_h(inputs,groups:Sequence[TVHGroup],h_cache,v_caches:Sequence[VariableSourceCache],allowed:Tensor,source_context_allowed:Tensor|None)->GroupedRoundOutput`实现一次T→V→H；allowed为`[B,C,N,G,3]`，source_context_allowed为`[B,C,N]`或None。

- [x] **Step 1:** 写组linear/MLP/LayerNorm对照测试，用不同权重的两组，与逐组原PyTorch模块结果及输入/每组参数梯度比较。写`test_grouped_operations_do_not_alias_or_detach_parameters`，扰动一组不会改变另一组输出。

独立linear oracle（用float64测输出及相同loss下的梯度）：

```python
expected = torch.stack([layer(x[..., g, :]) for g, layer in enumerate(layers)], dim=-2)
actual = grouped_linear(x, layers)
torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
assert layers[0].weight.data_ptr() != layers[1].weight.data_ptr()
```

- [x] **Step 2:** 写`test_grouped_h_v_t_match_independent_reference`，用真实非空H库、V四来源及T方向band，对照现有from_cache；写`test_grouped_v_uses_each_groups_source_embeddings`，受控变量嵌入让两组选出不同来源，断言批量结果与各自独立选择一致。
- [x] **Step 3:** 写`test_grouped_single_variable_and_empty_evidence_are_finite`及padding/隐藏输入测试；写受控T/V修正下的`test_grouped_tv_h_matches_one_round_reference`，比较预门控、来源状态、H条件与最终一次写回。运行两新文件并保存RED。
- [x] **Step 4:** 实现组维数学，H/V/T候选维位于组维前，V将K×offset展开成候选维；保留原权重归一化、有界状态匹配、来源资格与singleton回退。结构/有限性检查在组维边界执行，不移除旧路径检查。
- [x] **Step 5:** 运行两新文件及既有T/V/router/context测试；float64小样例对照误差≤1e-10、float32≤1e-6，有限非零梯度进入每组完整网络；保存证据与增量。

### Task 3: 组门控、拼接投影及精确回退

**Files:** Create `src/malr/models/e23/stacked_fusion.py`; Create `tests/unit/test_e23_stacked_fusion.py`。

**Interfaces:** `GroupFusionOutput(latents:Tensor,group_weights:Tensor,projected_deltas:Tensor)`；`StackedGroupFusion(d_model:int,group_count:int).forward(base:Tensor,deltas:Tensor,routes:Tensor,rho:Tensor,available:Tensor)->GroupFusionOutput`。base为`[B,C,N,D]`，deltas为`[B,C,N,G,D]`，routes为`[B,C,N,G,4]`，available为`[B,C,N,G]`；返回group_weights为`[B,C,N,G+1]`，projected_deltas为`[B,C,N,G,D]`，保留每组独立投影贡献。原始rho=1位置精确回到base。

- [x] **Step 1:** 写`test_fusion_initializes_as_uniform_valid_residual_sum`：D64/G8，手工delta，断言可用组加none均匀、projection每块为I、输出为base+加权修正和；写`test_fusion_missing_groups_and_fully_observed_patches_are_exact`，全无证据/观测完整时精确等于base。

八组全有效且rho<1的核心断言：

```python
torch.testing.assert_close(out.group_weights, torch.full_like(out.group_weights, 1 / 9))
torch.testing.assert_close(out.latents, base + deltas.sum(dim=-2) / 9)
```

- [x] **Step 2:** 写`test_single_group_is_exact_bypass_without_new_parameters`、`test_fusion_gradients_reach_every_valid_group_and_projection`、错误shape及非有限输入拒绝；运行文件保存RED。
- [x] **Step 3:** 实现spec固定134/65维打分、门控及拼接投影；D较小时输入维数为2D+6/D+1，正式D64保持134/65。G1省略新增参数且直接使用单组修正。
- [x] **Step 4:** 同一文件GREEN，权重合法、无效组权重为0、无证据有限；保存增量与证据。

### Task 4: 独立层/组控制与全局specialist exposure

**Files:** Create `src/malr/models/e23/stacked.py`; Create `tests/unit/test_e23_stacked.py`。

**Interfaces:** 消费Task2的TVHGroup、grouped_tv_h及prepare_grouped_variable_sources，和Task3的StackedGroupFusion。

`StackLayerDiagnostics(layer_index:int,latents:Tensor,routes:Tensor,preview_gates:Tensor,group_weights:Tensor,group_deltas:Tensor,allowed:Tensor,update_rms:Tensor)`。`StackedRepairState(router:RouterOutput,evidence:Mapping[str,ExpertEvidence],layers:tuple[StackLayerDiagnostics,...],specialist_selected:Tensor|None)`。

`StackedTVHController(variable_count:int,patch_length:int,d_model:int,*,layers:int,groups:int,activation_checkpointing:bool=False,variable_topk:int=4,variable_offsets:tuple[int,...]=(-1,0,1),temporal_local_radius:int=2,temporal_boundaries:tuple[int,...]=(2,4,8,16)).forward(inputs:ExpertInputs,*,historical:HistoricalCandidateProvider,epoch:int=0,explore_variable:bool=False,specialist_exposure:float=0.,specialist_seed:int=0,disabled_experts:frozenset[str]=frozenset(),disabled_groups:frozenset[tuple[int,int]]=frozenset())->StackedRepairState`；关闭组的tuple索引从0起。

- [x] **Step 1:** 写`test_stack_parameters_are_distinct_across_layers_and_groups`、`test_next_layer_reads_previous_fused_state_without_detach`、`test_stack_historical_lookup_occurs_once`；沿用v29真实库fixture，直接类小样例L2/G2/D8，正式配置另验4/8/64。
- [x] **Step 2:** 写`test_stack_specialist_sample_occurs_once_across_all_groups_and_layers`，捕获原可用性并集及一次sample；写`test_singleton_variable_cannot_read_temporal_source_updates`，受控改变T对邻变量的状态更新，V-only目标不变。

跨组singleton的核心断言（selected为`[B,C,N]`）：

```python
for layer in out.layers:
    allowed = layer.allowed[out.specialist_selected]  # [selected,G,3]
    assert allowed.any(dim=1).sum(dim=-1).le(1).all()
```

- [x] **Step 3:** 写`test_disabled_group_or_expert_cannot_change_any_downstream_layer`、`test_all_unavailable_returns_original_base_exactly`及fully observed/padding测试；写`test_stack_group_batch_matches_module_reference`比较批量单层与独立组reference。运行文件保存RED。
- [x] **Step 4:** 实现真实控制：共享一次H cache，提前按各层各组独立V参数构建来源缓存，T原来源可用性来自可见掩码/方向band；按各组实际原可用性并集抽样，跨全部层/组应用同一允许类型。按层调用Task2、Task3，每层状态不断梯度。诊断保留组维，公开router.available取最后层并集，weights按门控聚合且注明描述用途；公开evidence按最后层门控汇总，完整组信息在layers中。
- [x] **Step 5:** GREEN并验证八组完整参数独立。保存增量，不能把某组来源缓存给其他组，不增加跨forward缓存。

### Task 5: 模型、激活重计算、配置与两阶段身份

**Files:** Modify `src/malr/models/e23/model.py`, `integrations/e23/config.py`, `train.py`, `run.py`, `scripts/replay_e23_p4_validation.py`, `scripts/evaluate_e23_three_test_draws.py`; Create `src/malr/models/e23/stacked_reference.py`及spec指定三个YAML；Create `tests/integration/test_e23_stacked_model.py`, `test_e23_stacked_training.py`; Extend `test_e23_config.py`。

**Interfaces:** E23Settings与MultiExpertRepairEncoder增加`recovery_layers:int=1,expert_groups:int=1,activation_checkpointing:bool=False`。E23RepairOutput增加`stack_layers:tuple[StackLayerDiagnostics,...]|None=None`。新模式的model.historical为唯一provider，stacked持有层/组；旧模式仍原模块。新增身份`recovery_layers,expert_groups,group_width:64,group_fusion:'gated_concat_v1',activation_checkpointing`连同已有schedule/rounds进入两阶段验证。

`copy_v29_r1_into_stack_reference(source:MultiExpertRepairEncoder,target:MultiExpertRepairEncoder)->None`置于stacked_reference.py，显式复制base/decoder/sharedkey、H消息/V/T/router与整个条件嵌入，只允许目标L1/G1；不能用于正式三模型训练。

- [x] **Step 1:** 写三配置解析与未知键/组合拒绝测试，bool层/组、layers2/groups4、rounds3、旧多头/adapter/mixer/implicit接口拒绝。旧配置仍解析。运行新增用例RED。
- [x] **Step 2:** 写`test_stacked_model_preserves_observed_values_and_ignores_hidden_inputs`、`test_v29_single_group_reference_matches_output_and_gradients`；写`test_checkpointed_layers_match_normal_output_and_gradients_without_extra_h_queries`，重计算use_reentrant=False，层闭包不含H检索/库/EMA副作用。运行新文件RED。
- [x] **Step 3:** 接入新派发和最终一次decode；activation checkpointing仅包裹层内无副作用的Tensor计算，训练需要的当前状态保留梯度，诊断不引入持久缓存。旧默认不新增state_dict键。
- [x] **Step 4:** 写CPU两阶段小训练测试，复用`tests/integration/test_e23_stage1._protocol_settings/_batch`：全局尺度、最终层损失、100轮配置与小测试epochs参数分离、历史更新次数、Stage2冻结key/库；写错层数/组数/融合/重计算身份拒绝，旧多头即使形状类似也不能加载新模式。运行RED后实现构建/身份保存加载。

两阶段记录必须断言：

```python
assert record["recovery_schedule"] == "stacked_parallel_tv_h_v1"
assert record["recovery_layers"] == settings.recovery_layers
assert record["expert_groups"] == settings.expert_groups
assert record["stage1_selection_metric_space"] == "train_standardized"
assert payload["key_sha256"] == record["key_sha256"]
```

- [x] **Step 5:** 新模式正式训练接口只接受spec固定显式输入、无mixer/adapter、Stage2 MAE选模及mse_plus_0p2_mae；正式runner传100轮，CPU测试仍可显式少量epochs用于功能验证。更新评价profile包含`/l{layers}/g{groups}`及全套新身份。
- [x] **Step 6:** 运行新模型/训练/config与旧Stage1/Stage2/iterative/global-objective测试；极小std全局评分、错身份、无重复历史副作用全部通过，保存增量。

### Task 6: 实验矩阵、逐层/组诊断与报告

**Files:** Create `scripts/run_e23_v30_trial.py`, `diagnose_e23_v30_stack.py`, `report_e23_v30_trial.py`; Create `tests/integration/test_e23_v30_reporting.py`。

**Interfaces:** runner CLI `--root PATH --device cuda:1 --mode train|evaluate|report`；train按l1_g8/l4_g1/l4_g8执行两阶段并锁定验证，不自动开始test；evaluate须确认全部三个权重已锁定/重放/code freeze匹配后各执行一次三组测试；report只读取已产出证据。

`choose_v30_candidate(records:Sequence[Mapping],*,reference_mse:float=0.7499365438950035,reference_mae:float=0.5840456761470457)->Mapping|None`仅使用验证MSE严格更低/MAE不更高者，排序MAE→MSE→repair参数数→固定矩阵次序；原v28门槛另记。`validate_v30_diagnostics(record:Mapping,diagnostic:Mapping)->None`校验模型/层/组/config/checkpoint/hash、validation split、train_standardized尺度、H状态不变。

diagnostic CLI `--e23-config PATH --stage1-manifest PATH --stage2-manifest PATH --output PATH --device cuda:1`；各stage按层解码，Stage2各层接同一原预测器记录预测MSE/MAE，汇总组门控/路由/差异，并执行逐组关闭诊断。逐组关闭改变全部数据流，不用来改选checkpoint或正式深度。

- [x] **Step 1:** 写选择测试：严格MSE/MAE边界、tie顺序、fake test指标不影响选择；拒绝非有限/未完成100+100epoch/不匹配SHA记录。写diagnostic错checkpoint、错层/组、变更H SHA拒绝；运行RED。

边界fixture使用已完整构造的合法记录，核心断言：

```python
record["best_val_mse"] = 0.7499365438950035
record["best_val_mae"] = 0.5840456761470457
assert choose_v30_candidate([record]) is None  # MSE必须严格改善
```

- [x] **Step 2:** 写runner dry-run测试，断言三个config/目录顺序、100/100、原mask路径、验证选模参数、全部锁定后才测试、已有完成输出拒绝覆盖；资源绑定各自选定manifest，不复用其他模型资源。实现选择、诊断及runner/report。
- [x] **Step 3:** 报告CSV含v28/T1/v29-r1/v29-r3既有记录及三个新模型；逐组三组CSV含固定seed、窗口/坐标数、MSE/MAE，std为总体位置方差。新模型无证据时相应指标留空，不能填旧版成绩。
- [x] **Step 4:** GREEN报告/脚本CLI测试；逐层最终预测与正常forward一致，诊断的层/组数量准确，不用单组Missing参考计算三组DDR。保存增量；保证所有可运行脚本在代码冻结前已实现。

### Task 7: CUDA预检、旧权重验证、独立审查与冻结

**Files:** Create `scripts/profile_e23_v30.py`, `verify_e23_v30_compatibility.py`; evidence写root/`preflight.json`,`compatibility.json`,`verification.json`,`code-review.json`,`code-freeze.json`。

**Interfaces:** profiler CLI `--configs PATH PATH PATH --output PATH --device cuda:1`，临时训练数据H库、batch128/FP32、每阶段4warmup+10测量step，无optimizer更新；记录forward/backward ms、参数分项、峰值、GPU UUID、H查询数及各组有效性。compatibility CLI `--output PATH --device cuda:1`，读取既有checkpoint，严格加载旧v3/v12/v28/v29 Stage1/2并重放v29-r1及L1/G1参数映射参考。

- [x] **Step 1:** 实现并运行真实CUDA预检三个模型。若某模型batch128 OOM，先清除临时模型，再以activation_checkpointing=true重测该模型；正式配置在训练前固定该设置并进入SHA，不改变batch、precision、epoch。无checkpoint用于正式初始化。
- [x] **Step 2:** 严格旧权重加载；重放v29-r1验证MSE0.7499365438950035/MAE0.5840456761470457至1e-6，L1/G1映射参考同样复现，H key/库前后SHA一致。
- [x] **Step 3:** 运行`python -m pytest -q`，保存完整日志；验证用户CSV SHA及基线外任务改动范围。只针对真实失败/所需门槛修复，不为低影响文档补镜像测试。
- [x] **Step 4:** 按用户确认执行方式调用对应执行技能。采用本会话执行时，executing-plans要求正式训练前一位新鲜独立reviewer审查全部实现；重点检查独立参数、V来源、批量数学、specialist泄漏、重计算副作用、历史时间边界、身份、资源归属。解决Critical/Important并保存结论。
- [x] **Step 5:** 检查本轮增量`git diff --check`，不把既有CRLF警告归成新增失败；保存基线/增量与冻结文件SHA。新文件可独立提交，预先dirty文件先保存本轮patch证据再决定精准提交范围，不能整文件盲提交。

### Task 8: 完成三模型正式训练与共同测试

**Files:** root/`l1_g8`,`l4_g1`,`l4_g8`各自stage1/stage2/logs/resources/validation_replay/diagnostics/test结果；runner与源代码此时已冻结。

- [x] **Step 1:** 确认空闲GPU及所有预检/审查门槛，记录GPU UUID/并发条件；运行以下train命令，持久化完整日志及每阶段资源。长进程用可轮询session，每60秒内给用户有意义的状态更新。

```bash
export PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/run_e23_v30_trial.py --root artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v30_stacked_parallel --device cuda:1 --mode train
```

- [x] **Step 2:** 逐模型核对两个阶段都完成100epoch、自己的Stage1配对权重及库、选模结果、resource文件和代码SHA。每个选定权重重放2688验证窗口/1806336预测坐标，MSE/MAE至1e-6并确认H key/库不变。
- [x] **Step 3:** 保存三个模型的验证选择记录及预先选择结果，另列原v28门槛。按Task6诊断各层/组，先锁定这些结果；不能使用诊断手工截断层或删除组再评估成正式模型。
- [x] **Step 4:** 根据用户“每个跑完之后都用测试组试试”的最新要求，冻结检查后直接调用既有`evaluate_e23_three_test_draws.py`，每模型自己的权重锁定/重放完成后立即在空闲GPU0测试；GPU1继续顺序训练。三模型各评估seed2021/2026100201/2026100202一次，同一选定权重，检查每组2688窗口/1806336坐标和mask SHA。测试顺序变更记录在`per_model_test_authorization.json`，测试与训练并发条件另记；不再次执行矩阵evaluate而覆盖已有测试。
- [x] **Step 5:** 保留全部结果；若训练中发生非有限/系统中断，保存失败阶段和最后状态，只在具体原因解决并保留新证据后恢复，不能偷偷缩短epoch或以临时profiling权重替代。

### Task 9: 生成对照表、归档与最终汇报

**Files:** Create `docs/experiments/E23_v30_stacked_parallel_P4_trial.md`, `ETTh1_P4_E23_v30_T1_comparison_2026-10-03.csv`, `ETTh1_P4_E23_v30_test_groups_2026-10-03.csv`, `ETTh1_P4_E23_v30_layer_diagnostics_2026-10-03.csv`; root/`final-status.json`。

- [x] **Step 1:** 执行runner `--mode report`生成比较与诊断表。每阶段资源/总全流程耗时与选定manifest SHA绑定；显存allocated/reserved、实际参数数、激活重计算、设备和并发条件全部展示。
- [x] **Step 2:** 复核比较表7个模型、test组表三个新模型×三组=9行、诊断层数与组数、测试均值/总体std与原始JSON一致。旧记录保留原不同硬件/精度/epoch口径，不能伪装成此次重新实测。
- [x] **Step 3:** 检查冻结文件、用户CSV、test mask bank及历史状态未变；归档增量实现、测试日志、review、预检与各阶段证据，更新计划checkbox仅限实际完成的步骤。
- [x] **Step 4:** 汇报并行、深度、组合各自的实际MSE/MAE收益与成本，区分验证晋级、测试均值和固定权重诊断；如均无收益，明确保留旧单轮结果。三组缺失位置仅描述同一次训练的缺失位置稳定性。

## Self-Review and Execution Handoff

Spec的参数所有权/候选读取由Tasks1–2实现；融合由Task3；锚点、exposure、关闭数据流由Task4；构建、身份、重计算及两阶段训练由Task5；选择/诊断/报告由Task6；兼容、资源、审查与冻结由Task7；三个100/100实验与共同测试由Task8；最终证据与表格由Task9。五项Review Focus均有明确测试归属。

建议在当前会话由我按任务执行（superpowers:executing-plans），因为历史前端、组维数学、状态控制和加载接口连续依赖，正式训练前保留一次独立代码审查。用户已批准并执行完成Tasks1–9。三个模型各100/100轮、每模型三组测试及证据归档均完成；无新候选晋级，保留v29-r1。
