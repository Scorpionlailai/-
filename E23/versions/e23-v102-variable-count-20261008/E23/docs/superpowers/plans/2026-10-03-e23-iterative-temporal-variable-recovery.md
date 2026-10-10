# E23 v29 Iterative T/V Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实现共享参数的 T/V 交替修复，并用从头训练的一轮、三轮实验判断状态传递和迭代能否提高预测准确性。

**Architecture:** 每轮 T 预更新供 V 读取，V 预更新供 H 读取，再以该轮起始状态为基准做一次 none/H/V/T 仲裁写回。H 候选、T 原可见来源特征和 V 离散来源索引在当前 forward 内缓存，轮间重新形成消息；最终显式修复及原预测头只执行一次。

**Tech Stack:** Python 3、PyTorch 2.7.1、pytest、现有 E23/P4 数据与训练接口；不增加第三方依赖。

**Spec:** `docs/superpowers/specs/2026-10-03-e23-iterative-temporal-variable-recovery-design.md`，用户于 2026-10-03 确认。

## Global Constraints

- 正式配置为 ETTh1 96→96、7 变量、patch16/stride8、12 个 patch、修复宽度64、原预测头宽度16、explicit_v1。
- 保留 P4 start_rate0.10/block_len8/overlapTrue，动态训练位置、原固定验证位置；测试三组 seed 为 2021、2026100201、2026100202。
- H top-k4/threshold0.7，V window_context_topk_v1/top-k4/offsets(-1,0,1)，T 原方向距离带(2,4,8,16)，初始 residual_scale0.1。
- 两阶段各100轮，batch128，seed2021，Adam lr0.001/0.0001，weight_decay0。
- Stage1 使用最终一轮 hidden-valid 全局 MSE + 0.1 retrieval loss，按全局验证修复 MSE 选模；specialist_exposure0.2。
- Stage2 使用 MSE + 0.2 MAE、固定验证 MAE 选模；不启用 specialist exposure、不加修复辅助损失。
- 新模式 `iterative_tv_h_v1` 仅支持 rounds1或3；旧默认为 `parallel_v1`/1，不给旧模型增加参数或身份字段。
- 完全观察 patch 保持 base；掩码、validity、rho、query_mean/std 固定；最终观测值精确保留，padding 排除。
- 先完整实现和验证，再顺序训练 r1、r3；现有 v28/T1 记录不重跑。新模型验证 MAE <0.5855096656989619 且同一权重 MSE ≤0.7402293330687785 才晋级。
- 通过者按验证 MAE、MSE、较少轮数选择一个，再测三组；不得根据测试重选。都未通过则保留 v28，不产生新测试。
- 报告全流程及每阶段耗时、训练阶段耗时、峰值 allocated/reserved GiB、GPU UUID/并发条件；600秒仅是希望达到的预算。
- 工作目录 `/home/xyw/paper/E23`。执行命令使用 `PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH PYTHONPATH=src:.`。
- 当前目录含前序未提交实现；执行前保存改动起点。禁止整体暂存、回退或覆盖无关文件，尤其 `docs/experiments/E22_all_datasets_missing_scenarios.csv`。不能隔离提交的已有文件保留工作区和本轮增量记录，不强行提交。

## Review Focus

| 输入或条件 | 必须表现 | 所属任务 |
| --- | --- | --- |
| 同一进程换 mask、batch 或原型 snapshot | 缓存随新 forward 重建，不复用旧候选 | 2、5 |
| fully missing 目标、空历史库或 C=1 | 合法专家仍可工作；全无证据时精确回退且有限 | 2、4、5 |
| 专家关闭及 specialist 单专家暴露 | 被关闭专家不能经预更新影响后续消息 | 5 |
| 末尾复制填充和极小 query_std | padding 不进入输出/梯度/指标；全局评分不恢复逆方差放大 | 3、5、6 |
| 相同参数形状但不同轮数的 checkpoint | 身份检查拒绝混用；验证和测试不改变 H 库/key | 6、8 |

## Files and Boundaries

- `router.py`：形成专家修正、预门控及最终仲裁；继续保留旧批量打分行为。
- `historical.py`、`temporal.py`、`variable.py`：各自定义内部 cache dataclass 和可选上下文消息接口；旧路径保留。
- 新建 `iterative.py`：控制顺序与循环、轮次条件和诊断；不重复注册父模型持有的 H/V/T/router。
- 新建 `exposure.py`：迁移原 specialist 抽样函数，`model.py` 继续导入并暴露原函数，避免循环导入。
- `model.py`：按 schedule 派发，复用最终 decoder 和显式修复；E23RepairOutput 增加默认 None 的逐轮诊断字段。
- `integrations/e23/config.py`、`train.py`：配置校验、构建及新模式身份校验。
- 新建 `scripts/profile_e23_v29.py`、`scripts/report_e23_v29_trial.py`；扩展现有验证重放和三组评价元数据。
- 新测试集中在真实数据依赖、缓存、梯度、观测约束和身份；不为低影响文档或字段复制单独写测试。

---

### Task 1: 可复用的专家修正、预门控与最终仲裁

**Files:** Modify `src/malr/models/e23/router.py`; Test `tests/unit/test_e23_router.py`。

**Interfaces:** 在 `ExpertRouter` 增加：

- `expert_delta(inputs: ExpertInputs, name: str, item: ExpertEvidence) -> Tensor`
- `expert_logit(inputs: ExpertInputs, name: str, item: ExpertEvidence, delta: Tensor, *, round_embedding: Tensor | None = None) -> Tensor`
- `none_logit(inputs: ExpertInputs, *, round_embedding: Tensor | None = None) -> Tensor`
- `preview_gate(inputs: ExpertInputs, name: str, item: ExpertEvidence, delta: Tensor, *, allowed: Tensor, round_embedding: Tensor | None = None) -> Tensor`
- `route_precomputed(inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence], deltas: Mapping[str, Tensor], *, round_embedding: Tensor | None = None) -> RouterOutput`

embedding 形状为 [D]，只进入打分条件，不能改变返回状态的 base。预门控为 `sigmoid(expert_logit-none_logit)*allowed*damaged*item.available`。

- [x] **Step 1:** 写 `test_precomputed_router_matches_legacy_output_and_gradients`：固定 seed42，沿用 `_inputs/_evidence`，比较显式计算修正后的仲裁与旧 forward 的 latents、weights、梯度；写 `test_preview_gate_is_zero_when_disallowed_or_undamaged`，精确断言零；保留原 `test_router_scores_experts_in_one_batched_call`。
- [x] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_router.py -q`，确认新方法缺失导致新测试失败，记录失败原因。
- [x] **Step 3:** 按接口拆分路由器计算。旧 forward 继续一次批量调用 shared_score；预计算修正不再在最终仲裁重算，检查专家名称、形状和有限性。
- [x] **Step 4:** 重跑同一文件；全部通过，原 no-evidence/base 和批量调用断言保持通过。
- [x] **Step 5:** 保存本任务增量与测试结果；仅对已核对的本轮变更提交 `Refactor expert arbitration for iterative recovery`。

### Task 2: H 候选缓存与逐轮历史消息

**Files:** Modify `src/malr/models/e23/historical.py`; Create/Test `tests/unit/test_e23_historical_cache.py`。

**Interfaces:** `HistoricalCandidateCache` 包含 `normalized_raw:[B,C,N,K,P]`、`raw_quality:[B,C,N,K,5]`、`base_logits:[B,C,N,K]`、`valid`、`available`、`provenance`。增加构造参数 `contextual: bool=False`；增加 `prepare_candidates(inputs: ExpertInputs) -> HistoricalCandidateCache` 和 `from_cache(inputs: ExpertInputs, cache: HistoricalCandidateCache, *, round_embedding: Tensor | None=None) -> ExpertEvidence`。

新模式才实例化状态匹配投影。匹配项为 `0.5*tanh(dot(q(state),k(candidate))/sqrt(D))`，q 输出零初始化；消息沿用原 message_projector，candidate validity 和原可见质量始终保留。

- [x] **Step 1:** 用现有 `test_e23_historical._inputs/_memory` 写 `test_h_cache_queries_once_and_messages_depend_on_current_state`：真实索引 query spy 计数为1，三次 from_cache 不再查询，改变 state 后 message 改变；写 `test_h_cache_rebuilds_after_mask_or_snapshot_change`、`test_h_cache_empty_and_unobserved_queries_stay_unavailable`；验证原未来成员 starts12 被排除。
- [x] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_historical_cache.py -q`，确认缓存接口不存在导致失败。
- [x] **Step 3:** 实现 cache/from_cache。空库或无可检索目标返回固定 K 槽且全无效的缓存；缓存保留训练张量，诊断 provenance 不作为训练输入。旧 forward 留原路径，旧构造不生成新 state_dict 键。
- [x] **Step 4:** 运行 `python -m pytest tests/unit/test_e23_historical.py tests/unit/test_e23_historical_cache.py -q`；候选权重有效槽和为1，无效槽为0，状态匹配不复活无效候选，message_projector 梯度有限且非零。
- [x] **Step 5:** 保存增量与通过证据；按工作区约束提交 `Cache historical candidates within iterative forwards`。

### Task 3: T 的可见来源缓存与状态条件消息

**Files:** Modify `src/malr/models/e23/temporal.py`; Create/Test `tests/unit/test_e23_temporal_context.py`。

**Interfaces:** `TemporalSourceCache` 持有 `source_encoded:[B,C,N,D]` 和原可见/有效来源张量。新增构造参数 `contextual: bool=False`；`prepare_sources(inputs: ExpertInputs) -> TemporalSourceCache`；`from_cache(inputs: ExpertInputs, cache: TemporalSourceCache, *, round_embedding: Tensor | None=None) -> ExpertEvidence`。

来源特征为 `source_encoded + source_context_projection(LayerNorm(current_source_state))`，投影输出零初始化；查询条件使用当前 state 与 round_embedding。方向带、来源可见性及一致性仍来自原可见数据。

- [x] **Step 1:** 沿用 `tests/unit/test_e23_temporal.py` 的样例几何，写 `test_temporal_context_reads_updated_source_states_without_reencoding`：受控设置上下文投影为非零，改变原有效来源的 state 后消息改变，三次 from_cache 的原 source_encoder 调用数仍为1；写 `test_temporal_context_ignores_hidden_and_invalid_raw_values`，改变隐藏/无效槽有限数值不改变结果；断言无效槽输入梯度为零。
- [x] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_temporal_context.py -q`，确认新接口缺失导致失败。
- [x] **Step 3:** 实现 cache 和新消息路径，距离带不变、全局候选允许随 state 改变；保留旧 forward 的一次原来源编码行为。
- [x] **Step 4:** 运行 `python -m pytest tests/unit/test_e23_temporal.py tests/unit/test_e23_temporal_context.py -q`；来源可见性、无证据回退、缓存计数及状态投影梯度通过。
- [x] **Step 5:** 保存增量与证据；按工作区约束提交 `Condition temporal evidence on evolving repair states`。

### Task 4: V 读取 T 更新后的目标及来源状态

**Files:** Modify `src/malr/models/e23/variable.py`; Create/Test `tests/unit/test_e23_variable_context.py`。

**Interfaces:** `VariableSourceCache` 包含原选定四来源及 offsets、source_values/source_observed、quality、prior/logits、candidate_valid、索引 provenance。新增构造参数 `contextual: bool=False`；`prepare_sources(inputs: ExpertInputs, *, epoch:int, explore:bool) -> VariableSourceCache`；`from_cache(inputs: ExpertInputs, cache: VariableSourceCache, *, round_embedding: Tensor | None=None) -> ExpertEvidence`。

候选消息是原 MLP 消息加原选定来源的当前 state 投影；候选分数加 [-0.5,0.5] 的目标/来源 state 匹配项。来源投影和匹配初始影响为零。上下文模式仅走 scalar_v1/window_context_topk_v1，旧 v26/v27 保留。

- [x] **Step 1:** 写 `test_variable_context_reads_both_updated_target_and_source_states`，分别扰动目标、已选来源，受控启用新投影后两者均改变消息；写 `test_variable_context_keeps_source_ids_and_original_validity`，跨轮 source ids 相同、候选不超过4×3、原不可见来源不复活；写 `test_variable_context_single_channel_has_zero_available_evidence`。
- [x] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_variable_context.py -q`，确认缺接口导致失败。
- [x] **Step 3:** 实现来源缓存与重新形成消息，候选匹配 q 输出零初始化，避免每轮重做离散来源选择或训练探索轮换；质量只以原观测及本轮权重汇总。
- [x] **Step 4:** 运行 `python -m pytest tests/unit/test_e23_variable.py tests/unit/test_e23_variable_context.py -q`，新梯度、权重及既有来源选择/聚合测试通过。
- [x] **Step 5:** 保存增量与证据；按工作区约束提交 `Pass temporal repair context into variable evidence`。

### Task 5: 共享迭代控制与模型集成

**Files:** Create `src/malr/models/e23/iterative.py`, `exposure.py`; Modify `model.py`; Create/Test `tests/unit/test_e23_iterative.py`, `tests/integration/test_e23_iterative_model.py`。

**Interfaces:** `IterationDiagnostics(round_index:int, latents:Tensor, weights:Tensor, preview_gates:Tensor, allowed:Tensor, update_norm:Tensor)`；`IterativeRepairState(router:RouterOutput, evidence:Mapping[str,ExpertEvidence], iterations:tuple[IterationDiagnostics,...], specialist_selected:Tensor|None)`。

`IterativeTVHController(d_model:int, rounds:int)` 只持有固定三项、零初始化的 round_embedding，不持有重复注册的专家或路由器。`forward(inputs:ExpertInputs, *, historical:HistoricalExpert, variable:SparseVariableExpert, temporal:MultiScaleTemporalExpert, router:ExpertRouter, epoch:int, explore_variable:bool, specialist_exposure:float, specialist_seed:int, disabled_experts:frozenset[str]=frozenset()) -> IterativeRepairState`。

`MultiExpertRepairEncoder` 增加 `recovery_schedule:str='parallel_v1'`、`recovery_rounds:int=1`；E23RepairOutput 增加 `iterations:tuple[IterationDiagnostics,...]|None=None`。正式配置限制宽度64，直接模型及 controller 单元测试可采用较小 D。原 specialist 函数原样移至 exposure.py，model.py 保留原导入名。

- [x] **Step 1:** 写 `test_iterative_round_uses_previous_written_state`，捕获第二轮 T 读取的 state 等于第一轮输出；`test_t_preview_changes_variable_target_and_source_inputs` 与 `test_v_preview_changes_historical_condition` 捕获真实依赖；`test_disabled_expert_cannot_leak_through_previews` 关闭 T/V 时改变该分支输出不能影响下游或最终结果；`test_specialist_mask_is_sampled_once_and_applies_before_previews` 断言一次抽样、各轮同一允许集合。
- [x] **Step 2:** 写 `test_iterative_writeback_counts_each_delta_once`：受控 dH/dV/dT，断言最终状态为 zr + weighted deltas，未把 u/v 作为额外写回基准；写 fully observed、C1全无观测、padding、观测硬约束、完整三轮梯度测试。运行 `python -m pytest tests/unit/test_e23_iterative.py tests/integration/test_e23_iterative_model.py -q`，记录控制器缺失导致的失败。
- [x] **Step 3:** 实现三种缓存建立、单次 specialist 抽样、T预更新→V预更新→H→一次仲裁→下一轮。允许集合来自原有效性，在所有预更新及最终仲裁前应用；disabled_experts 在测试/诊断调用中直接传给控制器。循环状态不断梯度；最终一次 decoder/显式修复。embedding 只影响条件，不进入 writeback base。
- [x] **Step 4:** 重跑两新测试文件与 `tests/integration/test_e23_model.py`；r1/r3 参数键和形状相同、共享同一参数对象、旧默认模型无新增键；所有专家及路由/decoder 的最终损失梯度有限。
- [x] **Step 5:** 保存增量和真实状态传递证据；按工作区约束提交 `Add shared iterative temporal variable recovery`。

### Task 6: 配置、两阶段身份及评价元数据

**Files:** Modify `integrations/e23/config.py`, `train.py`, `scripts/replay_e23_p4_validation.py`, `scripts/evaluate_e23_three_test_draws.py`; Create 两个 spec 指定 YAML；Create/Test `tests/integration/test_e23_iterative_training.py`; Extend `tests/integration/test_e23_config.py`。

**Interfaces:** E23Settings 增加 `recovery_schedule:str='parallel_v1'`、`recovery_rounds:int=1`。`_trial_identity` 仅给非默认模式增加两字段，Stage1/2 保存和加载完整传播。`build_repair_model` 传入 schedule/rounds。评价脚本新增结果元数据两字段，不能改变 mask 或评分。

- [x] **Step 1:** 写新配置 r1/r3解析测试，严格拒绝 bool轮数、rounds2、parallel3、旧多头/后聚合/adapter组合、window_smooth_l1搭配新模式；保留旧配置解析。运行新增用例确认新键被旧校验拒绝。
- [x] **Step 2:** 写 `test_iterative_two_stage_training_preserves_schedule_and_global_selection`，沿用 `tests/integration/test_e23_stage1._protocol_settings/_batch`，以 monkeypatch batch 做 Stage1两轮/Stage2一轮；断言正式尺度、最终状态评分、manifest/checkpoint两字段一致；写 `test_identical_shape_checkpoint_with_wrong_rounds_is_rejected`，改变身份字段但保持权重形状仍拒绝。
- [x] **Step 3:** 实现受限配置、构建和身份传播，保持 Stage1 global_mse_v1 与 Stage2 mse_plus_0p2_mae 不变。history_adapter 非 none 搭配迭代明确拒绝。三组结果 profile 显式包含 schedule/rounds，新标签使用 E23_v29_r1 或 E23_v29_r3。
- [x] **Step 4:** 运行 `python -m pytest tests/integration/test_e23_config.py tests/integration/test_e23_global_repair_objective.py tests/integration/test_e23_iterative_training.py tests/integration/test_e23_stage1.py tests/integration/test_e23_stage2.py -q`；wrong-round身份、两阶段训练、极小标准差的全局评分全部通过。
- [x] **Step 5:** 保存增量与证据；按工作区约束提交 `Record iterative recovery identity across training and evaluation`。

### Task 7: CUDA 成本预检、旧权重兼容及代码审查

**Files:** Create `scripts/profile_e23_v29.py`; 检查 Tasks1–6 全部文件；结果写入新实验目录下 `preflight.json`、`code-review.json`、`verification.json`。

**Interfaces:** profiler CLI 为 `--config-one PATH --config-three PATH --output PATH --device cuda:1`，只做 throwaway profiling，不产出选定 checkpoint。输出每模型 repair参数数、forward/backward毫秒、峰值allocated/reserved GiB、候选可用性、GPU UUID；记录同批、同精度和 warmup 次数。

- [x] **Step 1:** 保存当前起点及冻结前文件哈希。profile模型分别从新配置构建，使用训练数据发布的临时 H bank；batch128、float32、4个 warmup、10个测量 step，CUDA同步后计时，不更新优化器或复用临时模型做正式训练。
- [x] **Step 2:** 运行 profiler，在新目录保存实际时间/显存；三轮 forward/backward 有限，H/T缓存计数符合 spec。若成本高于预期仍记录实际值，不擅自缩小 batch、轮数或训练 epoch。
- [x] **Step 3:** 用实际保存的 v3/v12/v28 config+manifest 严格加载旧 Stage1/2；对 v28 原固定验证重放，复现 0.7402293330687785 / 0.5855096656989619 至1e-6并保存 H key、bank前后哈希。现有 `scripts/replay_e23_p4_validation.py` 的旧路径检查保留。
- [x] **Step 4:** 运行 `python -m pytest -q`，保存完整日志；只修复真实失败或必需门槛。独立代码审查重点为梯度断开、预更新泄漏、重复写回、缓存跨forward、参数重复注册、新模式身份和旧默认行为；记录审查结论并解决 Critical/Important。
- [x] **Step 5:** 执行 `git diff --check`，保存实现冻结哈希与已验证状态；按工作区约束保存/提交本轮变更，随后开始正式训练。

### Task 8: r1/r3 完整训练、验证选模及最终报告

**Files:** Create `scripts/report_e23_v29_trial.py`; Test `tests/integration/test_e23_v29_selection.py`; Create `docs/experiments/E23_v29_iterative_TV_H_P4_trial.md` 和 `ETTh1_P4_E23_v28_v29_T1_comparison_2026-10-03.csv`；可晋级时 Create 相应逐组结果 CSV。

**Interfaces:** `choose_v29_candidate(records:Sequence[Mapping[str,object]], *, reference_mse:float=0.7402293330687785, reference_mae:float=0.5855096656989619) -> Mapping[str,object]|None`，只消费已经验证重放的 Stage2 记录，拒绝非有限指标、无身份或未完成100/100轮记录。返回通过双门槛者中按 MAE/MSE/rounds 排序的候选。

- [x] **Step 1:** 写 `test_v29_selection_uses_only_validation_and_both_gates`、`test_v29_selection_ties_prefer_fewer_rounds`、`test_v29_selection_rejects_incomplete_or_nonfinite_records`，在记录中放入与排序相反的假 test 指标仍不影响选择；运行确认选择函数缺失导致失败，随后实现选择/报告并使三测试通过。
- [x] **Step 2:** 检查两个新输出根目录尚无完成记录；冻结当前代码与两配置。顺序执行 r1 Stage1→Stage2、r3 Stage1→Stage2，每步保存日志/manifest/resource。每个模型必须独立从头训练100/100轮，不能复用 preflight 或另一个模型的权重。

训练命令采用以下模板，分别把 rounds替换为1、3后顺序执行；`E23_V29_CONFIG/E23_V29_ROOT` 为任务变量，不能改系统 HOME：

```bash
export PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
E23_V29_CONFIG=configs/e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml
E23_V29_ROOT=artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v29_iterative_tv_h_r1
python -m integrations.e23.run train-repair --protocol-config configs/stage1_seq96_dynamic_temporal_repair.yaml --e23-config "$E23_V29_CONFIG" --migration-catalog ../E22/migration/paper_catalog.jsonl --mask-artifacts ../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context --output "$E23_V29_ROOT/stage1" --device cuda:1
python -m integrations.e23.run train-forecast --protocol-config configs/stage1_seq96_dynamic_temporal_repair.yaml --e23-config "$E23_V29_CONFIG" --migration-catalog ../E22/migration/paper_catalog.jsonl --mask-artifacts ../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context --stage1-manifest "$E23_V29_ROOT/stage1/runs/e23-stage1.json" --output "$E23_V29_ROOT/stage2" --device cuda:1 --selection-metric mae --forecast-loss mse_plus_0p2_mae --forecast-input explicit_v1
python scripts/replay_e23_p4_validation.py --e23-config "$E23_V29_CONFIG" --stage2-manifest "$E23_V29_ROOT/stage2/runs/e23-stage2.json" --output "$E23_V29_ROOT/validation_replay.json" --device cuda:1
```

- [x] **Step 3:** 每个选定权重重放全部2688验证窗口、1806336预测坐标；比较 manifest 至1e-6，核对 H key/bank未改变。以选定 Stage1/2 权重记录各轮全局修复 MSE/MAE、状态更新幅度、预门控和最终路由统计。冻结代码检查未变；逐轮诊断不用于额外选模。
- [x] **Step 4:** 调用 choose_v29_candidate，先写 `validation_selection.json`，其中保留两模型验证指标、checkpoint SHA、门槛判定和唯一候选。没有候选时直接进入最终报告，不执行新测试。
- [x] **Step 5:** 若有候选，仅对该模型调用既有三组评价脚本一次；把下例的 config/root/model-label 改为预先选定者，保持现有 test mask bank 不变。

```bash
python scripts/evaluate_e23_three_test_draws.py --e23-config "$E23_V29_CONFIG" --selection "$E23_V29_ROOT/stage2/runs/e23-stage2.json" --test-mask-bank artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json --model-label E23_v29_r1 --output "$E23_V29_ROOT/three-test-draws.json" --device cuda:1
```

- [x] **Step 6:** 生成 Markdown/CSV：包含 v28、T1既有结果及r1/r3验证、修复、训练时间/显存。只有被选中并实际测试的版本填写三组测试均值/标准差；未测试者空值并明确标记。资源绑定实际selection SHA，最终测试耗时不算训练，不用单组 Missing 参考计算三组 DDR。
- [x] **Step 7:** 复核 epoch数、两个实验目录、配置/code/hash、资源、测试seed/window/count及CSV行数；保存最终 status。最终汇报区分验证选择、测试表现和固定权重诊断，明确是否超过T1，以及一轮与三轮的实际成本。

## Execution Handoff

推荐在当前会话按任务顺序实施：八项任务共享专家、路由和加载接口，连续实现能减少交接成本，完整训练前保留独立审查。

计划自查项：八个任务覆盖 spec 的缓存、串行依赖、共享迭代、约束、身份、兼容、资源和评价；上述五类 Review Focus 均有明确归属与测试；任务间 cache/返回类型及配置名保持一致。

2026-10-03 用户已确认执行；Tasks1–7完成，代码审查、旧权重兼容、CUDA预检及856项测试通过。Task8两版各100/100轮已完成。原验证双门槛未晋级；用户随后要求额外测试，两版三组测试及r3验证前缀诊断也已完成。证据见新实验目录final-status.json、报告和归档implementation-evidence/progress.md。
