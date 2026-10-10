### Task 5: 模型、激活重计算、配置与两阶段身份

**Files:** Modify `src/malr/models/e23/model.py`, `integrations/e23/config.py`, `train.py`, `run.py`, `scripts/replay_e23_p4_validation.py`, `scripts/evaluate_e23_three_test_draws.py`; Create `src/malr/models/e23/stacked_reference.py`及spec指定三个YAML；Create `tests/integration/test_e23_stacked_model.py`, `test_e23_stacked_training.py`; Extend `test_e23_config.py`。

**Interfaces:** E23Settings与MultiExpertRepairEncoder增加`recovery_layers:int=1,expert_groups:int=1,activation_checkpointing:bool=False`。E23RepairOutput增加`stack_layers:tuple[StackLayerDiagnostics,...]|None=None`。新模式的model.historical为唯一provider，stacked持有层/组；旧模式仍原模块。新增身份`recovery_layers,expert_groups,group_width:64,group_fusion:'gated_concat_v1',activation_checkpointing`连同已有schedule/rounds进入两阶段验证。

`copy_v29_r1_into_stack_reference(source:MultiExpertRepairEncoder,target:MultiExpertRepairEncoder)->None`置于stacked_reference.py，显式复制base/decoder/sharedkey、H消息/V/T/router与整个条件嵌入，只允许目标L1/G1；不能用于正式三模型训练。

- [ ] **Step 1:** 写三配置解析与未知键/组合拒绝测试，bool层/组、layers2/groups4、rounds3、旧多头/adapter/mixer/implicit接口拒绝。旧配置仍解析。运行新增用例RED。
- [ ] **Step 2:** 写`test_stacked_model_preserves_observed_values_and_ignores_hidden_inputs`、`test_v29_single_group_reference_matches_output_and_gradients`；写`test_checkpointed_layers_match_normal_output_and_gradients_without_extra_h_queries`，重计算use_reentrant=False，层闭包不含H检索/库/EMA副作用。运行新文件RED。
- [ ] **Step 3:** 接入新派发和最终一次decode；activation checkpointing仅包裹层内无副作用的Tensor计算，训练需要的当前状态保留梯度，诊断不引入持久缓存。旧默认不新增state_dict键。
- [ ] **Step 4:** 写CPU两阶段小训练测试，复用`tests/integration/test_e23_stage1._protocol_settings/_batch`：全局尺度、最终层损失、100轮配置与小测试epochs参数分离、历史更新次数、Stage2冻结key/库；写错层数/组数/融合/重计算身份拒绝，旧多头即使形状类似也不能加载新模式。运行RED后实现构建/身份保存加载。

两阶段记录必须断言：

```python
assert record["recovery_schedule"] == "stacked_parallel_tv_h_v1"
assert record["recovery_layers"] == settings.recovery_layers
assert record["expert_groups"] == settings.expert_groups
assert record["stage1_selection_metric_space"] == "train_standardized"
assert payload["key_sha256"] == record["key_sha256"]
```

- [ ] **Step 5:** 新模式正式训练接口只接受spec固定显式输入、无mixer/adapter、Stage2 MAE选模及mse_plus_0p2_mae；正式runner传100轮，CPU测试仍可显式少量epochs用于功能验证。更新评价profile包含`/l{layers}/g{groups}`及全套新身份。
- [ ] **Step 6:** 运行新模型/训练/config与旧Stage1/Stage2/iterative/global-objective测试；极小std全局评分、错身份、无重复历史副作用全部通过，保存增量。

