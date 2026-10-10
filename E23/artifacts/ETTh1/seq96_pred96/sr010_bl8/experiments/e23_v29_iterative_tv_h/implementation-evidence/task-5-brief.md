### Task 5: 共享迭代控制与模型集成

**Files:** Create `src/malr/models/e23/iterative.py`, `exposure.py`; Modify `model.py`; Create/Test `tests/unit/test_e23_iterative.py`, `tests/integration/test_e23_iterative_model.py`。

**Interfaces:** `IterationDiagnostics(round_index:int, latents:Tensor, weights:Tensor, preview_gates:Tensor, allowed:Tensor, update_norm:Tensor)`；`IterativeRepairState(router:RouterOutput, evidence:Mapping[str,ExpertEvidence], iterations:tuple[IterationDiagnostics,...], specialist_selected:Tensor|None)`。

`IterativeTVHController(d_model:int, rounds:int)` 只持有固定三项、零初始化的 round_embedding，不持有重复注册的专家或路由器。`forward(inputs:ExpertInputs, *, historical:HistoricalExpert, variable:SparseVariableExpert, temporal:MultiScaleTemporalExpert, router:ExpertRouter, epoch:int, explore_variable:bool, specialist_exposure:float, specialist_seed:int, disabled_experts:frozenset[str]=frozenset()) -> IterativeRepairState`。

`MultiExpertRepairEncoder` 增加 `recovery_schedule:str='parallel_v1'`、`recovery_rounds:int=1`；E23RepairOutput 增加 `iterations:tuple[IterationDiagnostics,...]|None=None`。正式配置限制宽度64，直接模型及 controller 单元测试可采用较小 D。原 specialist 函数原样移至 exposure.py，model.py 保留原导入名。

- [ ] **Step 1:** 写 `test_iterative_round_uses_previous_written_state`，捕获第二轮 T 读取的 state 等于第一轮输出；`test_t_preview_changes_variable_target_and_source_inputs` 与 `test_v_preview_changes_historical_condition` 捕获真实依赖；`test_disabled_expert_cannot_leak_through_previews` 关闭 T/V 时改变该分支输出不能影响下游或最终结果；`test_specialist_mask_is_sampled_once_and_applies_before_previews` 断言一次抽样、各轮同一允许集合。
- [ ] **Step 2:** 写 `test_iterative_writeback_counts_each_delta_once`：受控 dH/dV/dT，断言最终状态为 zr + weighted deltas，未把 u/v 作为额外写回基准；写 fully observed、C1全无观测、padding、观测硬约束、完整三轮梯度测试。运行 `python -m pytest tests/unit/test_e23_iterative.py tests/integration/test_e23_iterative_model.py -q`，记录控制器缺失导致的失败。
- [ ] **Step 3:** 实现三种缓存建立、单次 specialist 抽样、T预更新→V预更新→H→一次仲裁→下一轮。允许集合来自原有效性，在所有预更新及最终仲裁前应用；disabled_experts 在测试/诊断调用中直接传给控制器。循环状态不断梯度；最终一次 decoder/显式修复。embedding 只影响条件，不进入 writeback base。
- [ ] **Step 4:** 重跑两新测试文件与 `tests/integration/test_e23_model.py`；r1/r3 参数键和形状相同、共享同一参数对象、旧默认模型无新增键；所有专家及路由/decoder 的最终损失梯度有限。
- [ ] **Step 5:** 保存增量和真实状态传递证据；按工作区约束提交 `Add shared iterative temporal variable recovery`。

