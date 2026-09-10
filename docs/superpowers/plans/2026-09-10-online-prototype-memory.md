# 在线原型记忆基础 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏旧离线原型库的前提下，建立可从空库启动、按 epoch 发布、严格去重与因果约束的在线分组式原型记忆基础。

**Architecture:** 保留 `src/malr/prototypes/bank.py` 作为旧实验的离线库；新增独立的 `online_bank.py`，把当前冻结快照、epoch 内待提交缓冲区和永久 `seen` 身份集合分开管理。在线库只接收当前人工 mask 后完全可见的训练 patch，发布时按 `(variable_id, absolute_start)` 排序，依据当前成员重算组中心并执行 FIFO 容量策略；查询只读已发布快照，后续检索编码器和模型训练通过明确的快照接口接入。

**Tech Stack:** Python 3.10+, NumPy, PyYAML, pytest；不增加第三方依赖。

**Spec:** `docs/superpowers/specs/2026-09-10-seq96在线原型与关系感知恢复交接.md`

## Global Constraints

- 第一轮只服务 ETTh1、`seq_len=96`、`pred_len=96`、patch `16/8`、seed `2021` 的后续实验；不启动长窗口、Weather 或 S4M。
- 不覆盖或清理当前工作树中的既有源码、测试和实验产物；旧离线 `build-bank` 流程必须继续可加载。
- 训练 patch 只有在当前人工 mask 后状态为完全可见时才能进入待提交缓冲区；被遮挡 patch 的真实值只能作为训练监督，不能写入本轮记忆。
- 每个身份由 `(variable_id, absolute_start)` 唯一确定；身份一旦发布进入 `seen`，即使以后被淘汰也不得重新插入。
- 第 `e` 个 epoch 只能读取第 `e-1` 个 epoch 发布的冻结快照；当前 epoch 的待提交 patch 不得被当前 epoch 查询。
- 历史候选必须满足 `source_start + patch_len <= query_start`；库发布和哈希计算必须与输入顺序无关。
- 验证和测试只读，不得改变成员、待提交缓冲区、`seen`、组中心或任何检索状态。
- 所有生产代码遵循 TDD：每个新增行为先写一个会按预期失败的测试，再写最小实现，再运行局部和全量测试。

---

### Task 1: 在线记忆配置字段与 seq_len=96 配置骨架

**Files:**
- Modify: `src/malr/config.py:53-74, 387-410`，为 `PrototypeConfig` 增加在线存储模式与“仅收录完全可见 patch”字段，并保持旧配置默认值和哈希兼容。
- Create: `configs/stage1_seq96_online_memory.yaml`，提供 ETTh1 的 `96→96` 后续实验基线配置；该文件只声明在线记忆参数，关系路由尚由后续任务接入。
- Test: `tests/unit/test_config.py`，新增配置加载、默认兼容和非法在线模式校验。

**Interfaces:**
- Produces `PrototypeConfig.storage_mode: str`，允许值为 `offline_grouped` 和 `online_grouped`，默认值为 `offline_grouped`。
- Produces `PrototypeConfig.online_observed_only: bool`，默认值为 `True`；在线模式必须为 `True`，防止配置绕过已确认的数据口径。
- `load_config(Path("configs/stage1_seq96_online_memory.yaml"))` 返回已通过 `Stage1Config.validate()` 的配置，且 `data.seq_len == 96`、`data.pred_len == 96`、`patch.length == 16`、`patch.stride == 8`。

- [ ] **Step 1: Write the failing tests**

```python
def test_online_memory_config_loads_with_explicit_observed_only() -> None:
    config = load_config(Path("configs/stage1_seq96_online_memory.yaml"))
    assert config.prototype.storage_mode == "online_grouped"
    assert config.prototype.online_observed_only is True
    assert config.data.seq_len == 96
    assert config.data.pred_len == 96


def test_online_memory_mode_rejects_masked_patch_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        Path("configs/stage1_seq96_online_memory.yaml")
        .read_text(encoding="utf-8")
        .replace("online_observed_only: true", "online_observed_only: false"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="online_observed_only"):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/unit/test_config.py -k 'online_memory' -v `

Expected: FAIL because `PrototypeConfig` does not yet accept `storage_mode` or `online_observed_only`, and the new YAML does not exist.

- [ ] **Step 3: Write the minimal implementation**

Add the two defaulted dataclass fields, parse them from `raw["prototype"]`, and validate the two allowed modes plus the online invariant. Add the YAML by copying only the existing ETTh1 geometry, normalization, model and training values needed by the loader, setting `storage_mode: online_grouped`, `online_observed_only: true`, and retaining `relations.selection: hard_topk` until the later relation task replaces that path.

- [ ] **Step 4: Run focused and compatibility tests**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/unit/test_config.py -k 'online_memory or legacy' -v `

Expected: PASS, including existing legacy configuration hash tests.

- [ ] **Step 5: Commit**

```text
git add src/malr/config.py tests/unit/test_config.py configs/stage1_seq96_online_memory.yaml
git commit -m "feat: add online memory configuration skeleton"
```

### Task 2: 空库、待提交缓冲区与 epoch 发布状态机

**Files:**
- Create: `src/malr/prototypes/online_bank.py`，实现在线原型实体、分组、冻结快照和可变状态机。
- Modify: `src/malr/prototypes/__init__.py`，导出在线库公共类型。
- Test: `tests/unit/test_online_bank.py`，覆盖空库启动、完全可见门控、epoch 内不可读、排序去重、严格边界和容量行为。

**Interfaces:**
- `PrototypeIdentity = tuple[int, int]`，内容为 `(variable_id, absolute_start)`。
- `OnlinePrototypeMember(variable_id: int, start: int, raw_patch: NDArray, clean_key: NDArray)`：保存有限的一维完整 patch 和单位化检索 key。
- `OnlinePrototypeGroup(variable_id: int, group_id: int, members: tuple[OnlinePrototypeMember, ...])`：组中心由当前 `members` 的 key 归一化均值计算，不保存累计淘汰成员的贡献。
- `OnlinePrototypeSnapshot(epoch: int, groups: tuple[tuple[OnlinePrototypeGroup, ...], ...], seen: frozenset[PrototypeIdentity], diagnostics: Mapping[str, int | float])`：不可变查询快照。
- `OnlinePrototypeMemory(..., patch_len: int, patch_stride: int, train_end: int, variable_count: int, group_similarity_threshold: float = 0.8, group_capacity: int = 16, max_groups_per_variable: int = 128, observed_only: bool = True)`：从空库构造。
- `stage_patch(variable_id, absolute_start, raw_patch, clean_key, *, observed: bool) -> bool`：只有 `observed=True` 且在训练边界内的 patch 才进入当前 pending；同一身份在 pending 或 seen 中重复提交返回 `False`。
- `publish_epoch(epoch: int) -> OnlinePrototypeSnapshot`：按 `(variable_id, absolute_start)` 排序处理 pending，发布新冻结快照并清空 pending；返回值与输入顺序无关。
- `snapshot() -> OnlinePrototypeSnapshot`：返回当前已发布快照；初始快照 epoch 为 `0`，每个变量无组。
- `query(variable_id, query_start, query_key, *, max_items: int = 4) -> tuple[OnlinePrototypeMember, ...]`：只读当前快照，过滤 `member.start + patch_len <= query_start`，按 `query_key` 与组中心相似度选组后在组内按 key 相似度稳定排序；不改变任何状态。

- [ ] **Step 1: Write the failing tests**

```python
def make_memory(**kwargs: object) -> OnlinePrototypeMemory:
    return OnlinePrototypeMemory(
        patch_len=16,
        patch_stride=8,
        train_end=128,
        variable_count=1,
        **kwargs,
    )


def make_member(
    variable_id: int,
    start: int,
    *,
    clean_key: list[float] | None = None,
) -> OnlinePrototypeMember:
    key = np.asarray(clean_key or [1.0, 0.0], dtype=np.float64)
    raw_patch = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    return OnlinePrototypeMember(variable_id, start, raw_patch, key)


def test_epoch_one_starts_empty_and_pending_patch_is_not_readable() -> None:
    memory = make_memory()
    member = make_member(variable_id=0, start=0)
    assert memory.snapshot().epoch == 0
    assert memory.query(0, 32, [1.0, 0.0]) == ()
    assert memory.stage_patch(0, 0, member.raw_patch, member.clean_key, observed=True)
    assert memory.query(0, 32, [1.0, 0.0]) == ()
    published = memory.publish_epoch(1)
    assert published.epoch == 1
    assert [item.start for item in memory.query(0, 32, [1.0, 0.0])] == [0]


def test_masked_patch_is_not_staged_and_duplicate_identity_is_permanent() -> None:
    memory = make_memory()
    member = make_member(variable_id=0, start=8)
    assert not memory.stage_patch(0, 8, member.raw_patch, member.clean_key, observed=False)
    assert memory.publish_epoch(1).groups == ((),)
    assert memory.stage_patch(0, 8, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(2)
    assert not memory.stage_patch(0, 8, member.raw_patch, member.clean_key, observed=True)


def test_query_enforces_strict_nonoverlap_causality() -> None:
    memory = make_memory()
    for start in (0, 16, 32):
        member = make_member(variable_id=0, start=start)
        memory.stage_patch(0, start, member.raw_patch, member.clean_key, observed=True)
    memory.publish_epoch(1)
    assert [item.start for item in memory.query(0, 32, [1.0, 0.0])] == [0, 16]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/unit/test_online_bank.py -v `

Expected: FAIL with the new module or public types missing.

- [ ] **Step 3: Write the minimal implementation**

Implement immutable dataclasses with defensive NumPy copies, a mutable pending dictionary keyed by `PrototypeIdentity`, and a published `OnlinePrototypeSnapshot`. On publish, sort pending identities, assign each member to the highest-similarity existing group when the current center similarity is at least `0.8`, otherwise create a new group; when a group reaches 16 members remove its oldest member, and when a variable reaches 128 groups remove its oldest group. Recompute every retained group center from its current members after all assignments. Add the strict causal filter in `query` before member ranking.

- [ ] **Step 4: Run focused tests and the existing prototype tests**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/unit/test_online_bank.py tests/unit/test_grouped_bank.py tests/unit/test_prototype_bank.py -v `

Expected: PASS; old offline grouped-bank behavior remains unchanged.

- [ ] **Step 5: Commit**

```text
git add src/malr/prototypes/online_bank.py src/malr/prototypes/__init__.py tests/unit/test_online_bank.py
git commit -m "feat: add online prototype memory state machine"
```

### Task 3: 确定性中心重算、FIFO 诊断与只读边界

**Files:**
- Modify: `src/malr/prototypes/online_bank.py`，补充容量诊断、状态复制和确定性哈希。
- Modify: `tests/unit/test_online_bank.py`，增加顺序无关性、淘汰后不可重插入、中心重算和只读检查。

**Interfaces:**
- `OnlinePrototypeMemory.state_dict() -> dict[str, object]`：包含配置、当前 snapshot、pending 和永久 seen；数组以 NumPy 值返回，调用方修改返回对象不能反向修改 memory。
- `OnlinePrototypeMemory.load_state_dict(state: Mapping[str, object]) -> None`：验证几何、容量、变量数和格式版本后恢复状态。
- `OnlinePrototypeMemory.snapshot_hash() -> str`：对当前 snapshot 的排序后身份、raw patch、key、组成员顺序、组中心和 epoch 做 SHA-256；同一状态得到同一哈希。
- `OnlinePrototypeMemory.diagnostics() -> Mapping[str, int | float]`：至少提供 `created_groups`、`joined_groups`、`evicted_members`、`evicted_groups`、`staged_observed`、`skipped_masked`、`causal_exclusions`。
- `OnlinePrototypeMemory.read_only_view() -> OnlinePrototypeSnapshot`：返回与内部状态隔离的冻结对象；查询不会增加任何计数，验证/测试可复用该对象。

- [ ] **Step 1: Write the failing tests**

```python
def make_memory(**kwargs: object) -> OnlinePrototypeMemory:
    return OnlinePrototypeMemory(
        patch_len=16,
        patch_stride=8,
        train_end=128,
        variable_count=1,
        **kwargs,
    )


def make_member(
    variable_id: int,
    start: int,
    *,
    clean_key: list[float] | None = None,
) -> OnlinePrototypeMember:
    key = np.asarray(clean_key or [1.0, 0.0], dtype=np.float64)
    raw_patch = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    return OnlinePrototypeMember(variable_id, start, raw_patch, key)


def test_group_center_is_recomputed_from_retained_members_only() -> None:
    memory = make_memory(group_capacity=2, group_similarity_threshold=0.8)
    for start, key in (
        (0, [1.0, 0.0]),
        (8, [1.0, 0.0]),
        (16, [0.8660254, 0.5]),
    ):
        member = make_member(variable_id=0, start=start, clean_key=key)
        memory.stage_patch(0, start, member.raw_patch, member.clean_key, observed=True)
    snapshot = memory.publish_epoch(1)
    retained = snapshot.groups[0][0]
    assert [member.start for member in retained.members] == [8, 16]
    assert np.allclose(retained.center, np.array([0.9659258, 0.2588190]), atol=1e-5)
    assert not np.allclose(retained.center, np.array([0.9781476, 0.2070552]), atol=1e-5)


def test_publish_is_independent_of_pending_submission_order() -> None:
    first = make_memory()
    second = make_memory()
    members = [make_member(0, start) for start in (32, 0, 16)]
    for member in members:
        first.stage_patch(0, member.start, member.raw_patch, member.clean_key, observed=True)
    for member in reversed(members):
        second.stage_patch(0, member.start, member.raw_patch, member.clean_key, observed=True)
    first.publish_epoch(1)
    second.publish_epoch(1)
    assert first.snapshot_hash() == second.snapshot_hash()


def test_evicted_identity_cannot_be_inserted_again() -> None:
    memory = make_memory(group_capacity=1, group_similarity_threshold=1.0)
    first = make_member(0, 0)
    second = make_member(0, 16, clean_key=[1.0, 0.0])
    memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)
    memory.publish_epoch(1)
    memory.stage_patch(0, 16, second.raw_patch, second.clean_key, observed=True)
    memory.publish_epoch(2)
    assert not memory.stage_patch(0, 0, first.raw_patch, first.clean_key, observed=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/unit/test_online_bank.py -k 'center or order or evicted' -v `

Expected: FAIL because the snapshot hash, state/diagnostic APIs and explicit FIFO accounting are not implemented.

- [ ] **Step 3: Write the minimal implementation**

Track counters only on staging/publish events, never on queries. Recompute centers as the normalized mean of the retained members' normalized keys; do not retain a cumulative center sum. Serialize all mutable and frozen state with version `1`, deep-copy arrays at both boundaries, and hash a canonical JSON header plus C-contiguous array bytes in sorted identity order.

- [ ] **Step 4: Run focused tests plus all unit tests**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/unit/test_online_bank.py -v; $env:PYTHONPATH='src'; python -m pytest tests/unit -q `

Expected: PASS with no mutation of existing unit-test behavior.

- [ ] **Step 5: Commit**

```text
git add src/malr/prototypes/online_bank.py tests/unit/test_online_bank.py
git commit -m "test: harden online memory determinism and state boundaries"
```

### Task 4: 现有配置与在线状态的公共导出和轻量集成契约

**Files:**
- Modify: `src/malr/prototypes/__init__.py`，稳定导出在线状态类和 `PrototypeIdentity`。
- Create: `tests/integration/test_online_memory_protocol.py`，验证配置选择与内存协议能够被后续训练代码独立导入。

**Interfaces:**
- `from malr.prototypes import OnlinePrototypeMemory, OnlinePrototypeSnapshot, PrototypeIdentity` 必须可用。
- 在线状态机不在本任务接入 CLI、trainer、checkpoint 或模型前向；调用方只能通过 `stage_patch`, `publish_epoch`, `snapshot`, `query`, `state_dict`, `load_state_dict`, `snapshot_hash` 访问。

- [ ] **Step 1: Write the failing integration test**

```python
def test_online_memory_public_protocol_is_importable() -> None:
    from malr.prototypes import OnlinePrototypeMemory, OnlinePrototypeSnapshot

    memory = OnlinePrototypeMemory(
        patch_len=16,
        patch_stride=8,
        train_end=128,
        variable_count=1,
    )
    assert isinstance(memory.snapshot(), OnlinePrototypeSnapshot)
```

- [ ] **Step 2: Run test to verify it fails**

Run: ` $env:PYTHONPATH='src'; python -m pytest tests/integration/test_online_memory_protocol.py -v `

Expected: FAIL if the public exports are absent.

- [ ] **Step 3: Write the minimal export/integration code**

Export only the stable online types; do not alter `cli.py` or call the online memory from training yet. Keep the legacy exports and import paths unchanged.

- [ ] **Step 4: Run the complete available test suite**

Run: ` $env:PYTHONPATH='src'; python -m pytest -q `

Expected: all existing tests plus the new protocol tests pass; any S4M test requiring an external checkout may remain skipped under its existing condition.

- [ ] **Step 5: Commit**

```text
git add src/malr/prototypes/__init__.py tests/integration/test_online_memory_protocol.py README.md
git commit -m "feat: expose online prototype memory protocol"
```

## Plan Self-Review

- The plan is intentionally limited to the first independent subsystem; relation construction, unified edge-gain supervision, routing, GNN recovery, trainer integration and paired checkpointing remain separate later plans.
- The selected ingestion rule is explicit: only `observed=True` patches are staged, and the configuration rejects an online mode that disables this rule.
- Current-member center recomputation is tested separately from legacy cumulative-center behavior, so the old bank implementation is not silently changed.
- Epoch `0` is the empty initial snapshot; publication, not staging, makes a member queryable.
- Strict causality is expressed with the exact inclusive boundary `start + patch_len <= query_start`.
- No task changes CLI behavior or claims that the new YAML is training-ready before later relation and trainer tasks are complete.
