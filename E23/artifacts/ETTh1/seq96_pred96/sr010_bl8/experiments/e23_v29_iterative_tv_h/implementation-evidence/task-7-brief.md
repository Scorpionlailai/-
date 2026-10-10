### Task 7: CUDA 成本预检、旧权重兼容及代码审查

**Files:** Create `scripts/profile_e23_v29.py`; 检查 Tasks1–6 全部文件；结果写入新实验目录下 `preflight.json`、`code-review.json`、`verification.json`。

**Interfaces:** profiler CLI 为 `--config-one PATH --config-three PATH --output PATH --device cuda:1`，只做 throwaway profiling，不产出选定 checkpoint。输出每模型 repair参数数、forward/backward毫秒、峰值allocated/reserved GiB、候选可用性、GPU UUID；记录同批、同精度和 warmup 次数。

- [ ] **Step 1:** 保存当前起点及冻结前文件哈希。profile模型分别从新配置构建，使用训练数据发布的临时 H bank；batch128、float32、4个 warmup、10个测量 step，CUDA同步后计时，不更新优化器或复用临时模型做正式训练。
- [ ] **Step 2:** 运行 profiler，在新目录保存实际时间/显存；三轮 forward/backward 有限，H/T缓存计数符合 spec。若成本高于预期仍记录实际值，不擅自缩小 batch、轮数或训练 epoch。
- [ ] **Step 3:** 用实际保存的 v3/v12/v28 config+manifest 严格加载旧 Stage1/2；对 v28 原固定验证重放，复现 0.7402293330687785 / 0.5855096656989619 至1e-6并保存 H key、bank前后哈希。现有 `scripts/replay_e23_p4_validation.py` 的旧路径检查保留。
- [ ] **Step 4:** 运行 `python -m pytest -q`，保存完整日志；只修复真实失败或必需门槛。独立代码审查重点为梯度断开、预更新泄漏、重复写回、缓存跨forward、参数重复注册、新模式身份和旧默认行为；记录审查结论并解决 Critical/Important。
- [ ] **Step 5:** 执行 `git diff --check`，保存实现冻结哈希与已验证状态；按工作区约束保存/提交本轮变更，随后开始正式训练。

