### Task 7: CUDA预检、旧权重验证、独立审查与冻结

**Files:** Create `scripts/profile_e23_v30.py`, `verify_e23_v30_compatibility.py`; evidence写root/`preflight.json`,`compatibility.json`,`verification.json`,`code-review.json`,`code-freeze.json`。

**Interfaces:** profiler CLI `--configs PATH PATH PATH --output PATH --device cuda:1`，临时训练数据H库、batch128/FP32、每阶段4warmup+10测量step，无optimizer更新；记录forward/backward ms、参数分项、峰值、GPU UUID、H查询数及各组有效性。compatibility CLI `--output PATH --device cuda:1`，读取既有checkpoint，严格加载旧v3/v12/v28/v29 Stage1/2并重放v29-r1及L1/G1参数映射参考。

- [ ] **Step 1:** 实现并运行真实CUDA预检三个模型。若某模型batch128 OOM，先清除临时模型，再以activation_checkpointing=true重测该模型；正式配置在训练前固定该设置并进入SHA，不改变batch、precision、epoch。无checkpoint用于正式初始化。
- [ ] **Step 2:** 严格旧权重加载；重放v29-r1验证MSE0.7499365438950035/MAE0.5840456761470457至1e-6，L1/G1映射参考同样复现，H key/库前后SHA一致。
- [ ] **Step 3:** 运行`python -m pytest -q`，保存完整日志；验证用户CSV SHA及基线外任务改动范围。只针对真实失败/所需门槛修复，不为低影响文档补镜像测试。
- [ ] **Step 4:** 按用户确认执行方式调用对应执行技能。采用本会话执行时，executing-plans要求正式训练前一位新鲜独立reviewer审查全部实现；重点检查独立参数、V来源、批量数学、specialist泄漏、重计算副作用、历史时间边界、身份、资源归属。解决Critical/Important并保存结论。
- [ ] **Step 5:** 检查本轮增量`git diff --check`，不把既有CRLF警告归成新增失败；保存基线/增量与冻结文件SHA。新文件可独立提交，预先dirty文件先保存本轮patch证据再决定精准提交范围，不能整文件盲提交。

