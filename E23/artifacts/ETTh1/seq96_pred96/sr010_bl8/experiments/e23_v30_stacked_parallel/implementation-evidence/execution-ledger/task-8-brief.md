### Task 8: 完成三模型正式训练与共同测试

**Files:** root/`l1_g8`,`l4_g1`,`l4_g8`各自stage1/stage2/logs/resources/validation_replay/diagnostics/test结果；runner与源代码此时已冻结。

- [ ] **Step 1:** 确认空闲GPU及所有预检/审查门槛，记录GPU UUID/并发条件；运行以下train命令，持久化完整日志及每阶段资源。长进程用可轮询session，每60秒内给用户有意义的状态更新。

```bash
export PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/run_e23_v30_trial.py --root artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v30_stacked_parallel --device cuda:1 --mode train
```

- [ ] **Step 2:** 逐模型核对两个阶段都完成100epoch、自己的Stage1配对权重及库、选模结果、resource文件和代码SHA。每个选定权重重放2688验证窗口/1806336预测坐标，MSE/MAE至1e-6并确认H key/库不变。
- [ ] **Step 3:** 保存三个模型的验证选择记录及预先选择结果，另列原v28门槛。按Task6诊断各层/组，先锁定这些结果；不能使用诊断手工截断层或删除组再评估成正式模型。
- [ ] **Step 4:** 冻结检查后执行`python scripts/run_e23_v30_trial.py --root artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v30_stacked_parallel --device cuda:1 --mode evaluate`；三模型各评估seed2021/2026100201/2026100202一次，同一选定权重，检查每组2688窗口/1806336坐标和mask SHA。
- [ ] **Step 5:** 保留全部结果；若训练中发生非有限/系统中断，保存失败阶段和最后状态，只在具体原因解决并保留新证据后恢复，不能偷偷缩短epoch或以临时profiling权重替代。

