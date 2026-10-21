# 命令与环境

下列PKG是本包的绝对路径，REMOTE_REPO为原服务器保存原v125_ablation源码的E24工程根目录。不要直接在正式消融目录修改/输出。本包发布分支为 `handoff/v125-shared-prefix-acceptance-20261021`。

## 获取与验证包

```bash
git fetch origin handoff/v125-shared-prefix-acceptance-20261021
# 用交接消息提供的完整40位commit创建独立checkout；不要reset当前工作目录。
git worktree add --detach /tmp/v125-acceptance-handoff <PACKAGE_COMMIT>
export PKG=/tmp/v125-acceptance-handoff/experiment_results/v125_shared_prefix_acceptance_20261021
cd "$PKG"
sha256sum -c SHA256SUMS
```

## 只读核验资产

```bash
python check_assets.py --output /tmp/v125-asset-inventory.json
```

首次因三个expected hash未知返回2是预期行为。根据原源码锁建立dependency-lock.json：

```json
{"provenance":"原源码锁绝对路径及其SHA256；若不可恢复则明确newly_locked及源码commit", "hashes":{"/home/xyw/paper/E24/integrations/v125_ablation/common.py":"实际核验后的64位预期SHA256","/home/xyw/paper/E24/integrations/v125_ablation/adapter.py":"实际核验后的64位预期SHA256","/home/xyw/paper/E24/integrations/v125_ablation/run.py":"实际核验后的64位预期SHA256"}}
```

若真实源码位于原ablation checkout，而非上述paper/E24路径，创建path-map.json，逐路径指定映射，保留原身份。

```bash
python check_assets.py --dependency-lock /tmp/dependency-lock.json --path-map /tmp/path-map.json --output /tmp/v125-verified-assets.json
```

输出路径禁止覆盖。读原服务器资产属于本次任务；本机无需再扫描。原始数据、H库、外部worker/evaluate等额外依赖继续按随包原design_lock/source_audit验证；“6+3”不是完整依赖树。

## 可立即运行的独立helper单元测试

使用已有环境，不擅自升级torch。原PAMAP2归档为Python3.11.17、torch2.5.1+cu124、CUDA12.4、RTX4090D、CPU4线程。pytest是测试依赖。其余numpy及原工程依赖沿用原环境并导出版本。

```bash
cd "$PKG/code/local_worktree"
PYTHONPATH="$PWD" python -m pytest -q tests/unit/test_v125_shared_frozen_calibration.py
```

此测试不需要PAMAP2数据，不能代替真实训练验收。

## 本轮修改代码与适配

`local_changes.patch`是本轮5个新增文件的精确补丁，源工程base commit在provenance.json。用 `git apply --check "$PKG/local_changes.patch"` 检查后仅在隔离开发checkout按需应用。旧Weather脚本仅作执行参考，不能在PAMAP2直接启动。推荐只移植shared_frozen_calibration.py并保留独立测试；另按REMOTE_TASK实现原服务器 `run_acceptance.py`，提交源码，回传commit。

**以下是原服务器需实现的CLI契约，并非包内已有可直接运行的PAMAP2程序。** 接口就绪且单批检查通过后执行：

```bash
export CUDA_VISIBLE_DEVICES=<一张空闲物理GPU编号>
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$REMOTE_REPO:$REMOTE_REPO/src:$PKG/code/local_worktree"
python run_acceptance.py --asset-lock /tmp/v125-verified-assets.json --arm A0 --seed 2023 --epochs 2 --batch-size 128 --modes reference shared resume --output <独立的新运行目录>
```

物理GPU映射后程序用cuda:0，不能照搬Weather脚本硬编码cuda:1。所有未设置的数值后端开关必须显式读取原环境、登记并保持两臂相同。设置不同后端属于新受控执行比较，不能声称历史bitwise回放。

导出环境：

```bash
python -m pip freeze > <新运行目录>/pip-freeze.txt
nvidia-smi -q > <新运行目录>/nvidia-smi.txt
```

## 返回结果

复制templates并填写到独立results/<run_id>，附源代码commit、命令、日志、锁文件、指标与核验、所有资源字段；保留原服务器权重路径和SHA256。只发布新结果目录，给本窗口准确结果commit。失败也回传，不要自行扩大预算或改容忍阈值。本窗口收到结果commit后读取并指导下一步。
