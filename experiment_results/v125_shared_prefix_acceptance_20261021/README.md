# PAMAP2 双校准冻结计算复用：原服务器有限验收任务包

状态：待原服务器执行。本机持续目标保持 blocked；本包不宣称 PAMAP2 新实验已完成。

入口为 [REMOTE_TASK.md](REMOTE_TASK.md)，命令见 [RUNBOOK.md](RUNBOOK.md)。

唯一问题：保持原两个校准器、初始化、损失和独立训练轨迹，复用同批次冻结粗恢复能否降低实际训练成本？不删除分支、不改近期结构、不引入新损失、不改 HVT、不跑 Stage2、不读测试集选方案。

## 三类证据不可混用

- `evidence/existing_pamap2/`：原服务器 commit b5a72ddbf66d9822f53efa96b07354e2bd120b4d 的已有消融及校准归档。三seed收益是既有结论，不属于本轮新增验收。
- `evidence/weather_only/`：本机 Weather sr015/bl8，sealed original v125，128更新执行等价审计。40.49%仅为本机校准更新段节省，不能作为PAMAP2或完整Stage1实测。不是VC1。
- `templates/`：尚未执行的新增 PAMAP2 返回表模板。原服务器输出到新目录 `experiment_results/v125_shared_prefix_acceptance_20261021/results/<run_id>/`，必须标记 `NEW_PAMAP2_FINITE_ACCEPTANCE`。

`assets.json/csv` 包含6份粗权重确切原路径/预期SHA256及3个依赖路径。**三个依赖的预期SHA256未在已公开归档中提供，保留null，不能编造。** 原服务器必须从当时源码锁提取预期哈希并和实际文件比较。若原锁也无法恢复，明确标为新锁定源码版本，先建立原实现参考臂，不能声称严格重放历史原轨迹。

此包交接的是“需在原服务器接入的验收任务”，不是已验证的PAMAP2一键训练器。`code/local_worktree` 和 `local_changes.patch` 保存本轮全部5个新增源码文件；其中Weather runner依赖原本机工程，不可改dataset名字后直接用于PAMAP2。可复用的独立helper为 `integrations/e23/shared_frozen_calibration.py`。PAMAP2适配脚本须按REMOTE_TASK实现、记录commit并通过单批预检后才运行有限预算。

禁止重新跑已有三seed100轮消融、覆盖正式权重、将本轮两个epoch选出的诊断权重晋级正式模型。已知缺项不应引发本机重复扫描。
