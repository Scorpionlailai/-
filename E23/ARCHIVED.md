# E23 封存说明

2026-10-07按用户要求暂停并封存。保留模型为 **E23-v79**；后续工作使用同级 `E24/`。

- [交接文档](../E24/docs/2026-10-07_E23_to_E24_handoff.md)
- [后续实验规范](../E24/docs/EXPERIMENT_PROTOCOL.md)
- [封存校验清单](archive/E23-v79_seal_manifest_20261007.json)
- [封存包](archive/E23-v79_sealed_20261007.tar.gz)
- [历史版本目录索引](versions/index.csv)
- [旧路径映射](archive/legacy_path_map_20261007.json)

原始实验JSON、checkpoint、资源记录和负结果保留；没有批量改写历史绝对路径。旧 `.e23-*` 源码已归入 `versions/`，查验旧路径使用 `archive/resolve_legacy_path.py`。Git worktree指针已修复。

基线代码仓库实体已移至paper同级；`artifacts/upstream/`中只保留兼容链接。既有基线训练结果仍作为E23历史实验的一部分留在artifacts。

队列与监测保持暂停；v83只有部分场景完成，v84-v86没有正式训练。不得把它们当成封存模型恢复运行。
