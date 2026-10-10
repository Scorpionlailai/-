# E23-v79 设计与实验依据包

先读 `paper/E23/docs/experiments/2026-10-07_E23_v79_design_and_improvement_evidence.md`。

内容：设计脉络与三份提升统计CSV、相关历史报告及支持JSON、四场景v79/v69/T1匹配纯MSE冻结记录、主/次选权重、确认20位置结果与mask、Clean/修复诊断、源码冻结归档、配置/资源/CSV数据与可校验文件索引。保留v80-v82负结果报告。

没有复制E24运行中的Weather文件；没有启动训练。历史文件保持原SHA和绝对路径，不是脱离环境的一键运行镜像。`FILE_INDEX.csv`用于定位包内文件；旧源码目录移动由`paper/E23/archive/legacy_path_map_20261007.json`说明。完整旧实验仍在原workspace。v79源码位于正式实验的`frozen-source.tar.gz`。作者基线源码/运行环境未整仓重复打包，revision和source SHA留在原记录。

历史阶段配对与当前统一配方分开，不将不同seed、缺失参数、联合/冻结政策或单组/20组数字混为模块消融。修复为固定验证诊断，正式20组测试修复尚未补齐。
