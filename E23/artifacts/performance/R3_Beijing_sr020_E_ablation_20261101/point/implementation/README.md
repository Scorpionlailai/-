# R3 ETTh1 E8: patch-shared candidate attention

Isolated experiment, same R3 selected candidates and all operators except attention query pooling. Original per-point queries feed the original output head. See ../plan.json and ../design.md. Four scenes/three seeds, fixed main100+frozen calibration100. E0 reused. No Stage2. E7 excluded from reports at user request.

Beijing sr020_bl8 explanatory replication, 2026-11-01. Same model operators and hyperparameters as corresponding ETTh1 ablation. Three seeds; identical original masks; frozen calibration. No forecast, no E7.
