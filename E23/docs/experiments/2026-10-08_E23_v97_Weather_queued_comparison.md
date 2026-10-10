# E23-v97 Weather 对照队列

当前安排：先等待 E23-v103 的 Weather 0.10/8 Stage1 100轮及20组修复评估完成，再在GPU0运行v97；不改E24源文件或后台任务。

用户已授权跑v97。场景暂沿用当前v103的0.10/8；已询问是否改为打开的E24表所用0.15/8，等待用户回复时先完成独立准备。队列状态以实际 `queue_status.json` 为准。

## 模型与训练

- 使用v102中适配变量数后的v97 FP32实现。Weather21变量；回溯96/预测96，patch16/stride8。
- V为原固定4来源；H/T/路由/原PatchTST均保持v97设置，不混入v103动态策略。
- 继承同场景Weather-v79、seed2023的100轮粗修复权重与历史库。
- Stage1额外细化100轮：冻结专家主干及历史库，只训练已有均值分支和新dense残差，共109,719参数。Adam lr0.0001，batch128。损失为全隐藏位置MSE + 0.25×近32点MSE + 0.05×近32点MAE，均在训练集标准化空间。
- 按固定验证全窗口隐藏修复MSE选Stage1权重。测试不参与选模。
- Stage2从原预测头初始化训练100轮，seed2023、batch128、Adam lr0.0001，纯预测MSE，修复器全冻结。验证MAE为主checkpoint，验证MSE另存副checkpoint。启用固定验证缓存，预测头不变。
- 两个Stage2 checkpoint各测试同一组20个Weather开发mask，报告预测96点、修复96点、近32点的MSE/MAE、mask标准差。
- 记录细化、预测训练完整命令耗时和显存。总成本另计入父粗修复成本；这是三次命令成本相加，不是一轮连续冷启动实测。测试评估时间单列。

## 前置验收

CPU检查通过：父模型身份与checkpoint严格核验；新增缺失state keys只属于dense分支；真实Weather两个验证窗口中，新增零残差模型输出与父模型逐张量完全相同。正式训练还未开始，不存在v97 Weather的新精度结果。

## 路径

实验根目录：`/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v97_weather_20261008`。

- `plan.json`：设置与父记录。
- `preflight.json`：前置核验。
- `source_freeze.json`：源码与入口散列；各阶段启动前核验。
- `queue_job.json`、`queue_status.json`、`queue.log`：队列PID、状态及日志。
- `train_refinement.log`、`train_forecast_and_evaluate.log`：启动后生成。
- `E23-v97/sr010_bl8/stage1`、`stage2`：权重、选择与资源记录。
- `E23-v97/sr010_bl8/evaluation_mae`、`evaluation_mse`：最终20组对照。
- `complete_training_cost.json`：完成后生成，包含复用父阶段的成本。

如v103依赖阶段失败，队列会停止并记录失败，不抢占正在运行的GPU0任务，不自动改变批大小、轮数或场景。
