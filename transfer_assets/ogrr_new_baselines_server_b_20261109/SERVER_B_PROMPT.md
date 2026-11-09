你负责复现三个插补基线及冻结骨干后的 OGRR，只做 BeijingAirQuality、Exchange-Rate。

先解压交接包，阅读 EXPERIMENT_DESIGN.md、profiles.json、server_assignment.json、BUNDLE_MANIFEST.json，再检查路径与环境。不要启动全部六数据集的旧队列。

任务范围：两个数据集×四场景(sr015_bl8,sr010_bl8,sr020_bl8,sr010_bl16)×SAITS/ModernTCN/TimeMixerPP，共24组；每组先训练原模型，再冻结所选完整骨干、独立训练OGRR，最后在共同20组测试mask上配对评价。ETTh1、Weather、PAMAP2、PhysioNet2012由原服务器负责，不得重复。

口径：SAITS作者默认seed26、其余两种seed2021；TimeMixer++未发现作者最佳seed，2021是预先固定的缺省，不能写成作者最佳。OGRR固定seed2023，单种子。SAITS与ModernTCN用已封存作者NN代码，TimeMixer++为用户同意的PyPOTS实现，必须注明版本与固定配置。不要重新搜索种子或用测试调参。

使用交接包原始数据与mask。Beijing保留原生缺失、站点边界；Exchange保留行索引70/10/20划分与96窗口。训练动态遮挡，验证固定seed2021，测试20组原共享遮挡；所有指标只在native-valid且被人工遮挡的位置计算。逐值核验全部21份遮挡，不要新生成不同的test seed代替。

先运行 implementation/test_models.py 的5项测试和原数据/遮挡核验。源代码SHA256与implementation_sha256.json逐项对齐。每个正式case会自动执行短跑、前后向检查、选定checkpoint导出回放、缺失真值输入不变性、可见值保护、原生缺失与全窗缺失变量检查，再进入全训练。

OGRR必须完全冻结原模型参数与buffer且保持eval模式，不允许联合训练；使用当前100轮、batch128、原损失及验证full96 MAE选模。SAITS/ModernTCN/TimeMixer++各自原配置、最大轮数和早停见profiles.json，不统一硬改为相同结构或轮数。

环境：Python3.10、torch2.7.1+cu128、numpy2.2.6（完整环境见BUNDLE_MANIFEST）。主要NN仅依赖torch/numpy，旧数据入口另有封存依赖。源码内保留/home/xyw路径：若服务器路径不同，先核对实际目录映射，再做最小路径适配；不要修改NN数值实现、mask生成算法、损失、选模或配置，不要覆盖已有同名文件。路径适配及对应新哈希单独记录。

双卡启动命令（将PY替换成你服务器的Python环境）：
PY /home/xyw/paper/E23/artifacts/performance/OGRR_new_baselines_20261109/orchestration/partition_worker.py --server server_b --slot 0
PY /home/xyw/paper/E23/artifacts/performance/OGRR_new_baselines_20261109/orchestration/partition_worker.py --server server_b --slot 1
两个命令用独立后台进程并保存日志，实际只有一张卡时依次运行两个slot。不要运行implementation/dual_gpu_queue.py，它属于原始全量队列。

保存每轮状态、source/config/data/mask哈希、原模型与OGRR selected/latest权重、逐20mask原始与精修复MSE/MAE、均值及mask间样本标准差、训练/验证/完整流程耗时、峰值allocated/reserved显存、batch128同设备推理p50、实际参数量和负结果。20个mask不是20个训练seed。

每完成一个数据集先汇报并提供结果归档供原服务器合并。不要改论文、不要删CSDI历史记录、不要把缺失或失败项写为已完成。遇到缺项先列出具体路径和预期哈希，不自行重新训练其他旧基线。
