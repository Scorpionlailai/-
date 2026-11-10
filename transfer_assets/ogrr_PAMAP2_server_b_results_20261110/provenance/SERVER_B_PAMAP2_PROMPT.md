你接续已完成的 Server B 新基线任务，新增负责 PAMAP2，其他数据集不再重训。

范围：PAMAP2 × sr015_bl8 / sr010_bl8 / sr020_bl8 / sr010_bl16 × SAITS / ModernTCN / TimeMixerPP，共12组。先训练原插补模型，再冻结所选完整骨干，独立训练同一OGRR，配对评价原输出和＋OGRR的相同20组原始遮挡。无需运行PRISM或预测任务。

继续沿用交接commit f46822e78a4cb96b931ba8f54e233e1778ccf15c 的implementation、profiles、loss、选模、预算及implementation_sha256.json。不得通过减少轮数、改batch、改mask、重选seed加速。SAITS seed26，ModernTCN与TimeMixer++ seed2021，OGRR seed2023。TimeMixer++是预先固定缺省seed，不称作者最优。

本次附件PAMAP2_addon.tar.gz补充已准备的数据、原数据加载代码和四场景共84份验证/测试mask（21份/场景），不含模型权重。先用SHA256SUMS校验压缩包，在独立staging目录解压；逐文件核对FILE_MANIFEST.json。包内paper/...相对于/home/xyw。已有同路径文件先比哈希；相同则复用，出现差异先留存并排查，不能直接覆盖旧结果。原服务器已将PAMAP2全部移出待运行队列，不会重复训练。

安装数据后，先在CPU验证四场景load_data("PAMAP2",scene)：每场景必须verify_original_masks通过，训练10958、验证1306、测试3349窗口，40变量；原生缺失和记录/受试者边界必须保留，仅在原本有真值且人工遮挡的位置计分。禁止重新划分数据、跨记录切窗或自行重建不同mask。

将附件server_assignment.json安装到实验根目录（先备份原文件）。更新后server_a为ETTh1/Weather/PhysioNet2012，server_b为BeijingAirQuality/Exchange-Rate/PAMAP2，两边各36组，集合互斥且并集72。旧24组已完成的complete.json须保留，队列自动跳过。不要启动旧的未分区dual_gpu_queue.py。

使用现有环境，实验根目录为/home/xyw/paper/E23/artifacts/performance/OGRR_new_baselines_20261109。确认没有旧partition worker后，用两张空闲卡分别启动：

python orchestration/partition_worker.py --server server_b --slot 0 --gpu 0
python orchestration/partition_worker.py --server server_b --slot 1 --gpu 1

进程应通过nohup或独立持久会话运行并各写日志，禁止占用其他任务的GPU。首次单case smoke由队列自动执行；不改封存训练代码。PAMAP2原生缺失接口若出现异常，应先定位，不得忽略有效性或改损失。每组记录原/+OGRR MSE、MAE、20mask样本标准差、冻结摘要、权重/数据/mask/source哈希、完整时间、显存及同设备batch128推理p50。锁定验证checkpoint后再测试，不按测试改结构。

完成后保持与9e42d063e39f94e9bd22e98b16fcbd4c173ec491相同归档口径，结果、日志、selected/latest权重、清单及原始20mask指标全部保留。先本地完成核验并汇报；GitHub上传按用户授权执行，不改论文或覆盖旧结果。
