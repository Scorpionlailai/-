# Beijing sr020_bl8：T1基线权重

这是 BeijingAirQuality（12站点PM2.5）、sr020_bl8、训练seed2023的 **T1基线**，不是R3模型权重。
原实验于2026-10-13归档；现按用户要求公开原始权重文件，不重新训练，不重选轮次。

| 文件 | 用途 | 完整训练预算 | 选中轮次 |
|---|---|---:|---:|
| [stage1/selected.pt](stage1/selected.pt) | T1修复器；验证MSE选模 | 100轮／19000更新 | 90 |
| [stage2/selected_mae.pt](stage2/selected_mae.pt) | 冻结T1后训练的公共预测头；验证MAE主选择 | 100轮／19000更新 | 69 |
| [stage2/selected_mse.pt](stage2/selected_mse.pt) | 同一预测训练的验证MSE次选择 | 100轮／19000更新 | 69 |

文件大小分别为2,428,448、173,376、173,376 bytes，可直接从Git下载，无需LFS。逐文件SHA256见 [checkpoint_publication.json](checkpoint_publication.json)，已与选模记录核对并验证CPU `torch.load(..., weights_only=True)`读取成功。

Stage1为T1Adapter的state_dict，键带`core.`前缀；直接加载原T1.Model时需要对应去除这一层前缀。Stage2是公共PatchTST预测头的state_dict，不是T1修复器、不是第二套修复模型，且两个头必须搭配本目录的Stage1与原数据标准化/遮挡/窗口协议。
两个Stage2文件均选中第69轮，张量数值一致；PyTorch ZIP封装文件名等元数据使文件SHA不同，保留原始字节以对应各自选模记录。

配置及数据清单指纹见 [plan.json](plan.json)；训练设置与选模见 [Stage1记录](stage1/selection.json)、[Stage2记录](stage2/selection.json)。代码为仓库的原T1及Beijing基线适配入口。原评价20组mask平均全96修复MSE=0.22552305693429445、MAE=0.26433947761505483；公共预测头MSE=1.3030686654665173、MAE=0.7793389416960619。这些是单训练seed2023成绩，不是三训练seed均值。

本次只发布这三个权重和必要元数据，不上传原始数据或大型修复预测缓存。
