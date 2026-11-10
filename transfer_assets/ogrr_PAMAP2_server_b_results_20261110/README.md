# PAMAP2 Server B 新增12组：完整结果

PAMAP2 × 四场景 × SAITS/ModernTCN/TimeMixerPP，12/12完成，失败0组。每组先训练原模型，再完全冻结所选骨干训练OGRR，最后以共同20组原基线遮挡配对评价。旧北京/Exchange的24组结果及权重保留，其他数据集未重训。

新增交接 commit：`b97f79f4a65df143042c97dee49e9ed0d1ab0aab`；封存实现 commit：`f46822e78a4cb96b931ba8f54e233e1778ccf15c`。Python3.10.21 / torch2.7.1+cu128 / numpy2.2.6，模型源码、结构、loss、batch、轮数预算、早停及选模保持交接配置。SAITS seed26，其余骨干seed2021；TimeMixer++的2021为预先固定缺省seed，不称作者最佳。OGRR seed2023，完整100轮。

## 配对结果

12/12组MSE、MAE均改善。以下为各组合20组mask均值的相对降幅，再对四场景作算术平均；不是合并坐标后的全局降幅。

|模型|四场景MSE平均降幅|四场景MAE平均降幅|
|---|---:|---:|
|SAITS|8.17%|14.83%|
|ModernTCN|14.49%|10.78%|
|TimeMixerPP|17.00%|15.67%|

[逐场景报告](report.md) · [完整指标CSV](comparison.csv) · [资产索引](asset_index.json)。逐20mask原始与精修复指标见各 `cases/PAMAP2/<scene>/<model>/results.json`。单训练seed结果；20份mask样本标准差不是训练seed波动。

## 完整归档下载

[server_b_PAMAP2_20261109_results.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-pamap2-server-b-results-20261110/server_b_PAMAP2_20261109_results.tar.gz)

SHA256：`ac054088ad468ae1a729f5e725ca01112595093c5a4859cf1c711e27d5871ecb`

归档包含48份原模型/OGRR selected和latest权重、各轮记录、原配置与source/data/mask绑定、240条配对测试记录、原日志、参数量、耗时、显存及batch128同设备推理p50；清单与逐文件SHA256在包内和本目录均有副本。

下载后先执行 `sha256sum -c BUNDLE_SHA256SUMS`。在独立空目录解压，再运行 `sha256sum -c SHA256SUMS`。合并时包内 `runs/PAMAP2/`、`logs/PAMAP2__...` 相对于 `paper/E23/artifacts/performance/OGRR_new_baselines_20261109/`，已有差异文件应保留。归档的 `provenance/` 和根级清单供核验，勿覆盖原服务器已有全局文件。

## 核验

上传前校验整包SHA256及3039个清单文件，含48份权重。12份结果与24份selected权重对齐complete记录，骨干冻结摘要一致；四场景84份原始mask逐文件及数组一致，40变量，训练/验证/测试窗口10958/1306/3349。原生缺失、记录/受试者边界和旧24组不变性检查记录见provenance。未修改论文或覆盖原实验结果。
