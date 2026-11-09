# PAMAP2 追加交接给 Server B

先阅读 [执行提示词](SERVER_B_PAMAP2_PROMPT.md)，再按 [任务分配](server_assignment.json) 安排队列。

Server B 已完成北京与 Exchange 的24组，现追加 PAMAP2 的12组；Server A 保留 ETTh1、Weather、PhysioNet2012。原服务器已将 PAMAP2 从待运行队列移除，当前 Weather 子进程继续运行。

## 资产

[下载 PAMAP2_addon.tar.gz](https://github.com/Scorpionlailai/-/releases/download/ogrr-pamap2-server-b-20261109/PAMAP2_addon.tar.gz)，下载后执行 `sha256sum -c SHA256SUMS`。归档包含处理好的数据、原数据加载实现及四场景共84份原始验证/测试遮挡，不含模型权重；各文件哈希见 FILE_MANIFEST.json。先在 staging 解压核验，再按提示词安装，既有文件有差异时不得直接覆盖。

本轮只发布交接文件与数据补充包，没有新增训练、重新选模或改动论文。训练实现继续使用源交接 commit `f46822e78a4cb96b931ba8f54e233e1778ccf15c`，配置、预算、损失、种子和选模协议均不变。

本目录的 server_assignment.json 替换实验根目录的旧任务分配；PAMAP2 唯一归 Server B，旧完成记录保留，队列跳过已完成项。
