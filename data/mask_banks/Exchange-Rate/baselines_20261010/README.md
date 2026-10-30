# Exchange-Rate 原基线遮挡银行

包含四场景，每场景1份验证遮挡(seed2021)及20份测试遮挡(seed2026101001–2026101020)，共84个原始npz。逐文件SHA256及observed_mask数组SHA256见各场景manifest.json。全部文件直接复制原shared_protocol/masks，无重新采样。

读取manifest.json时以其所在目录解析masks路径。original_dataset_manifest.json保留原记录及历史绝对路径，仅供追溯，不把历史服务器绝对路径当作当前路径。

原始txt SHA256：0127465b51e3cd3c360f8eb2be30cfd294689a2a55903eb8245aafc396626c7f。划分0:5311 / 5311:6070 / 6070:7588，seq96/pred96。训练集统计、支持区间和窗口口径保持原协议。训练遮挡是既有动态epoch方案，这个银行只存验证和测试遮挡。

R3入口可先生成本地资产，再将本银行observed_mask逐值比对；zip文件时间元信息差异不能替代数组比较。正式训练必须在逐值一致后开始。原始数据和权重未包含。
