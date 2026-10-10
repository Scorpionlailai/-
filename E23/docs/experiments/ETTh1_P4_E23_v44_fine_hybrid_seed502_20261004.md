# E23 v44：细粒度三专家与CNN／跨变量注意力

状态：四组全部完成。

## 固定设置

- 修复输出16/16、8/8、4/4，分别6、12、24块；H保持16点父上下文及16/8训练观测库，候选按子块裁切。T/V读取跨度按原时间点对齐。
- 完整96点共享多尺度CNN，64通道、两个block、kernel31/5。Hybrid按通道进行变量轴注意力；CNN对照用本变量Q/K/V卷积结果均值。两者参数量和初始化完全一致。
- 支路在全局标准化尺度输出残差，读取完整96点、精修最近32点缺失值；Stage1 specialist singleton目标关闭支路残差。
- 新组各自100轮Stage1与100轮Stage2，batch128、FP32。Stage1全局缺失MSE+0.1检索，Stage2纯预测MSE；预测器LR1e-4、修复器LR1e-5。
- 原预测头PatchTST D16、35427参数、16/8输入与初始化一致；沿用T1对齐的padding和训练RNG重置。
- 开发seed502；主checkpoint按固定验证MAE，另存验证MSE最低版本。每组锁定后测试同一20组缺失位置。

## 验证MAE选模的主结果

| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | allocated / reserved峰值GiB |
|---|---:|---:|---|---:|---|
| E23-v44-P16-Hybrid | 0.438323 | 0.445739 | 0.733110 / 0.585212 | 8.11 | 1.360 / 1.732 |
| E23-v44-P8-Hybrid | 0.415202 | 0.429987 | 0.740012 / 0.586756 | 8.46 | 1.658 / 2.141 |
| E23-v44-P4-Hybrid | 0.424249 | 0.436216 | 0.740941 / 0.589443 | 9.79 | 2.278 / 2.916 |
| E23-v44-P4-CNN | 0.412746 | 0.424908 | 0.741703 / 0.586142 | 8.89 | 2.140 / 2.854 |
| E23-v41-tail32-slow | 0.401858 | 0.418268 | 0.737990 / 0.579953 | 8.77 | 2.200 / 5.930 |
| E23-v41-tail32-frozen | 0.403422 | 0.412411 | 0.758214 / 0.583535 | 7.55 | 2.200 / 5.930 |
| E23-v43-N | 0.411041 | 0.423668 | 0.739256 / 0.581975 | 7.66 | 1.053 / 1.564 |
| T1 | 0.388984 | 0.417673 | 0.756156 / 0.584897 | 15.34 | 1.779 / 2.518 |

T1、v41与v43-N为缓存参考，没有新训练。v41使用旧混合损失与修复保留，padding/RNG设置不同；新组之间才是粒度与注意力的匹配比较。T1保留作者Stage1 batch16/BF16配置。

## 修复与近期结构诊断

| 模型 | Stage1修复MSE | Stage2修复MSE | 最近32点均值误差 Stage1 → Stage2 | 最近32点斜率误差 Stage1 → Stage2 |
|---|---:|---:|---|---|
| E23-v44-P16-Hybrid | 0.293647 | 0.397957 | 0.041808 → 0.124949 | 0.00021264 → 0.00038003 |
| E23-v44-P8-Hybrid | 0.283302 | 0.344214 | 0.031738 → 0.081604 | 0.00017776 → 0.00026843 |
| E23-v44-P4-Hybrid | 0.320772 | 0.406371 | 0.036453 → 0.102708 | 0.00019388 → 0.00032507 |
| E23-v44-P4-CNN | 0.329636 | 0.429910 | 0.036923 → 0.076122 | 0.00021953 → 0.00033529 |

## 固定验证裁定

```json
{
  "status": "complete",
  "validation_gates": {
    "fine8_hybrid": {
      "matched_gate": false,
      "v41_reference_gate": false
    },
    "fine4_hybrid": {
      "matched_gate": false,
      "v41_reference_gate": false
    },
    "fine4_cnn": {
      "matched_gate": false,
      "v41_reference_gate": false
    }
  },
  "matched_pairs": {
    "fine8_hybrid_vs_fine16_hybrid": {
      "validation_mse_delta": 0.006902051350427096,
      "validation_mae_delta": 0.00154330238463396,
      "passed": false
    },
    "fine4_hybrid_vs_fine16_hybrid": {
      "validation_mse_delta": 0.007831120436964656,
      "validation_mae_delta": 0.004230846353128759,
      "passed": false
    },
    "fine4_hybrid_vs_fine4_cnn": {
      "validation_mse_delta": -0.0007615305669183003,
      "validation_mae_delta": 0.0033016767090950916,
      "passed": false
    }
  },
  "test_used_for_selection": false,
  "new_T1_training": false,
  "source_unchanged": true,
  "protected_files_unchanged": true
}
```

## 结果解读

**四组均未通过相对v41-slow的固定验证门槛，保留v41-slow开发参考。** 第一组的验证MSE更低，但MAE退步；其余组也未同时改善两个指标。全部主/副版本测试指标均未超过v41-slow的MSE和MAE。原正式五训练种子表保持。

- 本轮Hybrid中，8/8的测试MSE为0.415202，优于16/16的0.438323；4/4反而为0.424249。该趋势来自单开发种子的这组训练，不能认定patch越小越好。
- 同为4/4且参数、初始化一致，CNN测试MSE/MAE为0.412746/0.424908，Hybrid为0.424249/0.436216。新增跨变量注意力在本次方案中没有带来预测收益。
- 4/4 Hybrid的Stage1和Stage2整体修复MSE均比CNN更低，但Stage2最近32点均值误差更高：0.102708对0.076122。这与此前近期状态诊断一致：整体修复误差更低，仍可能没有保住预测需要的状态结构；本次没有对均值做新干预，不能仅由相关指标断言因果归因。
- 小块H上下文盲区已在实现中缓解：4/4 Hybrid的155905个完全缺失子块可以读取父16点上下文检索所得候选，父块完全缺失仍不启用H。四组训练库的原始观测坐标身份一致，训练后的key、分组和权重不同。
- 下一方向应检验在专家之前以共享CNN建立小块的上下文状态，以及保护Stage2的近期均值/趋势；本轮只验证末端精修方案，未实现或训练这两项后续改动。

## 统计和产物

修复按全局标准化尺度上的唯一缺失坐标计数；均值/斜率在完整最近32点窗口计算。原观测、mask和冻结H状态保持。
完整训练时间包括数据准备、两个训练阶段、每轮验证和保存；最终测试及额外诊断不计入训练时间。显存为PyTorch allocated/reserved峰值。
20组标准差衡量缺失位置变化；本轮单开发种子不能替代原五训练种子正式比较。
- [完整双选模CSV](ETTh1_P4_E23_v44_fine_hybrid_seed502_20261004.csv)
- [MAE主表](ETTh1_P4_E23_v44_fine_hybrid_seed502_20261004_primary.csv)
- [20组位置明细](ETTh1_P4_E23_v44_fine_hybrid_seed502_20261004_all_draws.csv)
- [修复与结构诊断](ETTh1_P4_E23_v44_fine_hybrid_seed502_20261004_recovery.csv)
- 实验目录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v44_fine_hybrid_seed502_20261004`；源代码、patch、配置、命令、资源与checkpoint均保存。
