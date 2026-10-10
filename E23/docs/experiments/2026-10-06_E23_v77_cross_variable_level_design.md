# E23-v77 跨变量全局水平补偿

## 证据与假设
已训练v72在保持target0及normalized形状不变、source1全局水平+1的访问探针中，target0修复/预测输出差为0。v75在CNN加两通道全局值后没有改善，不能继续声称全局特征即有效。本候选把少量全局水平信息直接送到已有近期均值修复器，独立于原CNN/形状分支。目标是长缺失下均值/水平偏差，尚不能证明因果机制。

## 唯一架构改变
recent_mean_projection=global_level_mean_v2。保留mean_v1原205→64→1网络和所有原计算，另建49→64→1训练MLP（3265参数，最后输出零初始化，fork_rng不改旧初始化）。49输入为7变量各6个真实可见全局摘要（全96均值、后32均值、最后可见值、全96可见比例、后32可见比例、global std）及target onehot7。std使用已有可见query_std、变量无可见时该变量6摘要均置零；完全无真实观测窗口新增残差为零。全局值仅从explicit.observed_sequence与可见mask、query_mean/std计算；隐藏/无效payload先mask再算，未来/clean缺失不进入forward。

新增输出为2*tanh(raw/2)，单位为训练集全局标准化，不乘target局部std。与原delta*r相加，仍只分配到后32真实缺失坐标，沿用share=min(32/hidden_count,4)、观测/前64/validity/coverage不变。若hidden少于8点，delta并非实际均值变化，沿用sharecap并如实描述；返回existing recent_mean_delta为总名义globaldelta。

## 固定设计与条件父版
修复器v72主干420973+3265=424238，最终head仍35427 seed2023原初始化/结构。是否继承v76辅助只能据固定验证晋级，正式前锁定；不依据其测试。原Stage1 globalMSE+.1retrieval+recent_mean_v2+gapmoments(.2/.2)和100轮/batch128，原Stage2 fullfreeze/MSE+.2MAE/uniform/lr1e-4/zero_invalid及100轮保持。Stage1原固定验证global修复MSE、Stage2固定MAE主/MSE次，一次训练双checkpoint先锁后20开发tests，T1缓存不重训。未评分新20确认保持锁定。

## 检验
7项新增行为RED→GREEN：旧函数/所有旧参数/RNG/head零起点相同；全局跨变量访问有效；hidden/invalid NaNInf隔离；无来源/小std有限；globaldelta单位与scope/observed保持；两步原loss可训练新分支；checkpoint严格往返和身份篡改拒绝。之后真实CUDA候选历史库/长段loss/freeze，资源含全部新增训练并列训练/推理参数。按验证双指标裁定；无保证能超过T1。
