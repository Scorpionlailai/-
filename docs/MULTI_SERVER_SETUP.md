# 多服务器使用说明

## 下载

取得实际仓库地址后，在其他服务器执行：

```bash
git clone <实际的GitHub仓库地址> paper
cd paper
```

私有仓库先完成该服务器的 GitHub 登录或配置 SSH key；不要把 token 写入配置文件、脚本或实验日志。

## 环境

我们的基础工程需要 Python >=3.10，依赖见 `E24/pyproject.toml`。例如：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './E24[dev]'
```

这只安装我们的基础工程，不保证安装所有作者基线依赖。基线分别使用作者环境：TimesNet/CSDI 查看各自 `requirements.txt`；ImputeFormer 使用 `conda_env.yml`，并检查作者随仓库提供的 `experiments/tsl` 与 `experiments/imputeformer`；BRITS 阅读其 README 并适配旧版代码兼容性。每个模型独立环境，记录实际版本。

## 路径和已有记录

1. 新服务器使用实际克隆根目录替代配置/脚本里的 `/home/xyw/paper`；保留本机历史记录不改写。
2. 历史控制器可能引用 `/home/xyw/.cache/codex-worktrees/...`、既有 checkpoint 或原服务器 Python 路径，不能直接启动这些控制器。对应源码已在 `E23/versions/.../E23`；按模型版本重建新运行配置。
3. 不对仓库全文执行盲目路径替换：source-freeze hash、checkpoint 记录和审计数据不可据此改变。仅在新运行配置及适配入口中使用新路径。
4. 本仓库不含训练权重或检索库二进制缓存；需要从头训练/构建，或另行同步指定权重及原 hash。
5. 新基线共享协议适配尚待实现。作者脚本可以作为参考；直接使用其默认随机点缺失不能得到我们的四个块缺失场景正式结果。

## 正式实验的固定要求

- 主任务：修复96点回溯窗口；未来预测长度96仅用于后续 Stage 2。
- 场景：起点率/块长0.10/8、0.15/8、0.20/8、0.10/16。
- 统一数据划分、训练集标准化、窗口列表、固定验证及20组测试掩码资产与 hash。
- 分别报告全96点和近32点隐藏有效位置 MSE、MAE，以及20组均值和样本标准差。
- 各模型按作者公开配置训练，准确记录任务适配、种子、实际更新次数和选模准则。
- 全流程时间、训练峰值 allocated/reserved 显存、GPU型号、精度与修复推理耗时均回传。
- 后续 Stage 2：完整冻结修复器并 eval，保持原 PatchTST，纯MSE，100轮；禁止联合训练。

完整规则与当前基线实现状态以 `E23/docs/experiments/2026-10-10_repair_baselines_remote_reproduction_handoff.md` 为准。

## 回传

每台服务器单独命名新运行目录，回传配置、依赖环境、作者源码 commit / 适配 diff、checkpoint、20组逐组指标、资源 JSON、日志及所有共享资产 hash。权重和大缓存单独传输；汇总 JSON/CSV 和文档可提交到 GitHub。不要将多台服务器同一路径的文件互相覆盖。
