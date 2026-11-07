# EviRefine: bilingual CVPR working manuscript

英文投稿式初稿＋中文逻辑对照稿。两版共用公式、文献和数据生成表格；不是自动逐词翻译。当前为v0.1工作稿，方法命名为EviRefine；内部实现对应原完整带邻点R3/H_GLOBAL。

```bash
git clone --single-branch --branch manuscript/cvpr2027-initial-20261107 https://github.com/Scorpionlailai/-.git r3_manuscript
```

## 阅读

- `preview/main_en.pdf`：英文预览，官方CVPR2027 review样式。
- `preview/main_zh.pdf`：中文预览，单栏便于逻辑核对。
- `main.tex`：默认英文入口（引入`main_en.tex`）。
- `main_zh.tex`：中文入口。
- `sections/en` / `sections/zh`：一一对应的章节。
- `shared`：共用数学公式；`tables/generated`：共用数值表。
- `docs/evidence_ledger.md`：数字和模型身份；`docs/next_steps.md`：内部写作清单。

六数据集正式R3成绩、两数据集机制对照均已纳入。全文基线总对照表尚在内部清单中，不用旧v125代替R3，也不虚构外部比较。架构图占位框可直接由`figures/evirefine_architecture.pdf`替换。

## Overleaf

使用`downloads/evirefine_cvpr_bilingual_v0.1_overleaf.zip`上传源代码后，英文设`main.tex`为Main document；中文设`main_zh.tex`并选择XeLaTeX。选择XeLaTeX也可编译英文。中文不套CVPR版式，公式/数据保持一致。

本分支只有论文，不包括实验代码、原始数据或权重。尚未连接具体Overleaf项目。提供Overleaf Git地址并在服务器本地配置认证后，可将同一源文件推送到该项目；不要把认证token写入仓库。

## 本地重建

```bash
python tools/generate_tables.py
python -m pytest -q
python tools/build.py
```

`tools/build.py`使用Tectonic编译并检查引用/布局，输出`preview/`与本地`dist/`。首次编译会下载TeX资源。也可以使用TeX Live：英文`latexmk -pdf main.tex`，中文`latexmk -xelatex main_zh.tex`。

## 数据和文献

`data/snapshots`只保存已汇总数值和归档说明，无原始样本/遮挡二进制/权重。来源与哈希见`data/source_manifest.json`。正式PAMAP2与机制PAMAP2来自不同评价队列，分表呈现，不混算。每模型20mask先平均，再汇总3训练seed；PAMAP2单来源消融单独标seed2023。

参考文献使用原站核对的BibTeX，来源见`data/reference_sources.json`。`cvpr.sty`、`ieeenat_fullname.bst`来自未修改的官方CVPR2027 author-kit；来源和SHA见`template_source.json`。论文尚有内部修订事项，见写作清单。

写作风格已应用[anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing)，移除重复免责声明，保留必要实验口径。

## 学术命名

论文统一使用EviRefine及TVCE/MPER/PEGR/OCRR。内部实验代号仅用于来源映射，见 `docs/terminology.md`。
