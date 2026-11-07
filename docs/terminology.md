# EviRefine naming and writing convention

Paper name: **EviRefine: Evidence-Guided Coarse-to-Fine Multivariate Time-Series Imputation**.
Chinese: **EviRefine：证据引导的粗到精多变量时间序列缺失修复**。

| Paper module | English | 中文 | Responsibility |
|---|---|---|---|
| TVCE | Temporal–Variable Context Encoding | 时间–变量上下文编码 | Temporal convolution and variable-signature attention produce point context and base patch features. |
| MPER | Multi-Source Patch Evidence Retrieval | 多来源Patch证据检索 | Organize eligible current-window and training-history members with source, quality and prior features. |
| PEGR | Pointwise Evidence-Guided Reconstruction | 逐点证据引导重建 | Point-query attention and joint query/readout residual reconstruction. |
| OCRR | Observation-Conditioned Residual Refinement | 观测条件化残差精恢复 | Frozen-coarse, visible-observation-conditioned numerical correction. |

Auxiliary names: Confidence-Weighted Overlap Fusion; Base Reconstruction Head; Joint Residual Head; Neural Residual Head; Gated Reference Correction. Operator VSAttn denotes variable-signature attention adopting T1 channel–head geometry; it is not claimed as an independently invented attention mechanism.

## Internal traceability (not paper labels)

EviRefine maps to complete R3/H_GLOBAL. E0 is full model; E1/E2/E3 remove temporal/cross-variable/historical evidence; E4 uses source-wise random legal selection; E5 uses unified similarity retrieval; E6 is the capacity-matched context-only control; E8 shares patch readouts. Frozen CSV row keys and original source filenames remain unchanged. Published T1 and other literature method names remain unchanged.

## Reference and prose

TimeMixer++ official paper: https://arxiv.org/html/2410.16032v4 . Its high-level exposition (challenge, design principle, component overview, operation and effect) informs the organization. Wording, claims and mechanisms describe EviRefine's actual implementation; no text is copied. Exact-name web queries for EviRefine returned no results on 2026-11-07; this is a preliminary naming check.

The user-provided drawio is renamed by label only. Geometries, IDs, styles and connections are preserved. The source backup is outside this Git root. Diagram export is still required to replace the compact textual architecture placeholder in the compiled PDF.
