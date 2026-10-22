# 原A5显式融合质量特征清单

固定常量均为0，只在原A5既定边界替换。内部检索/消息质量、候选权重、来源可用性和合法性掩码保留；不是均匀融合。

|特征名称|原作用|常量|消费位置|
|---|---|---|---|
|H.ExpertEvidence.quality[0:64]|Weighted raw quality projected to learned64D expert reliability used in preview/final/point fusion|0|ExpertRouter.preview_gate/expert_logit (T/V);ExpertRouter.route_precomputed.shared_score (H/V/T);PointEvidenceRouter.quality[expert]|
|H.CandidateEvidence.quality[0]=key_similarity|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[H] then key/value projections|
|H.CandidateEvidence.quality[1]=negative_visible_error|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[H] then key/value projections|
|H.CandidateEvidence.quality[2]=top_two_similarity_margin|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[H] then key/value projections|
|H.CandidateEvidence.quality[3]=target_observed_fraction|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[H] then key/value projections|
|H.CandidateEvidence.quality[4]=log1p_group_support|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[H] then key/value projections|
|V.ExpertEvidence.quality[0:64]|Weighted raw quality projected to learned64D expert reliability used in preview/final/point fusion|0|ExpertRouter.preview_gate/expert_logit (T/V);ExpertRouter.route_precomputed.shared_score (H/V/T);PointEvidenceRouter.quality[expert]|
|V.CandidateEvidence.quality[0]=selected_prior|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[V] then key/value projections|
|V.CandidateEvidence.quality[1]=source_observed_fraction|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[V] then key/value projections|
|V.CandidateEvidence.quality[2]=overlap_times_consistency|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[V] then key/value projections|
|V.CandidateEvidence.quality[3]=signed_lag|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[V] then key/value projections|
|T.ExpertEvidence.quality[0:64]|Weighted raw quality projected to learned64D expert reliability used in preview/final/point fusion|0|ExpertRouter.preview_gate/expert_logit (T/V);ExpertRouter.route_precomputed.shared_score (H/V/T);PointEvidenceRouter.quality[expert]|
|T.CandidateEvidence.quality[0]=source_observed_fraction|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[T] then key/value projections|
|T.CandidateEvidence.quality[1]=relative_patch_distance|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[T] then key/value projections|
|T.CandidateEvidence.quality[2]=visible_consistency|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[T] then key/value projections|
|T.CandidateEvidence.quality[3]=target_observed_fraction|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[T] then key/value projections|
|T.CandidateEvidence.quality[4]=is_local|Original per-candidate explicit quality for point readout key/value encoding|0|PointCandidateReadout.quality[T] then key/value projections|

## 仍保留的入口及解释边界

candidate weights from original internal ranking
expert messages/candidate messages including internally used quality features
query/latent/content embeddings
point visible context and coarse values
query observation geometry/rho
candidate/source availability and native validity
null token/no-source protection
overlap disagreement and repair values
source/candidate counts and original selection policy

PointEvidenceRouter gets already-zeroed ExpertEvidence.quality; OverlapConfidence has no raw explicit-quality input but retains content/weights/availability/disagreement. Calibrators consume repairs/router latents/visiblepoints/statistics, not raw quality directly. Original internal cand.weights and message-embedded raw quality remain deliberately, so this is not total information erasure.

Same operands mean raw candidate values/ranking preserved at each intervention and fixed-input expert replay; different A0/A5 trained parameters and quality-altered sequential states may naturally yield different contextual messages. Not candidate-pool redesign, capacity-matched causal effect or uniform fusion.

Zero is the fixed input; downstream learned Linear biases remain in original A5, so projected constants can differ across experts and training. We do not zero parameters or force uniform weights.
