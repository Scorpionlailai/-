# E23 Width 64 Accuracy Iteration

Date: 2026-09-30. This is a separate ETTh1 P4 variant after the completed E23 width 16 trial. The original selected Stage 1/2 checkpoints and one-shot test record remain immutable.

## Purpose and evidence

The primary goal is to improve selected-test forecast MSE and MAE, with SAITS test MSE `0.4405843816917229` and E22 test MAE `0.4332482160624463` as accuracy gates. Training and inference time are measured and reported, but need not beat SAITS. The width 16 E23 trial reached validation forecast MSE `0.7827891416830811`, yet selected-test MSE/MAE were `0.4481868149201616`/`0.4392647526972418`. Its selected Stage 1 hidden-coordinate validation MAE was `0.9176137581263981`, versus selected SAITS Stage 1 validation MAE `0.5978319887024138`. E23's repair module has 9,897 parameters. Validation-only removal of H or T worsened forecast MSE; removing V improved MSE slightly but worsened MAE. This supports testing repair capacity before replacing experts.

## Alternatives and choice

1. **Widen the existing repair encoder to 64 latent channels (selected).** This preserves the reviewed explicit repair, H/V/T evidence and shared router, isolates one architectural variable, and gives all experts and the decoder more room to model masked history. It increases compute and may still fail to generalize.
2. Replace T's directional candidate selection with dense self-attention. This may improve context use but changes selection semantics and would make it hard to attribute an accuracy gain.
3. Add a SAITS teacher or pretrained imputer. This would blur independence of the comparison and add a large dependency, so it is outside this iteration.

## Model and protocol

The only architecture setting changed from the width 16 trial is `model.d_model: 64` in a new E23 config. The repair base projection, H query/key width, H/V/T message layers, router, and shared decoder use width 64. The forecast predictor remains the same 16-channel PatchTST and mask projection used in E22 and the SAITS comparison. All expert candidate bounds, thresholds, offsets, temporal bands, residual initialization and original-observation clamp remain unchanged. Recent T band and router batching commits preserve their output semantics and are included in the width 64 code revision.

The locked P4 ETTh1 CSV, train/validation/test bounds, train-only normalization, dynamic training masks, cataloged fixed validation/test masks, seed `2021`, input/prediction lengths `96/96`, patch `16/8`, batch `128`, and both `100`-epoch stages remain unchanged. No future target enters repair inputs, H state, or training selection. H prototypes stay training-only, causally filtered and paired to the frozen retrieval key view. Stage 1 retains hidden-coordinate SmoothL1 plus `0.1` retrieval loss and validation repair MSE selection. Stage 2 starts a new optimizer and uses forecast MSE only, selecting the lowest fixed-validation forecast MSE. The Stage 2 record must still include the width 64 config hash and H bank/key hashes.

## Execution and assessment

The config loader accepts only the reviewed widths 16 and 64 and rejects any other P4/model/training overrides. Width 64 artifacts live under a fresh `e23_multi_expert_width64_v2` root, never under the width 16 trial. First verify config identity, model/forecast shapes, H snapshot key width, finite gradients, and unchanged observation clamp on synthetic batches. Run both complete stages on one GPU with progress and atomic selected checkpoints. Compare validation repair MSE/MAE and validation forecast MSE/MAE with width 16 and SAITS; do not choose based on test. Profile the selected model on the same fixed input order, then execute exactly one selected width 64 test and report MSE, MAE, both DRRs, expert availability/weights, training time and inference throughput. If the accuracy gates still fail, retain the result and start a new design iteration; do not retune this run to its test result.

## Failure handling

Config, source, mask, Stage 1 checkpoint, H bank and key mismatches fail before Stage 2 or evaluation consumes a batch. A failed or interrupted run uses a new output directory and never overwrites a selected record. A width 16 checkpoint loaded under width 64 must fail strict state restoration or config identity validation. The old E22/SAITS repositories and archived artifacts remain read-only.
