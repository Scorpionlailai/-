# E23-v78 preliminary source review

Scope: implementation-only review of global_aligned_v2 against the v78 design and plan. No GPU, training, optimizer, formal helper execution or source modification. The final parent, formal configuration, CUDA preflight and formal freeze remain outside this preliminary review.

## Findings

Critical: none found.
Important: none found.
Minor: no substantive source issue found.

## Implementation assessment

- Verified the copied-parent inventory against the v77 source tree. Exactly four existing files differ from that copy: config.py, train.py, model.py and historical_candidates.py. No head or parameter definition changed. The new mode uses the existing key encoder and adds zero parameters.
- `integrations/e23/train.py:497-523`: training query is converted from window normalization to global train-standardized values using the corresponding mean/std; hidden query payload is zeroed before arithmetic and masked again before key encoding. Clean lookback truth is transformed into the same coordinates, with invalid padded positions zeroed before the clean encoder. The batch replacement is local to this loss, leaving the main reconstruction and auxiliary paths in their original units. The existing all_floor adaptation also algebraically reconstructs the same global values.
- `src/malr/models/e23/historical_candidates.py:116-146`: inference intersects observation and validity masks, checks finite statistics, masks before arithmetic and encodes the global query through the unchanged frozen encoder. Empty-bank return, visible-evidence eligibility, causal index query, top-k, similarity threshold, candidate conversion and message path remain unchanged. Fully missing patches still lack historical eligibility. With a populated bank, changed queries may change retrieval by design.
- `integrations/e23/train.py:531-605`: existing bank staging reconstructs global raw lookback patches, selects only fully valid and completely originally observed patches, and retains them in raw global coordinates. Publication re-encodes every stored raw member with the EMA encoder on that same scale. Thus training query/clean, inference query and published bank keys now use matching numeric units.
- `integrations/e23/train.py:722-770`: global_aligned_v2 uses the original pair plus 0.5*contrastive/log(count) loss and declared retrieval weight; it receives no global_diverse variance/covariance additions and does not enter its per-step EMA branch. The new mode enters the single epoch-end EMA update and publication; frozen keys remain fixed within an epoch. Legacy and global_diverse code paths preserve their previous formulas and update schedule.
- The key loss reads no forecast target; changing forecast_target is independently covered below. This is specifically a key-loss isolation claim. The retained Stage1 auxiliary forecast objective intentionally uses training future labels through its separate existing loss and remains declared in the design; the entire Stage1 must not be described as future-label-free.
- `_trial_identity` explicitly binds global_aligned_v2, train_standardized coordinates, legacy_pair_and_contrastive loss, epoch EMA, legacy damaged-patch eligibility and zero extra parameters. Existing manifest and checkpoint identity loops check these fields. Legacy identity remains absent, and global_diverse retains its old metadata.

## Verification

Ran only the six allowed non-training tests on CPU, with CUDA disabled and no pytest/bytecode cache writes to source: **6 passed, 1 deselected in 1.62 seconds**. They verify exact old parameter/RNG/head initialization and empty-bank function, actual training query/clean inputs, actual frozen inference inputs, invariance across equivalent local coordinate frames, hidden NaN/positive Inf/negative Inf exclusion, and finite nonzero key gradients with forecast-target independence. The seventh EMA/bank/checkpoint test invokes training and was not run by this reviewer.

Inspected the seventh test and epoch/publication code for the training-only contracts. The retained full-suite log records 1187 passed, the disclosed nine inherited failures, and one skip. This review did not repeat the full suite or re-execute checkpoint training.

Reviewed changed-source SHA-256 values:

- `src/malr/models/e23/historical_candidates.py`: `6b8300a2508b3e5e1ba02a772ff541d2bae074e91195ce9f0675fad56bce6189`
- `src/malr/models/e23/model.py`: `88438e8739eb51057b6e463c429484b7d4028745ec9cfab3c7a5d486a7086e18`
- `integrations/e23/config.py`: `f9173ac03014a123dba2d3b416bad29ecfcefa10d17da01874d619e2c51b89eb`
- `integrations/e23/train.py`: `607d6342cfa59c91b914ba5d73e4b9d914b5cecff25b1d373375b376bd591f4b`

## Disposition and limits

The bounded source change is coherent and has no identified blocking issue. This preliminary source review does **not** approve formal training. Final parent/configuration inheritance must still follow v77 fixed validation; formal helpers, actual CUDA preflight, final source/controller/archive freeze and independent formal-run review remain required. Accuracy, convergence and any T1 advantage are not established by this coordinate-consistency change or the passing unit behaviors. Preserve existing trial artifacts and the unscored confirmation bank.
