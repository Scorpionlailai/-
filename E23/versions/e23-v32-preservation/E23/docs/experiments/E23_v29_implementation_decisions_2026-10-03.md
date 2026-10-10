# E23 v29 实施取舍与审查记录

- Ruling: Continue in approved /home/xyw/paper/E23 on paper/e23-migration, not main; snapshot existing dirty baseline instead of creating a checkout from HEAD, which lacks previous uncommitted E23 work. Cost if wrong: this run shares the user workspace; preserve user files and review against saved baseline.

- Ruling: Cache dataclasses live with each expert; exposure moves to a small module to avoid model/controller circular imports. Cost if wrong: strict legacy-import regression catches API breakage.

- Ruling: Edited pre-existing dirty files are not blindly committed; use baseline snapshot and preserve incremental patch evidence. Cost if wrong: git history alone is incomplete, so artifact snapshots and ledger retain the exact run.

- Ruling: H/T legacy forwards now factor through their cache builders with unchanged non-contextual math, instead of duplicating the candidate extraction. Cost if wrong: numerical compatibility; unit regressions passed and full v28 replay is required before training.

- Task 5: Ruling: Add source_context_allowed to T/V cache readers; specialist-selected targets use original encoded/raw sources instead of mixed iterative source states. A RED test demonstrated T leaking through another variable into a V-only target. Cost if wrong: restricting 20% training targets can reduce contextual training signal, but preserves the specialist contract.

- Task 5: Ruling: New mode masks hidden values before base encoding, while legacy mode is unchanged; formal P4 already supplies zero-masked observations. Cost if wrong: custom callers depending on hidden inputs intentionally lose that information; hidden-input invariance and padding-gradient tests pin the intended behavior.

- Ruling: Implement Task8 selection/report and validation-only round diagnostics before Task7 final review and source freeze; the plan places script implementation after review. This keeps all runnable code covered by the single fresh review and immutable training hashes. Cost if wrong: task order differs, but no training/test protocol changes.

- Final: Ruling: Regrade missing report diagnostic identity from Minor to Important; a stale diagnostic can misattribute scientific metrics. Fix before source freeze. Cost if wrong: extra guard and focused tests, no architecture/protocol change.

- Final: Ruling: Reviewer deferred formal accuracy/cost judgment until training; measure both full experiments and report actual outcome. Cost if wrong: no improvement claim before evidence.

- Final: Ruling: Reviewer excluded unrelated E22 CSV; retain its baseline bytes and SHA. Cost if wrong: pre-existing CRLF diff warnings remain.

- Final: Ruling: Reviewer deferred multi-seed and alternative parallel-round architecture; retain approved single training seed and label exploratory results. Cost if wrong: general stability and isolated serial-order contribution remain unproven.

Deferred minors: none; diagnostic identity finding was regraded Important and fixed with nine RED→GREEN tests.

User steering: requested additional tests after failed gate; both models evaluated on original three test draws and original validation-selected weights. Original validation selection retained.

Evidence archive: `/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v29_iterative_tv_h/implementation-evidence`.
