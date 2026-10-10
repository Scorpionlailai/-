# Experiment ledger — plan: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v66_point_context_sr015_bl8_20261005/plan.json

Intent: exceed fixed T1 in ETTh1 96→96, sr015/bl8 (72.9252% actual), preserving original PatchTST.

Task 1: complete — existing point-context implementation, focused 13 tests passed; CPU tests and preflight gradient/head hash evidence recorded.
Task 2: in progress — full existing test suite before frozen source archive.
Task 3: pending — Stage1 and Stage2 100 epochs; MAE primary/MSE secondary locked before tests.
Task 4: pending — old20 tests, diagnostics, full resource CSV; fresh-mask confirmation if exceeds T1.

Ruling: existing isolated copies and artifact/hash ledgers are retained; git executable is unavailable, no git operations or new worktree required. Main model source and old frozen trials remain untouched.
Ruling: implementation predates newly available skills; preserve reviewed implementation, apply current verification and TDD to subsequent changes.

Task 2: complete — 51 current-path tests passed; entire suite 1128 pass/21 fail/1 skip. Same 21 failure names reproduced on inherited v67 sources (1129 pass). All failure names and logs retained in test-results.json.
Ruling: proceed with the verified current path; inherited missing old fixtures/git and obsolete historical assertions remain disclosed, with no model changes to force legacy tests green.
Task 3: in progress — source and plan freeze before measured 100+100 training.

Stage1 complete:100 epochs/6600 updates, best epoch87, global repairMSE0.28089139639236677. Stage2 started; head hash checked.
