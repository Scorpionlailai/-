# SDD ledger — plan: docs/superpowers/plans/2026-10-03-e23-stacked-parallel-tri-expert.md

Ruling: Work in approved /home/xyw/paper/E23 on paper/e23-migration with hash-verified dirty baseline snapshot; HEAD alone lacks prior E23 work — preserve authorized workspace and exact baseline — cost if wrong: shared workspace, protect user files and retain incremental patches.
Pre-flight: Task1 provider/cache/message -> Task2 grouped encoders: matching APIs.
Pre-flight: Task2 TVHGroup/grouped_tv_h -> Task4 controller: matching ownership and cache arguments.
Pre-flight: Task3 fusion -> Task4 layer control: matching group/delta shapes.
Pre-flight: Task4 StackLayerDiagnostics -> Tasks5/6 model/diagnostics: matching stack_layers field.
Pre-flight: Task5 identity/config -> Tasks6/7/8 loaders/reports: matching l/g/checkpointing fields.
Pre-flight: Task6 runner/report -> Tasks7/8/9 freeze/train/report: scripts built before freeze.
Task 1: in progress; base 4cf3cd1642d2a1141ed8e73577b02c1b889650c3
Task 1: complete (commits 4cf3cd1..cb33977, tests: python -m pytest tests/unit/test_e23_historical_provider.py tests/unit/test_e23_historical.py tests/unit/test_e23_historical_cache.py tests/integration/test_e23_model.py -q → 25 passed in 1.40s)
Task 2: complete (commits cb33977..a880844, tests: python -m pytest tests/unit/test_e23_group_ops.py tests/unit/test_e23_grouped_experts.py tests/unit/test_e23_temporal.py tests/unit/test_e23_variable.py tests/unit/test_e23_router.py tests/unit/test_e23_iterative.py tests/unit/test_e23_variable_context.py tests/unit/test_e23_temporal_context.py -q → 57 passed in 1.22s)
Task 3: complete (commits a880844..761e78b, tests: python -m pytest tests/unit/test_e23_stacked_fusion.py -q → 9 passed in 0.85s)
Task 4: complete (commits 761e78b..8b30634, tests: python -m pytest tests/unit/test_e23_stacked.py tests/unit/test_e23_grouped_experts.py tests/unit/test_e23_iterative.py -q → 27 passed in 1.42s)
Task 5: complete (commits 8b30634..f2812fa, tests: python -m pytest tests/integration/test_e23_stacked_model.py tests/integration/test_e23_stacked_training.py tests/integration/test_e23_config.py tests/integration/test_e23_stage1.py tests/integration/test_e23_stage2.py tests/integration/test_e23_iterative_training.py tests/integration/test_e23_global_repair_objective.py -q → 82 passed in 6.47s)
Task 6: complete (commits f2812fa..12d674e, tests: python -m pytest tests/integration/test_e23_v30_reporting.py -q → 18 passed in 1.90s)
Ruling: Lifecycle skill package was removed from the active environment after Task6; use existing approved plan and directly record equivalent briefs/tests/BASE/completion in the same ledger — no need to abandon authorized implementation — cost if wrong: bookkeeping differs from unavailable helper scripts.
Ruling: Full CUDA mapped checkpoint comparisons retain absolute 1e-6 MSE/MAE replay gates; FP32 pointwise forecasts allow 1e-5 and report actual maximum difference because batched GEMM reductions alter rounding (observed 1.9e-6), with independent float64 kernel outputs/gradients still within 1e-10 — cost if wrong: pointwise difference could conceal a small issue, mitigated by exact reference math and full validation gates.
Ruling: Re-grade review NaN summaries and winner/gate conflation as Important, and fix with test provenance in one pass — both can materially misstate paper results — cost if wrong: small additional reporting test effort.
Final: Ruling: Reviewer excluded independent CUDA measurement and pending replay — executor verified all three real CUDA profiles plus strict legacy loads and full validation replay — cost if wrong: independent review did not duplicate GPU evidence.
Final: Ruling: Reviewer excluded future full experiments — Tasks8/9 must verify all fresh100/100 results and final artifacts before completion — cost if wrong: later experimental evidence may reveal issues beyond static review.
Final: Ruling: Reviewer excluded unrelated preexisting dirty work — preserve baseline and review only authorized v30 changes — cost if wrong: preexisting issues remain outside this iteration.
Final: fixed test protocol/selection/mask provenance — eight rejection cases RED→GREEN, suite947passed/1skipped.
Final: fixed NaN summaries — two nonfinite rejection cases RED→GREEN, suite947passed/1skipped.
Final: fixed qualifying gate versus selected winner — multiple-passing-candidate test RED→GREEN, suite947passed/1skipped.
Task 7: complete (base12d674e; CUDA128/FP32 all3 passed; real old weights strict-loaded; v29 metrics exactly replayed; mapped maxpoint2.6226e-6; independent review findings fixed; full suite947passed/1skipped; incremental whitespace clean; freeze 359 files at 1eee2829be14b44d3d6118c9afc8fc459744b02a).
Task 8: in progress; base 1eee2829be14b44d3d6118c9afc8fc459744b02a; full matrix100/100 on GPU1, no source changes until all runs/tests finish.
Ruling: User explicitly requests per-model immediate three-draw testing; use frozen evaluator directly after each own lock/replay, on idleGPU0 while sequential training continuesGPU1; original all-three test gate/order superseded; never alter later configs from these tests — cost if wrong: test timings/hardware concurrency differ, record separately and exclude from training costs.
Ruling: User requests training progress checks every five minutes; adopt that waiting cadence and retain immediate per-model tests after observed completion — no training configuration or selection change.
Task 8: complete (three fresh Stage1/2 100/100 trials on GPU1; own paired manifests/resources/checkpoints; all3 fixed-validation replays and layer/group diagnostics; per-model immediate frozen three-draw tests on GPU0; all359 freeze hashes and all3 test bank files verified).
Task 9: complete (runner --mode report; final_audit.py passes seven comparison rows, nine test rows, eighteen layer rows with exact JSON means/std/resources/checkpoints; all3 gates fail, retain E23_v29_r1; report/plan/final-status and hash-verified execution-ledger archive completed).
Final: No deferred review findings; three Important reporting/provenance findings fixed in one pass. Rulings retained above; scope exclusions independently verified or preserved. Source freeze intact through every training, replay, diagnostic, test and final report.
