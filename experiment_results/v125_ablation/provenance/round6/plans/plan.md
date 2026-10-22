# A5质量常量复验 Implementation Plan

> Execute inline with executing-plans; preserve prior assets.

Goal: Evaluate originalA5 explicitfusionquality contribution without redefining the model. Architecture: isolatedround6 diagnostics/adapted exporter reuse immutablecoarse/calibrationworkers. Techstack: originalPython/PyTorch/GPUenvironment. Spec: spec.md.

GlobalConstraints: 3seeds; original100+100+100/batch128/loss/selectors/masks; fixed1:1; nootherstages.

ReviewFocus: internalquality remainsactive; rawcandidatevalues/weights/rankingnotzeroed; allqualityconsumerscovered; native/missingmasksandnullprotectionunchanged; priorresultsimmutable.

- [ ] Task1: policytestsRED→GREEN; sourcequalityfeatureinventory; realruntimequalityhooks, rawcandidatepreservation and expertcachefixedinputflaginvariance; projectedquality/candidatequalitypoisoninvariance; masks/null/nohiddentruthchecks; mainMAE/MSEgradientvalidity; asset2023audit. Files: audit6.py,quality6.py,policy6.py,test_policy6.py,quality_features.json/md.
- [ ] Task2: unchangedoriginalcoarse/calworkers; diagnosticqualitymonitor onallformaltrain/eval; controlledinit/order/masks. Filesrun6/supervisor6/verify6/export6. Independentreview, locksource/audit, archiveallbaselinesandpreviousreports; publishregistration.
- [ ] Task3: launchtwoGPUoriginalbudgets, verifyfirstepochpairedinputs. Auto100coarse→own100cal64+100cal256→lock9checkpoints→val/test20/allstrata→sameGPUresources→finalverification→publishreport/commit; noexpansion.
