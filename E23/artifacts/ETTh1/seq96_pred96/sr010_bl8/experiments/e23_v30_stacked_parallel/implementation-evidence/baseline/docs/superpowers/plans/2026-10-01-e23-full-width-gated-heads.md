# E23 v25 full-width gated heads plan

1. Write failing unit tests for eight 64-dimensional proposals, learned gate normalization, gradients, and exact fallback.
2. Extend the candidate router with an opt-in gated-sum fusion mode while leaving concatenation checkpoints unchanged; add diagnostic fields.
3. Add strict v25 config and builder wiring; test width/policy constraints and a full repair/forecast forward pass.
4. Run focused and E23 regression tests, then train fresh Stage 1 and Stage 2 for 100 epochs each. Replay selected validation metrics and apply the unchanged v12 promotion gate.
5. Run head diagnostics, test only a promoted selection, and write an experiment record. Preserve v12 as selected otherwise.
