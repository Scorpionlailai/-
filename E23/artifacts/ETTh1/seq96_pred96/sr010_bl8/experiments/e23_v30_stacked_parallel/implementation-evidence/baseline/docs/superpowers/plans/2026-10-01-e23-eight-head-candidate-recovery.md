# E23 eight-head candidate recovery implementation plan

1. Add typed candidate evidence and opt-in eight-head router; write failing unit tests for candidate attention, masking, gradients, and head independence first.
2. Expose H/V/T candidate tensors without repeating selection; connect opt-in router to repair encoder. Test end-to-end repair and unchanged observed coordinates.
3. Add strict config schema and matched 128-dimensional single-head and eight-head YAML files. Test config differences and builder wiring; verify old checkpoint strict load.
4. Run focused unit/integration tests and a one-batch backward pass. Fix observed failures.
5. Train both trials with locked ETTh1 P4 protocol, 100+100 epochs and matched selection. Replay selected validation checkpoints, apply v12 MAE/MSE gate, and test only a promoted model. Record results, head diagnostics, and limitations in the experiment ledger.

Work remains in the current E23 checkout because the relevant preceding experiments and integration code are present there. Do not alter the user-edited E22 results CSV.
