# E23/E24 variable scaling and runtime audit

## Recorded training costs

Comparable original v79 recipe: 100 Stage1 epochs + 100 frozen-repair pure-MSE Stage2 epochs, batch128, FP32, RTX5090. ETTh1:7 variables,6600 optimizer updates per stage; Weather:21 variables,28600 per stage. Weather uses all validation/test windows. These are different datasets and GPUs; this is not a controlled variable-count scaling experiment.

| Dataset | Stage1 full command seconds | Stage2 full command seconds | Total seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|
| ETTh1 v79 sr010_bl8 | 561.082 | 278.374 | 839.455 | 4.513 |
| Weather E24-v79 sr010_bl8 | 7693.647 | 2781.591 | 10475.239 | 20.062 |

Weather is 3 times the variables and 4.33 times the optimizer updates. Full training is about12.48 times slower. Average recorded Stage1 seconds/update increases about3.16 times; Stage2 about2.31 times; those quantities include preparation/validation/checkpoint overhead and are not pure kernel timing.

v97 currently additionally trains100 refinement epochs after a selected v79 trained100 epochs. Its Stage1 budget is200 epochs, followed by100 Stage2 epochs. Current v97 dense code has Linear(35,64),Linear(64,7), fixed96 positions; it is an ETTh1 implementation and requires variable-count adaptation before Weather use.

## Read-only Weather checkpoint profiling

GPU0; existing Weather v79 Stage1 selected checkpoint; three fixed validation input batches128; no optimizer updates and no test metrics. Existing E24 source/run/GPU1 processes unchanged. Scripts and raw results in `/home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008`.

Repair forward average59.38 ms/batch in the first probe. Selected component CUDA elapsed times (not an exhaustive additive decomposition):

- Point candidate eight-head readout:22.36 ms.
- Window context CNN/signature:12.27 ms.
- Historical candidate lookup:2.69 ms.
- Variable source selection:1.19 ms.
- Recent/global mean projector:0.91 ms.

Thus historical lookup was not the leading bottleneck in these Weather validation batches. This does not establish the bottleneck for hundreds of variables or all training batches.

## Probes

1. Release gathered historical keys after reduction, gather raw patches only for selected members; FP64 and stable full sort retained. Three batch outputs exactly equal.59.38 vs59.35 ms forward, no meaningful speed benefit; do not promote as a training speed improvement.
2. Evaluate point candidate query/output, point routing and overlap-confidence MLPs only at hidden valid points. Boolean compaction:174.39 vs176.12 ms forward/backward;10.296 vs9.410 GiB allocated. No speed improvement.
3. Same point compaction with shared precomputed indices/index_select/index_copy:174.32 vs171.93 ms median forward/backward;10.296 vs9.487 GiB allocated. Outputs/loss exactly equal on checked input; gradient maxabs5.59e-9, relativeL2 difference1.57e-7. About1.4% faster and7.9% lower peak on this probe, insufficient to claim a material end-to-end speed improvement.

Forward/backward probes use repair eval mode (dropout disabled) and hidden-coordinate train-standardized MSE. They exclude retrieval loss, bank refresh, optimizer, validation loop, reporting and data generation. Only four measured steps after one warmup; do not extrapolate directly to full Stage1 training or other missing rates. Observed-coordinate candidate-attention diagnostics can differ with sparse evaluation; reconstruction and loss gradients were checked. No official training runner/source was modified or rerun.

## Priorities

- Stage2 fixed validation inputs can cache frozen repair outputs once. Training masks regenerate each epoch, so one shared repaired training cache would change the protocol and must not be silently introduced.
- Profile a complete Stage1 step before claiming kernel-level speed improvements. Test BF16/SDPA/compilation in a separate recipe and report metric deviations; precision or implementation changes are not assumed bit-exact.
- For larger C, contextual V still builds B*N*C*C*P pairwise visibility/error tensors before top4. Limit local scoring to a larger fixed shortlist before top4, checking recall/repair/forecast accuracy; this changes candidate selection and needs an experiment.
- Window signature cross-variable processing and pointwise MLP activations deserve first attention for Weather. Preserve eight heads until a compression ablation shows it can retain quality.
- To reduce v97 total training cost, evaluate a compact repair student trained from the v97 teacher using history-only supervision/distillation. Account for teacher construction and any cache costs explicitly; this is a new experiment, not an existing result.
