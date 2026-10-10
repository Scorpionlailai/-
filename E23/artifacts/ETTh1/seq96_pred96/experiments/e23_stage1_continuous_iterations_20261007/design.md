# Stage1 continuous iterations

V80 ablation on fixed validation shows the added band branch worsens repair MSE in all three scenes, while helping its already-trained predictor. The relation between mean supervision, pointwise error, and learned prediction input therefore needs a controlled Stage1 test.

Queue order: v81 retains v80 and weakens recent mean weight 1 -> .25; v82 retains v80 and removes recent mean loss; v83 applies .25 to original v79 architecture; v84 removes that loss from original v79. Other Stage1 data/seed/epochs/optimizer settings remain fixed. No auxiliary forecast in Stage1. Every Stage2 uses pure MSE, all repair parameters/buffers/memory frozen, original head architecture and initial SHA, identical training settings. Each model runs all three scenes and tests old20 after validation locks. All positive/negative results retained with timing and VRAM.

After each candidate, verify the result and continue automatically unless all scenes beat T1 in repair MSE/MAE and improve v79 forecasting MSE/MAE. These are development criteria, not proof of universal optimality. A successful candidate still needs new unexposed shared-mask confirmation. Logs are inspected every five minutes. A failed experiment is reported as failed, never completed.
