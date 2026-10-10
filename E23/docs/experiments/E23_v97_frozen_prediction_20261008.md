# E23-v97 frozen-repair prediction

ETTh1 history96/horizon96; seed2023; Stage2 100 epochs, batch128, Adam lr0.0001, pure MSE; complete repairer frozen; original PatchTST unchanged. Main checkpoint validation MAE; secondary validation MSE. Same 20 development masks as T1 and v79; one training seed. GPU timing and memory below refer only to this new Stage2, not total training cost.

| Scene | v97 MSE / MAE | T1 MSE / MAE | v79 MSE / MAE | Stage2 seconds | GPU allocated / reserved GiB |
|---|---|---|---|---|---|
| sr015_bl8 | 0.398613 / 0.407094 | 0.408589 / 0.414075 | 0.405380 / 0.411311 | 290.64 | 1.614 / 1.977 |
| sr010_bl16 | 0.569030 / 0.499734 | 0.579435 / 0.500068 | 0.555073 / 0.491735 | 285.06 | 1.613 / 1.977 |
| sr020_bl8 | 0.483583 / 0.454429 | 0.491737 / 0.456650 | 0.475542 / 0.449796 | 299.85 | 1.613 / 1.977 |
| sr010_bl8 | 0.383287 / 0.397058 | 0.386992 / 0.400414 | 0.385919 / 0.398839 | 287.84 | 1.614 / 1.977 |
