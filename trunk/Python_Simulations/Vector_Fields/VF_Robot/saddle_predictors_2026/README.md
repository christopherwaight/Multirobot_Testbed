# saddle_predictors_2026: the reconstructed saddle field (saddle_nn), rebuilt 2026-11-16

Replaces `../saddle_predictors/` in `vecsim` (`saddle_nn`) and `validate_vecsim.py`. The old folder is unchanged; legacy scripts still load it.

## Why

The old net was trained on 4,000 points within ±0.65 m. It missed its own training data by 4 to 5° beyond 0.35 m (noise floor 2.5°), with a −3 to −10° bias. Its plane-fit critical point sat 2.8 cm from the saddle point the simulations scored as truth.

## How it was built

`nn_trainer_saddle_2026.py` (run from VF_Robot; about 35 s) writes the model and `build_report.json`.

1. **Data.**
   - 2025: `../saddle_predictors/saddle_data.csv`, in the old trainer's frame (x, −y).
   - 2026: all 110 saddle runs of `New_Hardware_Data/data_hardware_campaign_11_11_2026` (raw folder, logged variables only).
2. **Scale, computed from saturation** (S = a + b·d): slope 0.6439 /m in 2025 and 0.6417 /m in 2026, so 2026 lengths are multiplied by 0.9965. Same mat, same scale.
3. **Alignment, data to data:** 2026 Motive (x, z) is rotated 273.59°, mirrored and translated so each 2026 hue matches the 2025 hue map at the same mat point.
4. **Frame:** the old saddle_nn frame, shifted so the critical point (centre of the pooled saturation fit, (+0.0092, +0.0127) m in the old frame) sits at (0, 0).
5. **Held out:**
   - every 5th 2026 run;
   - redwood, which is never trained on;
   - the last 20% of the 2025 rows.
6. **Training data:** 1 cm cells (circular mean hue, mean saturation, readings more than 20° from their cell mean dropped).
   - Each cell uses 2026 readings where 2026 measured it, and 2025 readings only elsewhere: 7,900 cells from 2026, 500 from 2025, mostly corners.
   - The 2025 file mixes sessions about 3° apart and has blocks with 24 to 33% outliers.
7. **Nets:** hue (sin, cos) [32, 32, 16] tanh; saturation [32, 16] tanh. Adam, 6,000 full-batch epochs, cosine learning rate.

## Validation (median direction error, degrees)

| Held-out data | New | Old | Noise floor |
|---|---|---|---|
| 2026 runs, trained robots (18,966 readings) | 2.62 | 6.24 | 2.94 |
| 2026 redwood, unseen robot (15,050) | 3.09 | 4.27 | 3.07 |

- **2026 held-out runs:** no bias above 0.5° at any radius; saturation error 0.014 to 0.023 (old 0.026 to 0.040).
- **2025 held-out block:** dominated by that file's outlier blocks (both nets show biases of 10 to 27° there), so it is not a useful test. Beyond 0.65 m the old net does better on it (3.7 vs 5.2°), but those rows were in the old net's training data.

**Plane fit over a formation-sized ring (0.08 to 0.35 m), as the simulator sees it:**

| | New | Old |
|---|---|---|
| p* | (−0.0014, +0.0009) m | (−0.024, −0.015) m |
| det J | −0.58 (saddle) | −0.58 (saddle) |
| trace | −0.01 | +0.03 |

**Saturation at the centre:** 0.284 (old 0.342). Decoded magnitude, (S − 0.3)/0.7, is therefore slightly negative within about 2.5 cm of the centre. This matches what the robots read: 2026 intercept 0.27.

## Using 2026 hardware positions with this field

Map a 2026 Motive point p into this frame with the transform in `build_report.json`:

```
q = R(273.59°) · M · 0.9965 · (p − c26) + t − c
c26 = (0.8133, 0.7970)
M = diag(1, −1)
t = (+0.0099, +0.0127)
c = (+0.0092, +0.0127)
```
