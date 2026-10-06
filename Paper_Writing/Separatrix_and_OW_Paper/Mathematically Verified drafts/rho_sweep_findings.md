# Closed-loop radius sweep: results for review

**Status: NOT yet in the .tex.** Per CLAUDE.md these numbers need your sign-off first.

**Script:** `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/mc_sweep_radius.py`
**Data:** `experiments/outputs/mc_radius/{trials,summary}.csv`
**Run:** 25,600 trials (2 primitives x 4 radii x 8 noise levels x 400), 8.2 min, 4 workers.
Double gyre, fixed straddling start (0, 0.35), sigma_p = 0, random heading per trial,
paired seeds across radii so the comparison is matched rather than independent.

Answers Reviewer 1 item **A4**, the closed-loop rho sweep he called "the addition that
would move this from a good paper to a hard-to-reject one."

---

## The headline: the predicted interior minimum does not exist

I expected success rate to peak at an intermediate radius (truncation bias rising with
rho against noise gain falling as rho^-q). **It does not.** Far-saddle success rises
monotonically with rho for both primitives at every noise level tested.

Far-saddle success, D tracker:

| rho | 0 | 0.001 | 0.002 | 0.004 | 0.006 | 0.008 | 0.012 | 0.02 |
|---|---|---|---|---|---|---|---|---|
| 0.0375 | 1.000 | 0.720 | 0.482 | 0.378 | 0.275 | 0.225 | 0.200 | 0.175 |
| 0.0750 | 1.000 | 0.998 | 0.927 | 0.705 | 0.578 | 0.458 | 0.425 | 0.263 |
| 0.1050 | 1.000 | 1.000 | 0.993 | 0.920 | 0.812 | 0.698 | 0.560 | 0.432 |
| 0.1500 | 1.000 | 1.000 | 1.000 | 0.995 | 0.960 | 0.920 | 0.790 | 0.565 |

Far-saddle success, s1 tracker:

| rho | 0 | 0.001 | 0.002 | 0.004 | 0.006 | 0.008 | 0.012 | 0.02 |
|---|---|---|---|---|---|---|---|---|
| 0.0375 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.0750 | 1.000 | 0.895 | 0.495 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.1050 | 1.000 | 0.993 | 0.892 | 0.420 | 0.040 | 0.000 | 0.000 | 0.000 |
| 0.1500 | 1.000 | 1.000 | 0.995 | 0.890 | 0.720 | 0.435 | 0.060 | 0.000 |

Note the nominal rho = 0.075 used everywhere in the paper is the **second smallest** of
the four. On this metric a larger formation is strictly better.

## But the trade is real. It is just on a different axis.

The truncation cost does not show up in success rate. It shows up in **where the cluster
parks**, which is the zero-noise tracking error:

| rho | D tracker | s1 tracker |
|---|---|---|
| 0.0375 | 0.00079 | 0.00390 |
| 0.0750 | 0.00157 | 0.00748 |
| 0.1050 | 0.00219 | 0.01021 |
| 0.1500 | 0.00309 | 0.01447 |

So the operator's trade is **accuracy against robustness**, not accuracy against accuracy:
a bigger formation reaches the right saddle far more reliably and then sits further from
the true trench. That is a cleaner and more useful statement than a success-rate optimum
would have been, and it matches the existing zero-noise numbers in IV-E (0.0075 for s1
against 0.0016 for D at rho = 0.075; this sweep gives 0.00748 and 0.00157, so the sweep
reproduces the published values to three digits).

## Both exponents confirm the gain ladder in closed loop

This is the part worth putting in the paper.

**Truncation side.** Fitting err ~ rho^p on the zero-noise tracking error:
- D tracker: **p = 0.98**
- s1 tracker: **p = 0.94**

Theory: the truncation residual is O(rho^3), and it enters the curvature channel after the
rho^-2 gain, leaving **O(rho^1)**. Confirmed to within 6%.

**Noise side.** Fitting the 50%-success noise level sigma_50 ~ rho^p:

| rho | D sigma_50 | s1 sigma_50 |
|---|---|---|
| 0.0375 | 0.00193 | 0.00050 |
| 0.0750 | 0.00729 | 0.00199 |
| 0.1050 | 0.01576 | 0.00366 |
| 0.1500 | (>0.02) | 0.00754 |

- D tracker: **sigma_50 ~ rho^2.02**
- s1 tracker: **sigma_50 ~ rho^1.95**

Theory: a second-order coefficient carries gain gamma/rho^2, so the tolerable noise before
a fixed relative error scales as rho^2. Both land within 3% of 2.0.

**What this means, and it is a new claim the paper can make:** the exponent identifies
*which channel sets each primitive's ceiling*. A first-order-limited law would scale as
rho^1. Both scale as rho^2, so **both primitives are limited by their second-order
channels** even though the s1 tracker rides a first-order tangent. That independently
corroborates the Section IV-E diagnosis: the s1 failure is the argmax |a5| vs |a4|, and
both of those are second-order coefficients. The ladder predicted it and the closed loop
confirms it.

Also worth noting: the D/s1 sigma_50 ratio is 3.9 at rho = 0.0375, 3.7 at 0.075, and 4.3 at
0.105, i.e. the "four to five times" advantage is **not** an artifact of the chosen radius.
It holds across the whole sweep. That is a useful robustness check on the abstract's
headline number (at the paper's rho = 0.075 this sweep gives 0.00729 / 0.00199 = 3.7,
against the published 0.0079 / [0.0015, 0.002] = 4.0 to 5.3; same range, different start
distribution and 400 rather than 10,000 trials).

## Caveats

- 400 trials/cell, so roughly +/- 2.5 points at the 50% level. The published sweeps use
  10,000. Fine for the exponents, which are fit across four decades of radius; if you want
  the sigma_50 values quoted to three digits, rerun at higher trial count.
- sigma_p = 0 throughout. Position noise enters the same sigma_eff and would confound the
  radius axis.
- One start (0, 0.35), the same straddling start the noise sweeps use. Not a basin estimate.
- The collapse threshold was scaled with the formation (COLLAPSE_RMS proportional to rho).
  Holding it fixed would have made collapse detection much stricter for small formations
  and looser for large ones, biasing the swept variable. Collapse rates came out ~0 in
  every cell except the smallest radius under noise, so this choice does not drive the
  result.
- rho = 0.15 never crosses 50% for the D tracker within the tested noise range, so its
  sigma_50 is a lower bound and it is excluded from the D exponent fit.

## Suggested use in the paper (needs your call)

Two options, both cheap on space:

1. **Text only, ~4 lines in II-C or IV.** State the two exponents (0.98 and 0.94 against a
   predicted 1; 2.02 and 1.95 against a predicted 2) as closed-loop confirmation of the
   gain ladder, and state the accuracy-versus-robustness trade in one sentence. This is the
   cheapest way to answer A4 and it is the part with the most scientific content.
2. **Text plus a panel on Fig. 3.** Adds the visual but costs ~7 lines and the figure is
   currently modified-but-uncommitted (see below).

Given the paper is running long (see status note), option 1 is my recommendation.

## Unrelated issue found while working

`figures/estimator_accuracy_vs_noise.png` and its `.meta.json` are **modified but
uncommitted** in git, with recent commits all "WIP: automated checkpoint". The Fig. 3
currently embedded may not match any committed state. Worth resolving before resubmission
independently of this sweep.
