# Tracker rewrite: Primitives 7 and 11 (2026-10-08 and 09)

The Separatrix paper's two trackers were rewritten so that every test they use
is set by the formation's own size rather than tuned to a field. The s1 tracker
is Primitive 11, `oecs_separatrix_step` (Draft_12 Sec. III-C). The D tracker is
Primitive 7, `separatrix_logic_c_step` (Sec. III-B). Both live in
`src/control/pentagon_primitives.py`, and each previous version is kept there,
commented out. This folder holds the design notes, the evidence for each
choice, and the rerun drivers.

## Current design

rho below is the formation's mean ring radius, measured from the robot
positions each step. Neither tracker needs any other length scale.

### s1 tracker (Primitive 11), two modes

```
q <- rho |grad s1| / |s1|                     # fraction s1 changes across the footprint
if captured or (s1 < 0 and q < c_in):         # CAPTURE, latched, never released
    descend s1:  -k sat(g_perp grad s1)
else:                                         # TRACK
    t <- whichever of e1, e2 has the larger |grad s1 . e|
    sign t against the previous t (against the measured flow on the first step)
    command k [c_max tanh(1) t - sat(g_perp P grad s1)],  P = I - t t^T
```

One tracker parameter, c_in = 0.03 (q_cap in the paper). Removed: ACQUIRE,
PARK, CROSS, the release, and s_trim, r_band, g_capture, s_capture.

### D tracker (Primitive 7), three modes (2026-10-09 evening)

```
w1, w2 <- eigenvectors of H_D, lambda1 <= lambda2
if captured or (det J0 < 0 and |J0^-1 v0| < rho):         CAPTURE, latched: k sat(-J0^-1 v0)
elif lambda1 lambda2 < 0 and |H_D^-1 g0| < c_crest rho:    FLOW: k [sat(v0.w1) w1 + sat(-w2.g/lambda2) w2]
else:                                                       TRACK: orient w1 so v0.w1 >= 0,
                                                            k [sat(|w1.g|/|lambda1|) w1 + sat(-w2.g/lambda2) w2]
```

The capture test and the hold are the critical-point test and attraction law
of the 3-robot paper (cite [10]). Both read the flow, so neither is
objective, which the D tracker already was not. TRACK is the earlier SLIDE
and ATTRACT, which differed only in the sign of the along-trench step, and it
replaces ATTRACT_FALLBACK where H_D is definite. The across step keeps the
signed lambda2 (Chris's choice, see "Three-mode screen" below). Removed over
both rewrites: the FLOW band and eps_raw, eps_dim, FLOW_DRIFT, the
ATTRACT_FALLBACK mode, the |lambda| descent at capture, and the experimental
`guard='transverse'`. The five-mode version of the 2026-10-09 afternoon is
kept commented out in the file, and a copy of the whole file before the change
is `overnight/pentagon_primitives_before_three_mode.py`.

### Compatibility

Callers that still pass the removed thresholds (s_trim, r_band, g_capture,
s_capture, eps_raw, eps_dim) run unchanged, and the values are ignored. The
removed diagnostic arguments (override_seed, override_argmax, margin_gate,
sigma_eff, rho, capture, guard='transverse') raise. Two one-off scripts used
them, `experiments/margin_gate_sweep.py` and
`experiments/hop_rate_instrumentation.py`. Both now say so in their header,
and say how to run them against `overnight/pentagon_primitives_pre_rewrite.py`.
`separatrix_logic_park_step` and Primitive 10 are unchanged.
`tests/test_oecs_separatrix.py` passes, including a start off the trench at
(-0.45, 0.30).

## Status

| Item | State |
|---|---|
| Code, both trackers | done, verified bit for bit against the tested variants |
| Paper method text (III-B, III-C, Table I, latch sentences, Limitations) | done, compiles clean, no result numbers changed |
| s1 reruns | done 2026-10-09 02:45, `overnight/run.log` |
| D reruns, then Fig. 7 with both trackers | done 2026-10-09 15:54, `overnight/run_d.log` |
| Open checks (1000-step runaway, starts off the trench) | done at sigma 0.005, `open_checks.log`. Runaway at sigma 0.002 running, `runaway_002.log` |
| Result text, numbers, and Figs. 5 to 9 in the paper | applied 2026-10-09, compiles clean. Old figures in `overnight/backup_paper_figures_before_install/`, old draft in `overnight/backup_after_s1_rerun/Draft_12_before_results_batch.tex` |
| Three-mode D tracker installed | done 2026-10-09 23:20, identical to the screened variant on 63/63 runs, unit tests pass |
| Three-mode D rerun, every D result | done 2026-10-10 01:37, `overnight/run_d3.log`, outputs tagged `_d3`, five-mode outputs in `overnight/backup_before_three_mode/` |
| Three-mode .tex batch and Figs. 5 to 9 | applied 2026-10-11, compiles clean with no new warnings, 12 pages (was 11; references [35] to [37] spill onto p. 12). Old draft and figures in `overnight/backup_before_three_mode/` |

Figs. 5 to 9 were installed into the paper's `figures/` on 2026-10-09. Fig. 7
comes from the contact-based `noise_both_S*_final.csv` set via
`overnight/fig7_regen.py`. `plot_flip_resolution_starts.CSV_FILES` still names
the old CSVs. Clean runs (Figs. 5, 6) now stop once a saddle has been held 20
steps, with the exit at |y| > 0.60 (`main_separatrix_traverse.py`). Fig. 7
keeps contact as success, and the paper states that contact and a 20-step
hold agreed in 1259 of 1260 checked trials (`hold_k_check.py`).

## Hold drift under noise: fixed for D, open for s1

Over 1000 steps at sigma_uv or sigma_p 0.005 (30 trials x 7 starts per
tracker), the five-mode D tracker and the s1 tracker kept only 9 to 12 of 210
trials at a saddle, with a median drift of 0.45 to 0.69 after capture.
Descending a noisy gradient at the saddle is a random walk at up to full
speed. Two fixes, both in place:

- Scoring. A trial ends once a saddle has been held 20 steps (clean runs), or
  at contact (Fig. 7), which agree in 1259 of 1260 checked trials. This is
  memoryless, needs no dead band, and needs no release.
- D tracker only. The three-mode CAPTURE holds with the Newton step to the
  fitted stagnation point, from zeroth- and first-order coefficients. Now
  198 and 207 of 210 trials sit at a saddle at step 1000 (the rest never
  captured), median drift 0.063 to 0.065, which is the distance from where
  capture fires to the stagnation point.

The s1 tracker still drifts (9 and 11 of 210, median 0.69 and 0.54). The
Newton hold would cost it its objectivity. The paper's s1 claims are scored
by the hold rule, so no number depends on it.

Starts off the trench (noise-free): every tracker acquires and holds some
saddle with no ACQUIRE, though from a gyre centre or the wall it is often a
corner saddle rather than p1*.

## Three-mode screen (2026-10-09 evening)

`exploration/dnew.py` holds the scratch tracker, with switches for the
capture action (Newton or |lambda| descent), the fallback (kept or merged),
and signed or |lambda2| across. Drivers are `exploration/screen_clean.py`,
`screen_noise.py` (log `screen_noise.log`), `screen_sbc.py` (map
`maps/d_tracker_newton_sbc.png`), and `screen_sbc_ridge.py` (log
`screen_sbc_ridge.log`).

Merging SLIDE and ATTRACT alone (fallback and old capture kept) reproduces the
five-mode tracker bit for bit on all seven clean starts. ATTRACT_FALLBACK never
fired on the clean runs, fired on 37 of 168 Santa Barbara steps (34 positive,
3 negative definite), and fired in 77 to 98% of noisy double-gyre trials.

Far saddle held 20 steps, 350 trials (7 starts x 50, random heading):

| Variant | sigma_uv 0.002 | sigma_uv 0.005 | sigma_p 0.005 |
|---|---|---|---|
| five-mode | 346 | 281 | 288 |
| Newton capture, fallback kept | 346 | 281 | 288 |
| abs(lambda) descent capture, fallback merged, abs(lambda2) | 341 | 275 | 275 |
| Newton, merged, abs(lambda2) | 341 | 275 | 275 |
| **Newton, merged, signed lambda2 (adopted)** | **339** | **250** | **267** |

The Newton capture costs nothing. Merging the fallback costs the rest. Signed
lambda2 costs more under noise, and Chris accepted that for Santa Barbara:

| Santa Barbara, paper start | End | Mean ridge distance | Within 1 km |
|---|---|---|---|
| pre-rewrite D (published) | landfall 24.0 h | 1.49 km | 33% |
| **three-mode, signed lambda2** | landfall 23.0 h | 1.60 km | 41% |
| s1 | at sea 28 h | 2.10 km | 23% |
| three-mode, abs(lambda2) | at sea 28 h | 2.62 km | 25% |
| five-mode | at sea 28 h | 2.75 km | 17% |

From 16 starts within 1 km of the paper's, signed lambda2 reaches south of
34.12 N in 14 (|lambda2| 7) with a mean ridge distance of 1.66 km (2.24 km).
The signed path stays on the s1 tracker's branch. It does not reproduce the
published pre-rewrite D path, which landed at (34.039 N, 120.277 W).

## Three-mode rerun results (2026-10-10, `overnight/run_d3.log`)

Unchanged by the rewrite: noise-free control 10^4 of 10^4 at every start, the
rotating-frame numbers (max |x| 0.084 against 0.010 inertial, straddle lost
40 steps, no capture in the rotating frame, first loss at Omega 0.205), and
contact against the 20-step hold (all 630 D trials agree, 1259 of 1260 over
both trackers). Clean runs end at (0.000, -0.504 to -0.506), 0.004 to 0.006
from p1*, against 0.009 to 0.014 for the five-mode tracker.

Fig. 7, D tracker (`overnight/figures_for_review_d3/noise_seven_starts_col.png`),
retention level at 50% as a multiple of the s1 tracker's:

| Start | sigma_uv, five-mode | sigma_uv, three-mode | sigma_p, five-mode | sigma_p, three-mode |
|---|---|---|---|---|
| S1 | 5.3 | 5.4 | 4.1 | 4.6 |
| S2 | 5.5 | 5.8 | 4.1 | 4.7 |
| S3 | 5.6 | 5.7 | 4.7 | 5.3 |
| S4 | 3.8 | 4.2 | 2.5 | 2.9 |
| S5 | 2.6 | never below 50% | 2.3 | never below 50% |
| S6 | 1.4 | 1.2 | 1.5 | 1.4 |
| S7 | 1.3 | 1.1 | 1.4 | 1.3 |

Success falls (S1 at sigma_uv 0.005, 84% to 79%; S6, 66% to 48%) while
retention barely moves above the origin. The retention curves at S6 and S7
cross the s1 tracker's near sigma_uv 0.006 (was 0.0075). At sigma_uv 0.01 no D
retention exceeds 65% (was 61%). Success minus retention up to sigma_uv 0.005
is at most 6 points (was 10). S3 retention runs from 99.4% at 0.0015 to 23% at
0.03 (was 99.5% to 13%). S5 success rises again above sigma 0.0125, to 81% at
sigma_uv 0.03, as noise random-walks the team onto the saddle 0.25 below.

Sec. IV-C early firing (`outputs/mc_mechanism/d_fallback.csv`, definite-H
steps before acquisition, sigma_uv 0.005): S6 and S7 92% and 83% of trials,
of which 45% and 51% succeed, against 85% for the rest (five-mode 91%, 81%,
64%, 68%, 88%, 89%). S1 and S2 75% and 83%, success unchanged.

## Three-mode batch (applied 2026-10-11)

Every old text below matched Draft_12.tex exactly and was replaced on
2026-10-11. `plot_flip_resolution_starts.CSV_FILES` now names the `_d3` CSVs.

B1, Sec. III-B, capture hold. Old:
```
coefficients, the least noisy the fit returns. Once it passes, the tracker
descends $\hat{D}$ for the rest of the run, with the Newton step taken
with both curvatures in magnitude, $\sum_j
\mathrm{sat}(-\mathbf{w}_j^\top\hat{\mathbf{g}}_0/|\lambda_j|)\mathbf{w}_j$.
The same step serves, without the latch, wherever the fitted Hessian is
definite, $\lambda_1\lambda_2 \geq 0$, and the trench frame is undefined.
The test reads the measured flow, so it is not objective.
```
New:
```
coefficients, the least noisy the fit returns. Once it passes, the tracker
holds for the rest of the run with the attraction law of \cite{10},
\begin{equation}
    \dot{\mathbf{p}}_c = k\,\mathrm{sat}\bigl(-\hat{\mathbf{J}}_0^{-1}\hat{\mathbf{v}}_0\bigr),
    \label{eq:d_hold}
\end{equation}
with $\mathrm{sat}$ applied to each component, which steers the centroid to
the fitted stagnation point from the same coefficients as the test. Both
read the measured flow, so neither is objective.
```

B2, Sec. III-B, after (\ref{eq:d_law}). Old:
```
Both branches of $v_\parallel$ are nonnegative, so the along-trench
motion never opposes the flow.
```
New:
```
Both branches of $v_\parallel$ are nonnegative, so the along-trench
motion never opposes the flow. The same law runs where the fitted Hessian
is definite and no trench exists, with $\mathbf{w}_1$ still the
eigenvector of the smaller eigenvalue, so the tracker needs no separate
rule there.
```

B3, Sec. III-B, end of the stability paragraph. Old:
```
frame's transport velocity, leaves $n$ ultimately bounded while it stays
below the largest across-trench speed $k\,c_{\max}$.
```
New:
```
frame's transport velocity, leaves $n$ ultimately bounded while it stays
below the largest across-trench speed $k\,c_{\max}$. The argument needs
$\lambda_2 > 0$. Where noise makes the fit negative definite, the
across-trench channel steps toward a maximum of $\hat{D}$ along
$\mathbf{w}_2$, and no stability claim is made there.
```

B4, Sec. IV-B. Old `0.014 of it and the $D$ tracker within 0.009 to 0.014.`
New `0.014 of it and the $D$ tracker within 0.004 to 0.006.`

B5, Sec. IV-B, fitted-Hessian paragraph. Old:
```
test (\ref{eq:d_capture}), built from the zeroth- and first-order
coefficients alone.
```
New:
```
test (\ref{eq:d_capture}) and holds with (\ref{eq:d_hold}), both built
from the zeroth- and first-order coefficients alone.
```

B6, Sec. IV-C. Old
`falls from 99.5\% at $\sigma_{uv} = 0.0015$ to 13\% at $\sigma_{uv} = 0.03$,`
New
`falls from 99.4\% at $\sigma_{uv} = 0.0015$ to 23\% at $\sigma_{uv} = 0.03$,`

B7, Sec. IV-C. Old:
```
which its retention falls to 50\% is 5.3 to 5.5 times the $s_1$ tracker's
above the origin, 3.8 times at it, and 1.3 to 2.6 times below it. The
order reverses only at high noise from S6 and S7, where the two retention
curves cross near $\sigma_{uv} = 0.0075$. By $\sigma_{uv} = 0.01$, however,
neither tracker retains more than 61\% of its trials from any start, so
```
New:
```
which its retention falls to 50\% is 5.4 to 5.8 times the $s_1$ tracker's
above the origin, 4.2 times at it, and 1.1 to 1.2 times from S6 and S7.
From S5 its retention never falls to 50\%. The
order reverses only at high noise from S6 and S7, where the two retention
curves cross near $\sigma_{uv} = 0.006$. By $\sigma_{uv} = 0.01$, however,
neither tracker retains more than 65\% of its trials from any start, so
```

B8, Sec. IV-C. Old `two rates differ by at most 10 percentage points up to`
New `two rates differ by at most 6 percentage points up to`

B9, Sec. IV-C, early firing. Old:
```
definite. At $\sigma_{uv} = 0.005$ this happens in 91\% and 81\% of trials
from S6 and S7, and 64\% and 68\% of those trials go on to reach
$\mathbf{p}^*_1$, against 88\% and 89\% for the rest. From S1 and S2 the
step fires in 74\% and 83\% of trials without changing success. Runs from
```
New:
```
definite. At $\sigma_{uv} = 0.005$ this happens in 92\% and 83\% of trials
from S6 and S7, and 45\% and 51\% of those trials go on to reach
$\mathbf{p}^*_1$, against 85\% for the rest. From S1 and S2 the
step fires in 75\% and 83\% of trials without changing success. Runs from
```

B10, Fig. 9 caption. Old:
```
    to the bifurcation and take the same branch past it. Neither reaches
    land within the record.}
```
New:
```
    to the bifurcation and take the same branch past it. The $D$ tracker
    reaches land at 23~h, and the $s_1$ tracker is still at sea when the
    record ends.}
```

B11, Sec. V-B. Old
`arrives. Both are still descending when the record ends, 3.4~km apart.`
New:
```
arrives. The $D$ tracker reaches land at 23~h, while the $s_1$ tracker is
still descending when the record ends.
```

B12, Sec. V-B. Old `is 2.8~km for the $D$ tracker and 2.1~km for the $s_1$ tracker. The grid is`
New `is 1.6~km for the $D$ tracker and 2.1~km for the $s_1$ tracker. The grid is`

B13, Limitations. Old `Its capture reads the measured flow, so it is not objective.`
New `Its capture and hold read the measured flow, so neither is objective.`

Figures to install into the paper's `figures/` (back up the current ones first):

| Paper file | Source (VF_Robot) |
|---|---|
| `traverse_vs_logic_c_seven.png` (Fig. 5) | `experiments/outputs/oecs/traverse_vs_logic_c_seven.png` |
| `separatrix_formation_seven.png` (Fig. 6) | `experiments/outputs/oecs/separatrix_formation_seven.png` |
| `noise_seven_starts_col.png` (Fig. 7) | `overnight/figures_for_review_d3/noise_seven_starts_col.png` |
| `objectivity_traverser.png` (Fig. 8) | `experiments/outputs/oecs/traverse_objectivity.png` |
| `ocean_progression_2km.png` (Fig. 9) | `../ocean_data/det_jacobian_plots/ocean_progression_2km.png` |

Then point `plot_flip_resolution_starts.CSV_FILES` and the Fig. 7 producers at
the `_d3` CSVs, and compile twice.

## Review batch (applied 2026-10-09)

Numbers, from the new outputs, against the paper's current values:

- Noise-free control (Sec. IV-C, "99.2%"). s1 is now 10^4 of 10^4 at every
  start (`overnight/noisefree_s1.csv`). D is in `overnight/noisefree_d.csv`.
- Fig. 7 retention numbers and 50% crossings (Sec. IV-C), from the printout of
  `fig7_regen.py`. The interim s1-only plot already shows the s1 tracker
  ahead of the old D tracker below the origin (S6, S7 crossings 0.014, 0.019
  against 0.004, 0.005), which reverses "the D tracker tolerates more noise
  from every start".
- Reversal and early-firing statistics (Sec. IV-C), `mc_mechanism` CSVs.
- Settle distances (line 886). See the exit-bound note below.
- Rotating frame (Sec. IV-D). The s1 numbers came out unchanged (0.0162 and
  0.0099). The D capture fires in the inertial frame only.
- SBC (Sec. V). The s1 path is unchanged. The D path leaves the main ridge
  near 34.2 N and follows the s1 tracker's route, so the branch story and the
  ridge distances change (`overnight/sbc_ridge_distance.py`).

Text to rewrite: IV-B terminal step and the fitted-Hessian analysis (the D
tracker now holds at p1*), Fig. 5, 6, and 8 captions, IV-C capture paragraph,
IV-D, Sec. V, line 886.

Decide: the |y| > 0.52 exit bound in `main_separatrix_traverse.py`.

## Evidence

Noisy comparisons below use random heading, 300 steps, exit at |x| > 1 or
|y| > 0.60, success radius 0.06 around the far saddle (0, -0.5), and do not
stop at saddle contact. They compare variants with each other and are not
comparable with the paper's sweep numbers. Trial counts are 7 starts times 50
unless stated.

### Why the s1 tracker lost ACQUIRE, PARK, and CROSS

Mode logs of the previous code on every paper run (`exploration/modes_s1.py`,
`modes_all.py`): the clean runs, the S1 noise sweep, and the rotating frame
are TRACK then CAPTURE, and Santa Barbara is TRACK throughout. ACQUIRE fired
only at S4, for 3 steps, because the origin is the isotropic point where
s1 = 0. CROSS never fired and on the double gyre could not. In
incompressible flow s1 = -r, so the latch needs r > s_latch = r_band, and the
CROSS test r < r_band fails on the latch step. PARK was off. s_trim came from
Primitive 10's TRIM threshold (commit 17a2c72), has units of 1/time, and was
tuned per field (0.05 gyre, 0.3 ocean).

### The s1 footprint test q

Noise-free along x = 0 (`exploration/qmap.py`):

| y | 0.45 | 0.35 | 0.25 | 0.10 | +-0.02 | -0.25 | -0.35 | -0.45 | -0.48 | -0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| q | 0.028 | 0.111 | 0.227 | 0.719 | 3.75 | 0.247 | 0.130 | 0.047 | 0.024 | 0.009 |

q < 0.05 only within about 0.05 of a saddle. At the gyre centres and the
origin q is in the thousands, so no separate shallow-spot gate is needed.
Under noise, grad s1 comes from second-order coefficients (noise gain about
13 times that of s1 at rho = 0.075). At sigma_uv 0.002 the median noisy
|grad s1| at the saddle is 0.79 against a true 0.12, and q there has median
0.061 and drops below 0.03 on 17% of steps, against 3% at y = -0.35
(`exploration/qnoise.py`).

### s1 capture variants

| Variant | sigma_uv 0.002, far | sigma_uv 0.005, far | Notes |
|---|---|---|---|
| previous code | 331 / 700 | 101 / 700 | |
| q entry, release when q > 0.06 | 19 / 350 | | released in 349 of 350 trials, then slid onto the wall |
| **q entry, no release, descend s1** | **490 / 700** | **214 / 700** | adopted |
| q entry, no release, zero command | 48 / 350 | 20 / 350 | stops where q first dips, 0.06 to 0.3 short |
| flow-saddle entry, no release, descend s1 | 197 / 350 | 82 / 350 | against 242 and 99 for q on the same seeds |

Release fails because q at the saddle is mostly noise. Descent works because
it converges from wherever capture fires, and under noise capture often fires
mid-ride (about half the trials more than 0.1 from a saddle), so descent
finishes the ride. The flow-saddle entry is precise but lets more trials
reverse before the end, and it would cost the s1 tracker its objectivity.

### s1 first step

Taking the first tangent along the flow, rather than signing the argmax
tangent against it, matched on the double gyre (242/350 either way at
sigma_uv 0.002) but sent the Santa Barbara run north, 33 km from the published
end. That trial sits near a branch point, and a 0.28 km offset at step 0
flipped it. The adopted rule reproduces the published Santa Barbara path bit
for bit.

### D tracker: the FLOW test and FLOW_DRIFT

The previous D tracker on the clean runs used FLOW 340, SLIDE 445, ATTRACT 164,
ATTRACT_FALLBACK 200, FLOW_DRIFT 0 steps (`exploration/modes_d.py`). With SLIDE
alone it slides to the nearest saddle and holds, and stalls at S4. SLIDE and
ATTRACT alone traverse every off-origin start, so FLOW is only needed to get
moving where grad D = 0 (`slide_only.py`, `slide_attract.py`).

On Santa Barbara the previous D tracker is mostly FLOW (152 of 168 steps) and
FLOW_DRIFT fires 8 times. Every replacement for the FLOW band was judged on
three checks: the clean gyre, noise, and the Santa Barbara path
(`maps/`, `exploration/d_ocean_*.py`).

| FLOW test, fallback | Clean gyre | sigma_uv 0.005, far | Santa Barbara |
|---|---|---|---|
| previous band, signed fallback | 7/7 | 196 | published path |
| q_D < 1 (D within reach of zero inside the footprint), signed, drift kept | 7/7 | 205 | within 0.6 km of published |
| crest within rho, signed, no drift | 7/7 | | loops at the start, lands at 22.8 h |
| crest within 0.25 rho or tighter, signed, no drift | 7/7 | | stalls near 34.2 N |
| crest within 0.5 rho, abs fallback | 7/7 | 238 | on the main ridge to landfall at 23.7 h |
| **crest within 0.1 rho, abs fallback, no drift** | **7/7** | **242** | **follows the s1 route, at sea at 28 h** |

The signed fallback climbs D wherever the fit comes out negative definite, as
it does at the double-gyre saddles. Tightening FLOW handed steps to it on
Santa Barbara, and the team stalled. With |lambda| it descends instead, which
is what let the crest test shrink to 0.1 rho (chosen by Chris).

### D tracker: the capture

Flow-saddle capture, det J0 < 0 and |J0^-1 v0| < rho (`exploration/fsad*.py`):

| | Clean gyre | sigma_uv 0.002, held at far saddle | sigma_uv 0.005 | False captures more than 0.1 from a saddle | Santa Barbara | Rotating frame |
|---|---|---|---|---|---|---|
| no capture | 0/7, rides the wall | 0 / 350 | 0 / 350 | | | |
| flow-saddle capture | 7/7 | 337 / 350 | 224 / 350 | 0 | never fires | fires in the inertial frame only |

The test uses only zeroth- and first-order coefficients, the least noisy ones.

### Found along the way

- Exit bound. `main_separatrix_traverse.py` stops at |y| > 0.52. Both trackers
  overshoot to about y = -0.522 on arrival and then settle at (0, -0.514), so
  with that bound S3, S4, and S7 are cut off mid-overshoot. With 0.60 all seven
  hold at 0.014 from the saddle.
- Stale docstrings, fixed 2026-10-08. Primitives 10 and 11 claimed
  componentwise saturation in ACQUIRE and PARK, and the code saturates the
  magnitude.

## Files

Run everything from `trunk/Python_Simulations/Vector_Fields/VF_Robot` with
`venv/bin/python3 experiments/primitive11_simplification/<path>`.

| Path | Contents |
|---|---|
| `overnight/run_overnight.sh`, `run.log` | s1 rerun driver and log |
| `overnight/run_d_rerun.sh`, `run_d.log` | five-mode D rerun driver and log (`run_d_stale_no_capture.log` is a stopped run) |
| `overnight/run_d3_rerun.sh`, `run_d3.log` | three-mode D rerun driver and log |
| `exploration/dnew.py`, `exploration/screen_*.py` | three-mode screen |
| `overnight/noisefree_s1.py`, `noisefree_d.py` | noise-free controls, 10^4 trials per start |
| `overnight/sbc_ridge_distance.py` | Sec. V ridge distances. Pass `old` for the pre-rewrite code, which gives 1.49 and 2.10 km as published |
| `overnight/fig7_regen.py` | Fig. 7 from a chosen CSV set into a review folder |
| `overnight/ocean_path.py` | s1 Santa Barbara path, new or `old` |
| `overnight/pentagon_primitives_pre_rewrite.py` | primitives as they were before 2026-10-08 |
| `overnight/backup_pre_rewrite/`, `backup_after_s1_rerun/` | outputs, figures, Draft_12, and the primitives file at each stage |
| `overnight/figures_for_review*/` | regenerated Fig. 7 |
| `open_checks.py`, `open_checks.log` | 1000-step runaway (sigma 0.005) and off-trench checks |
| `runaway_002.py` | the same runaway check at sigma 0.002 (stopped, superseded by the hold rule) |
| `hold_k_check.py`, `hold_k_check.log` | contact against a 10- and 20-step hold, 1260 trials |
| `rate_sweep_d.py` | Sec. IV-D rotation rate at which the D tracker first loses the straddle (now 0.205) |
| `maps/` | Santa Barbara maps of the D-tracker variants |
| `exploration/` | the scripts behind every table above |

The exploration scripts ran against the code of their day. Those that call the
repo primitives directly now exercise the rewritten versions. `variant_q.py`,
`dvar.py`, and `fsad.py` hold the scratch versions of the trackers that the
other scripts import.

| Exploration scripts | Section |
|---|---|
| `modes_s1.py`, `modes_all.py`, `cross_first.py`, `cross_no_acquire.py`, `s4_nocross.py` | why ACQUIRE, PARK, CROSS went |
| `qmap.py`, `qnoise.py` | the footprint test q |
| `variant_q.py`, `clean_q.py`, `noise_q.py`, `ocean_q.py`, `noise_norelease.py`, `noise_zero.py` | s1 capture variants |
| `seed_init.py`, `opt_a.py`, `ocean_step0.py` | s1 first step |
| `modes_d.py`, `slide_only.py`, `slide_attract.py`, `band_map.py` | D tracker modes and the FLOW band |
| `dvar.py`, `d_variants.py`, `d_noise.py`, `d_noise_drift.py`, `crest_test.py`, `crest_gyre_tight.py`, `abs_tests.py` | D FLOW test and fallback |
| `d_ocean.py`, `d_ocean2.py`, `d_ocean_plot.py`, `d_ocean_plot2.py`, `d_ocean_pent.py`, `d_ocean_crest_tight.py`, `d_ocean_absfb.py` | D on Santa Barbara, and the maps |
| `fsad.py`, `fsad_tests.py`, `fsad_sbc_rot.py` | flow-saddle capture, both trackers |
