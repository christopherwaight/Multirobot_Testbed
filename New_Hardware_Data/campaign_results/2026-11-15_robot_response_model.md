# How the real robots respond to a velocity command, and what the Python robot model should be

**Status:** analysis done on 2026-11-15 by Claude (AI assistant), at the user's request. **Nothing has been changed in the Python code.** The user will decide later.
**Data:** the 2026-11 hardware campaign so far: vortex 3R/4R from 2026-11-13, and saddle 3R from 2026-11-15, using the clean copies in `data_hardware_campaign_11_11_2026_clean/`.
**Scripts:** in `trunk/campaign_results/scripts/`: `fit_coulomb.m`, `fit_coulomb_delay.m`, `delay_locate.m`.

---

## 1. Short answer

- **Model:** each robot does what it's told, but **one step (0.1 s) late**, and at about **85% of the commanded speed**:

  `v(k) = 0.11·v(k−1) + 0.89·(0.85·u(k−1))`

  - Almost no momentum: τ ≈ 0.045 s.
  - Kinetic (Coulomb) friction ≈ 0.
  - Speed saturates at about 0.3 m/s.
  - Static stiction (breakaway) is **not measured yet**.
- **The Python model's structure is fine.** A lag plus Coulomb friction plus a speed cap is a sensible form. Its parameters are off, though, and it has no delay.

| Parameter | Python `OmniCluster` now | Proposed | Confidence |
|---|---|---|---|
| input delay | none | **1 step (0.1 s)** | high |
| α (lag), dt 0.1 | 0.70 (τ 0.28 s) | **≈ 0.1** (τ ≈ 0.05 s) | high (per robot 0.06–0.17) |
| command gain | 1.0 (no parameter) | **≈ 0.85** | high; depends on battery (0.8–0.93 healthy) |
| kinetic friction v_k | 0.025 m/s | **≈ 0** (≤ 0.01) | medium-high |
| static stiction v_s | 0.025 m/s | **unknown**; keep 0.025 as a placeholder | none: not tested by this data |
| speed cap | 0.4 m/s | **0.3 m/s** | medium (few samples at high command) |

- **Fit quality (free-run velocity RMSE):**
  - Python defaults: **0.045 m/s**;
  - best no-delay fit: 0.037;
  - delay model: **0.032**, about 30% better than Python.
- **If adding a delay isn't wanted:** the best no-delay compromise is α 0.47, gain 0.89, v_k ≈ 0. Python's `Omnibot` currently has **neither a gain nor a delay parameter**.

---

## 2. What the Python model is

`from_mac/VF_Robot/VF_Robot/src/robot/omnibot.py`, as used by `OmniCluster` (`omni_cluster.py`), whose default has been `stiction_model="coulomb"` since 2026-10-29:
- **α:** `α = exp(−dt/τ)` with τ = 0.28 s, so α = 0.70 at dt = 0.1 s.
- **Velocity update:**
  - while stuck: `v = 0` until |u| > v_s (breakaway);
  - while moving: `free = α·v + (1−α)·u`, then `v = free·(1 − (1−α)·v_k/|free|)`.

  The robot settles at |u| − v_k, and stops (and sticks) when that reaches 0.
- **Friction:** v_s = v_k = 0.025 m/s.
- **Speed cap:** 0.4 m/s (raised from 0.3 on 2026-10-29).
- **Gain:** unity steady-state gain; there is no gain parameter.
- **Delay:** none. The command acts in the same step.
- **Where τ comes from:** the docstring says the paper derived τ = 0.3 s for the Decabot.

---

## 3. How I got here (in order, including the wrong turn)

### 3.1 Data preparation
- **Clean copies:** used the clean campaign files, which hold only what each run logged; see `data_hardware_campaign_11_11_2026/README_STALE_VARIABLES.md`.
- **Per robot and per run:**
  - command `u(k)` = `cmd_robots_selected` (global [vx vz] for that robot);
  - measured velocity `v(k) = (p(k+1) − p(k)) / dt(k)` from `pose_rN`, with `dt(k)` from the **wall-clock** `timing` column (median 0.100 s, p5–p95 0.094–0.109).
- **Battery screen:** robot-runs whose median |v|/|u| was below 0.75 were dropped, as low battery (e.g. tidal in saddle r_d 0.01 runs 3–6, redwood in 4R r_d 0.01 runs 3–7). That left **550 robot-runs**.
- **Excluded:** the failed `Robot4_vortex_050_runs_off_the_mat.mat`.

### 3.2 First fit: the Python structure as-is, no delay (`fit_coulomb.m`)
- **Models fitted:**
  - Python's model with its defaults;
  - a linear lag `v(k) = α·v(k−1) + (1−α)·g·u(k)`;
  - the full Coulomb model with α, g and v_k free.
- **How they were scored:** one-step-ahead (starting from the measured previous velocity) and free-run (simulate from rest using only the commands).
- **Result:** α ≈ 0.47 (τ 0.13 s), g ≈ 0.89, v_k ≈ 0.005. Free-run 0.037, against 0.045 for Python's defaults.
- **Steady-state check:** for samples where the command had been constant within 2 cm/s for 0.5 s, speed along the command / |u| was **flat at 0.80–0.86** over |u| = 0.10–0.25 m/s.
  - A Coulomb offset would make that ratio rise with speed. So friction for a moving robot is essentially zero, and the shortfall is a **proportional gain**.
  - A straight-line fit over the narrow 0.13–0.20 range gave a misleading "gain 0.68, +0.026 offset". The binned ratios are the reliable view.
- **Clue:** a quick linear delay test in the same script showed `v(k) = 0.11·v(k−1) + 0.76·u(k−1)` fitting **much** better than any no-delay form. Per component, the RMSE was 0.021 vs 0.026.

### 3.3 Second fit: same structure plus an input delay d (`fit_coulomb_delay.m`)
| delay | α (τ) | g | v_k | 1-step RMSE | free-run RMSE |
|---|---|---|---|---|---|
| 0 | 0.47 (0.13 s) | 0.89 | 0.005 | 0.037 | 0.037 |
| **1** | **0.11 (0.045 s)** | **0.85** | **0.000** | **0.030** | **0.032** |
| 2 | (fails) | 0 | – | 0.052 | 0.149 |

- **Per robot, delay 1:**

  | Robot | α | g | v_k |
  |---|---|---|---|
  | celeste | 0.17 | 0.85 | 0.009 |
  | tidal | 0.06 | 0.88 | 0 |
  | pacific-blue | 0.09 | 0.89 | 0.001 |
  | redwood | 0.09 | 0.88 | 0.006 |

  They're consistent with each other.
- **Speed cap:**
  - max measured speed: 0.41 m/s;
  - commands of 0.30–0.35 m/s: median speed 0.27 (ratio 0.82);
  - commands of 0.35–0.40: median 0.27 (0.74);
  - commands of 0.40–0.45: median 0.21 (0.49).

  So the robots saturate at about 0.3 m/s. These are few samples, mostly transients.

### 3.4 Where is the delay? (`delay_locate.m`, 3R r_d 0.1 runs)
- **Inside the controller?** No. The logged Orbiter command `cmd_cluster_jacobian(k)` matches the Orbiter law evaluated on **the same step's** `cluster_state(k)` and `field_cp(k)` **exactly** (median difference 0). Using the previous step's state misses by 1.5 cm/s.
- **So the delay sits between the command and the measured motion:**
  - radio, Decabot Helper or Decabot latency, and/or OptiTrack latency (the pose arriving up to a frame late);
  - for the closed loop these are equivalent: about one step of loop delay.
  - Correlations agree, though only weakly: corr(v(k), u(k−1)) is 0.985 vs 0.979 for u(k). Orbit commands are smooth, so correlation alone discriminates poorly; the model fit is the stronger evidence.

### 3.5 The wrong turn, corrected
- **What the earlier fits said:** the 2026-11-11 fit (README_hardware §5.7) and the first campaign fit found "momentum ≈ 0.40–0.44, gain ≈ 0.8".
- **Why that was wrong:** with no delay in the model, the lag term absorbs the one-step delay. A lag of about 0.45 at the same step mimics "the robot follows the previous command".
- **Why the 11/11 fit missed it:** it said "adding a one-sample delay fits no better", but it used commands *reconstructed* offline from the logged state. That blurred the timing.
- **Why it's clear now:** the campaign logs the actual commands and the wall-clock time, and that makes the delay unambiguous.
- **What's superseded:** the "momentum 0.4–0.44" figure in earlier notes (CLAUDE.md and README_hardware §5.7 have been updated).

---

## 4. Caveats
- **Stiction can't be identified from this data.** Robot commands were essentially never below 0.1 m/s (the steady-state bins start at 0.08–0.10 with only 6 samples). Breakaway was never tested.
- **The speed cap rests on few samples,** mostly from transients (the radial pull far from the radius, estimate jumps).
- **Gain depends on battery.** Healthy robots read 0.8–0.93; fading batteries go down to 0.34 (redwood, 11/13). Any sim gain is a "healthy battery" number.
- **Velocity from pose differences** at 0.1 s includes OptiTrack noise and frame-timing jitter. That sets an RMSE floor, so the absolute RMSE values overstate model error. Compare them relative to each other.
- **The delay may not be exactly one step.** One step fits clearly better than 0 or 2. A fractional delay (e.g. 0.7 step) wasn't tested, and with α ≈ 0.1 left over, about one step is a fair summary.
- **Coverage:** the fit uses the vortex and the first saddle runs. Re-run `fit_coulomb_delay.m` at the end of the campaign on all data, which is quick.

---

## 5. Open items / options for later
1. **Low-speed stiction test (about 5 min):**
   - one robot, from rest, straight-line commands of 0.01, 0.02 … 0.08 m/s, 3 s each, logged as usual;
   - the lowest speed that moves it is the breakaway threshold v_s;
   - steady speed below the commanded speed would give v_k.
2. **Sim model changes,** if adopted:
   - add a command gain and a one-step input delay to `Omnibot`;
   - set α ≈ 0.1, g ≈ 0.85, v_k ≈ 0, cap 0.3 m/s;
   - keep v_s 0.025 until item 1.
   - Then re-run the Monte Carlo baselines. The `Omnibot` docstring warns that Table II depends on the robot model.
3. **Alternative if no delay is added:** α 0.47, g 0.89, v_k ≈ 0.
4. **Check the closed loop:** with the chosen model, re-predict the measured orbit radii, as was done on 11/11 with the old model (within 1 cm). That's the real test of whether the sim matches the hardware.
5. **Paper / thesis:** the hardware loop has about 0.1 s of sense-to-act delay, and the robots track commanded velocity at about 85% with negligible lag. This is a cleaner statement than "momentum 0.4".
