# 3-Robot Hardware Testbed: How It Works and Lessons Learned

This covers the printed floor maps, the Simulink model `three_robot_cluster_all_primitives.slx`, and what went wrong during the hardware campaign on 2026-11-11 and how it was fixed. It assumes you have never seen this code.

For the daily lab procedure (cameras, Motive, Decabots, batteries), see [`../Optitrack_demo/OptiTrack_SOP.md`](../Optitrack_demo/OptiTrack_SOP.md).

---

## 1. The idea in one paragraph

A vector field is printed on the floor as colour. **Hue = direction** and **saturation = magnitude**. Each robot has a colour sensor, so it reads "which way and how strong" at the spot where it sits. Three robots move together as a triangle (a *cluster*). From three readings at three known positions (OptiTrack gives the positions), the cluster fits a local linear model of the field. From that model it estimates where the field is zero: the *critical point*, i.e. the centre of a vortex, a saddle, and so on. A control *primitive* then decides how the cluster moves, for example orbiting that point. A cluster-space controller turns "move the triangle like this" into three robot velocity commands.

---

## 2. Data flow through the Simulink model

```
OptiTrack (Motive) ──► Robot_signal_processing_1..3 ──► robot poses [x, z, heading]
                                                          │
Decabot colour sensor (R,G,B,K) ──► RGB to HSV to vel ──► hue, saturation (0..1)
   (celeste / tidal / pacific-blue)  (normalise → neural net → post-process)
                                                          │
                       ┌──────────────── primitives (pick one with Manual Switches) ─────────┐
                       │ Cluster_Controller_setpoint  hold a triangle at a set point          │
                       │ Jacobian_primitive           find the critical point, then orbit it  │
                       │ vector_sum                   follow the field direction              │
                       │ vector2scalar_mag_steepest_* climb/descend the magnitude             │
                       └──────────────────────────────────────────────────────────────────────┘
                                                          │
             cluster velocities ──► inverse Jacobian ──► per-robot global velocity
                                                          │
             Global to Body Rotation (uses each robot's OptiTrack heading) ──► Decabot
```

- **Primitive selection:** a chain of root-level `Manual Switch` blocks picks the active primitive. Check them before every run. Nothing else tells you which primitive actually drove the robots.
- **Sensor neural nets:** they live in the base workspace. Run `neural_network_loader.m` first; the MATLAB Function blocks pull the nets in with `evalin('base', ...)`.
- **Sensor normalisation** (`RGB to HSV to vel/RGB to HSV`): R, G, B → `(x − 120)/1300`, K → `(x − 800)/3500`, each clipped to [0, 1]. These constants are tied to the sensors as they are now.

### Cluster kinematics (ported 2026-11-11)
- **Definitions:** the cluster variables are the triangle `p = |r1 − r2|`, `q = |r3 − r2|`, `β` = the angle at robot 2, the centroid `(x_c, z_c)`, and `θc = atan2` of the direction from robot 2 to robot 1. They come from the Python `TripleBlock` reference, and the forward and inverse kinematics are exact inverses of each other (checked to ~1e-10).
- **Handedness flag `s`:** the forward kinematics also outputs `s`. It is +1 if robot 3 is CCW from robot 1 about robot 2, and −1 if it's CW. This testbed's layout gives **s = −1**.
- **Old version:** the original model's kinematics are kept in `three_robot_cluster_all_primitives_og.slx`. Its inverse kinematics were only exact when p = q.

---

## 3. The printed floor maps: how and why

The maps are made by `Plots_to_print/Plots_to_print/VectorField_plot_creator.ipynb`. An older copy is in `trunk/Python Simulations/Plots_to_print`. The exported PNGs and PDFs sit next to it.

### 3.1 What the notebook does (vortex example, the cells for `vortex_strips`)
1. **Grid:** a 4000 × 4000 grid on x, y ∈ [0, 1], centred at (0.5, 0.5).
2. **Field:** `u = −(y − 0.5)`, `v = (x − 0.5)`. This is a counter-clockwise vortex whose speed grows with radius.
3. **Hue (direction):** `h = (atan2(v, u) + π) / 2π`, which maps to 0..1.
   - **Note the + π.** It means hue 0 (red) points along −x, not +x. Decoded, the direction is `ψ = 2πh − π`.
   - It is stored as OpenCV hue: `h·179`, uint8, range 0–179.
4. **Saturation (magnitude):** the magnitude is min-max normalised over the whole image, then squeezed into [0.3, 1]: `S = 0.3 + 0.7·m_norm`.
   - **Why 0.3:** at zero saturation, hue is undefined (everything is white), so the floor of 0.3 keeps every point coloured enough for the sensor to read a direction.
   - **Matching decode:** Simulink undoes it in `Sat_rescaling`: `(S − 0.3)/0.7`, clipped to [0.05, 1].
   - **Caveat:** because of the min-max normalisation, the map only carries *relative* magnitude.
5. **Value:** 255 everywhere (full brightness). The image is converted HSV → RGB.
6. **Strips:** the image is cut into three vertical strips (x from 0 to 1/3, 1/3 to 2/3, and 2/3 to 1), each printed tall and narrow. Each strip is labelled with its x range so the pieces can be laid side by side.
7. **Deliberately mirrored:** each strip is drawn with `invert_xaxis()`, so x runs **right-to-left** on paper (the labels read "1/3 … 0").
   - **Why:** OptiTrack (Motive) is Y-up. Its floor plane is (x, z), and going from +x to +z is **clockwise when seen from above**: a mirror image of a normal x–y plot.
   - Printing the map mirrored makes the paper's axes match Motive's, so the field keeps its type when read in Motive coordinates.
8. **No `savefig`:** the PNGs and PDFs were saved by hand from the figure output.

### 3.2 Where the map sits on the floor (as of 2026-11-11)
- **Inferred from the data:** sensor readings from the 2026-11-11 runs fit the map lying **rotated about 90°** relative to Motive's axes (89–93° across the three robots).
  - Map +x runs along Motive +z, and map +y runs along Motive −x.
  - The map's centre is at Motive (x, z) ≈ **(0.41, −0.12)**.
- **Check this physically.** A 90° hue offset in the sensor calibration would produce the same data.
- **If you re-lay the map, the decode changes** (section 4).
- **Record when you lay a map:** the orientation, the offset, and the printed scale (metres per map unit).

### 3.3 The saddle map is NOT mirrored (found 2026-11-15)
- **What the notebook does:**
  - The vortex strips (cell 11) are drawn with a reversed `extent` plus `invert_xaxis()`, so they are mirrored on paper (section 3.1, item 7).
  - The saddle strips (cell 29, `hyperbolic_saddle_vector_field`, rotation 0, strength 2, x and y in [−0.5, 0.5]) use a normal `extent` and no `invert_xaxis()`, so they are **not mirrored**.
- **Consequence:** relative to the vortex map, the saddle map carries one extra reflection. The model's decode is tuned to the vortex map (section 4), so on the saddle map it reads a reflected saddle, which looks like a **sink/node**, not a saddle. A reflection flips the sign of det(J) (section 4, lessons).
- **Evidence:** first saddle run, `saddle_3robots_radius_of_0_point_1_run1.mat` (2026-11-15, model e3584dd0e1c3, same decode as the vortex runs):
  - online `field_jacobian` type: saddle 0%, node 76%, spiral 24%;
  - global fit J ≈ [−0.73 −0.18; 0.02 −0.80], with trace −1.53, det +0.59 and eigenvalues ≈ −0.77 ± 0.05i: a sink;
  - the vortex runs, with the same decode: J ≈ [0.15 −0.99; 0.76 −0.07], with det +0.74 and curl 1.75: a vortex, as expected.
- **What this does NOT affect:**
  - **The critical-point estimate and the Orbiter.** The zero of the field is the same under any reflection of the vectors, so the estimate is identical to what a correct decode would give. Orbiting the saddle is valid.
  - **What it affects:** anything that uses the field's direction or its Jacobian, namely the online `field_jacobian` type and any field-following primitive.
- **For analysis:** to classify saddle runs, re-decode offline with one vector component negated, i.e. reflect `field_uv` before fitting J. That gives det < 0. Don't trust the logged type column (`field_jacobian(:,8)`) for saddle runs.
- **Not changed in the model** (the campaign was in progress). If the model is ever fixed, it should get a per-map decode setting rather than another sign flip.
- **Map size:** the user measured the saddle map as physically smaller than the vortex map, although both were "printed the same way".
  - **In the notebook:** both figures are `figsize=(30, 90)` with a 1 × 1 map-unit domain. However, the vortex figure has subplot titles while the saddle figure has axis labels and different ticks, so `tight_layout` gives the image a different share of the page. Printing fit-to-page then scales the two maps differently. That is a plausible cause.
  - **What the field data shows:** it **can't confirm** a few-percent size difference. The fitted field gain is 0.77 /m on the saddle vs 0.86 /m on the vortex, and the |u|-vs-distance slope ratio is 0.97. If anything that points to the same size or slightly larger, but the sensor calibration isn't accurate to a few percent in magnitude.
  - **To get the scale:** measure the width of one strip with a tape on each map.
  - **For analysis:** distances are in Motive metres, so the size only matters when converting map units to metres.
- **Saddle centre (first run):** about (0.830, 0.798) in Motive (x, z); the online estimate's median is (0.820, 0.802). The vortex centre was (0.839, 0.856).
  - So the saddle centre is about 5 cm closer to the Motive origin, almost all of it in z (−6 cm), with x the same within 1 cm.
  - A uniformly smaller map pinned at one corner would shift both coordinates, so most of this is probably placement, not scale.
  - Re-fit the centre over all saddle runs, the way it was done for the vortex.

---

## 4. Decoding hue back into a vector (the "vortex looked like a saddle" bug)

A hue number is only a direction once you pick a convention. **Different parts of this project picked different ones:**

| Where | Angle from hue h | Notes |
|---|---|---|
| Print notebook (map frame) | `ψ = 2πh − π` | What was actually encoded |
| **Simulink, Motive frame, map placed as in 3.2** | `ψ = 2πh − π/2`, i.e. **u = m·sin(2πh), v = −m·cos(2πh)** | Correct as of 2026-11-11 |
| Simulink before 2026-11-11 | u = m·sin(2πh), v = +m·cos(2πh) | **A mirror image**, so every vortex/source/sink reads as a saddle |
| Python `from_mac/VF_Robot` `field_types.py` | `ψ = 2πh + π/2` | Opposite flow to the line above |
| Python `VF_3_robot` `robot_cluster.py` | `ψ = 2πh` | Rotated 90° |

### Lessons
- **A mirror flips the type of a critical point.** Any reflection applied to the positions but not the vectors (or vice versa) turns a vortex, source or sink into a saddle. Flipping the sign of one component is enough. Rotations never do this; they only change which way the flow goes.
- **The critical point's location doesn't depend on the convention.** The point where the field is zero is the same under every decoding. That's why critical-point finding and orbiting worked even with the wrong decode. Anything that uses the field's *direction* or its *Jacobian* (classifying vortex vs saddle, following the field) needs the right decode.
- **The fix (2026-11-11):**
  - In `Jacobian_primitive/Critical_point_predicter/Subsystem`, `Subsystem1` and `Subsystem2`, the z component is multiplied by `Constant = −1`.
  - In `vector_sum/3_robot HSV to vector`, `z-Direction Reverser = −1`.
- **Result:** with the fix, all three runs on 2026-11-11 read as a clean vortex at 99–100% of time steps (trace ≈ 0, det > 0, curl > 0). Before it they read as a saddle 100% of the time.
- **If the map is re-laid lined up with Motive's axes**, use `ψ = 2πh − π` instead (u = −m·cos, v = −m·sin). A single "map angle" constant would be cleaner than sign flips if the map moves often.

### How to check the decode from any run
Fit the field to the three robots at each step and look at the Jacobian J:
```matlab
A  = [x(:) z(:) ones(3,1)];                  % three robot positions
uv = [m.*sin(2*pi*h), -m.*cos(2*pi*h)];      % the decode under test (h, m per robot)
co = A \ uv;  J = co(1:2,:)';                % J = [du/dx du/dz; dv/dx dv/dz]
% vortex: trace(J) ~ 0 and det(J) > 0     saddle: det(J) < 0
```
- **What to use:** the logged `hue` and `saturation` (sensor values, see section 6) and the logged `robot*_pose`.
- **Run it over every step:** a correct decode gives the same type at almost every step.

---

## 5. Lessons learned on the hardware (2026-11-11)

### 5.1 OptiTrack rigid-body zero → robots drive off at a constant angle
- **Symptom:** the cluster wandered and never held its triangle, with the old kinematics and the new.
- **Cause:** each rigid body's yaw zero in Motive didn't match its robot's forward axis.
  - The model turns global velocity commands into body-frame commands using the measured heading, so every command came out rotated by a constant angle.
  - The angles were celeste ≈ +155°, tidal ≈ −80° and pacific-blue ≈ −150°.
- **Fix:** re-create or re-zero each rigid body with the robot's **blue wheel pointing +x** (see the SOP). Afterwards the robots were within −2°, −2° and +11°.
- **Diagnose it:** rebuild each robot's commanded velocity from the logged poses, and compare its direction with the measured velocity (the difference in position between samples). A **constant** angle per robot means the rigid-body zero is wrong.
- **Not the cause:** missing heading control. The robots are omnidirectional, so heading drift is harmless as long as the measured heading is right.

### 5.2 θc (cluster orientation) is controlled by default
- **What happens:** `Compute_error` feeds the θc error into the inverse kinematics at gain 1 with **no saturation**, while x_c and z_c are clipped to ±0.3.
- **When it's harmless:** if the cluster starts near the θc setpoint.
- **When it bites:** if the cluster starts far away (for example 150°+), it asks for large spin speeds that swamp everything else. Near ±180° the wrapped error also flips sign.
- **Python does it differently:** Python never controls θc (`omega_c = 0`) and uses a 0.1 gain on angles.
- **Done 2026-11-12:** a Gain block `thC_gain` = **0** ("no spin", like Python) now sits between the θc error and the inverse Jacobian, in both `Cluster_Controller_setpoint` and `Jacobian_primitive`. Set it to 1 to hold θc at its setpoint again. The `err_*` logs still show the θc error; `cmd_cluster_*` column 3 shows the commanded value (0).

### 5.3 Proportional control stops short
- **What happens:** the cluster stops a few cm and a few degrees short of the setpoint.
- **Why:** the remaining errors only ask for about 0.03–0.05 m/s, which is below the speed at which the robots start to move (the Python robot model uses a stiction threshold of 0.025 m/s).
- **Options:** raise the gains, add a small integral term, or set a minimum command speed.

### 5.4 The orbit can't be smaller than about 5 cm at 10 Hz
- **The law:** the Orbiter (`Jacobian_primitive/MATLAB Function`) commands a **fixed** tangential speed (`orbit_speed = 0.15` m/s) plus a radial correction `K_p·(r_d − r)`.
- **Why there's a floor:** with a 0.1 s step, each straight step along the tangent pushes the centroid outward. The radius settles at about

  `r* ≈ v_t·dt / sqrt(1 − (1 − K_p·dt)²)`

  and the one-step lag of the real robots makes it larger again.

| Settings | Predicted (no lag … one-step lag) | Measured |
|---|---|---|
| K_p 0.2, r_d 0.01 | 0.080 … 0.137 m | 0.105 m |
| K_p 1, r_d 0.1 | 0.110 … 0.127 m | 0.107 m |
| K_p 1, r_d 0.01 | 0.040 … 0.067 m | 0.057 m |

- **Python too:** the Python simulators step at 0.1 s as well, so they show the same floor (about 0.15 m with their own gains and robot model). Only a simulation with a much smaller time step reaches 1 cm.
- **For a smaller orbit:** scale the tangential speed with the radius (a constant angular rate, `v_t = ω·r_d`), keep it above the stiction speed, and raise `K_p`. As a rule of thumb, you need `v_t ≲ r_d·sqrt(2·K_p/dt)`.
- **Same result in the Python code:** `from_mac/VF_Robot/VF_Robot/experiments/hyperparameter_study/theory.py` derives the same thing: `r* = (r_d + sqrt(r_d² + 4C))/2` with `C = V²(T_lag + dt/2)/k_r`. The smallest orbit grows with the tangential speed and shrinks with the square root of the radial gain.

**Changed 2026-11-12: the Orbiter now has the Python law's structure, with the user's gains.**
- **The law:** `critical_point_orbiter_plane_fitting` in `from_mac/VF_Robot/VF_Robot/src/control/primitives.py` (paper Eq. 8/11):

  `v = k_t·r̂⊥ + k_r·(r − r_d)·r̂`, with the radial part clipped to ±0.3 m/s.
- **What r̂ is:** a **unit vector** (length 1) pointing from the centroid to the critical point; r̂⊥ is r̂ turned 90°.
  - **Tangential part:** always exactly `k_t`, near or far.
  - **Radial part:** grows with |r − r_d|. On the orbit it is small and at right angles to the tangential part, so it barely changes the speed: √(k_t² + radial²).
  - **Orbit speed:** therefore set by `k_t`. Actual ≈ 0.78·k_t (5.7).
- **Gains:**
  - At first (2026-11-12) the Orbiter took Python's gains (k_t 0.2, k_r 0.5).
  - The same day, the user chose **`k_t` 0.15, `k_r` 1**, the old Simulink values. The reasoning: a slower orbit holds the radius better, and the strong radial pull is what they want.

| | Before 2026-11-12 | Now | Python |
|---|---|---|---|
| tangential speed | `orbit_speed` 0.15 m/s | **`k_t` 0.15 m/s** | `k_t` 0.2 m/s |
| radial gain | `K_p` 1 (0.2 in the first run) | **`k_r` 1** | `k_r` 0.5 |
| limit | radius *error* clipped to ±5 m (no real speed limit) | radial *speed* clipped to ±0.3 m/s (only bites when > 0.3 m off the circle) | same as now |
| direction flag `Jacobian_primitive/Constant` | −1 | **+1** | (+1) |
| θc | held at π/3 | not controlled (`thC_gain` = 0) | not controlled |

- **Code:** `tools/orbiter_python_match.m`. It is the **single source** of the Orbiter code and gains for **both** the 3-robot and the 4-robot models.
  - **To push it into a model:** run `tools/sync_orbiter_code.m <model.slx>`. It is safe to re-run.
  - **First install:** `tools/apply_orbiter_python_match.m` (it also set the direction flag and `thC_gain`).
- **Backups:** `three_robot_cluster_all_primitives_BACKUP_pre_orbiter_python_20261112.slx` (old law) and `..._BACKUP_pre_orbiter_gains_20261112.slx` (Python gains, just before the switch to 0.15 / 1).
- **Expected effect on the hardware** (prediction from the fitted plant in 5.7; not yet measured):
  - **Radius:** same as the old law at the same `r_d`. The gains are identical, the 0.3 m/s clip only matters far from the circle, and the direction doesn't change the radius. So the first column below applies (not re-simulated with the clip).
  - **Speed:** the orbit runs at about 0.12 m/s actual. The commanded 0.15 m/s is still far above stiction (about 0.03 m/s), so there is no stall.

| r_d | k_t 0.15, k_r 1 (now; = old law) | Python gains k_t 0.2, k_r 0.5 (not used) |
|---|---|---|
| 0.01 m | 0.048 m | 0.088 m |
| 0.05 m | 0.077 m | 0.115 m |
| 0.10 m | 0.116 m | 0.146 m |
| 0.20 m | 0.209 m | 0.229 m |
| 0.40 m | 0.404 m | 0.417 m |

- **Why the Python gains give a larger orbit:** the smallest orbit scales like `k_t/sqrt(k_r)`. Python has 33% more tangential speed and half the radial gain, so its floor is about 1.9× bigger.
- **Comparing with simulation:** the Python simulations must be re-run with `base_orbital_speed` (k_t) = 0.15 and `radial_gain` (k_r) = 1. See 5.7.

### 5.5 "Is it real or hardcoded?" checks
- **The model contains a fake field.** `Fixed_Vortex_Simulated_Environment` is an analytic field at (0, 0), with hue = atan2(z, x) **in radians** and sat = 0.54·r + 0.3. Its outputs are **only logged** (now `simfield_hue` / `simfield_sat`; earlier `hue_anticipated*`, `sat_anticipated*`); they're not wired into any controller. Don't confuse them with sensor data.
- **The 2026-11-11 runs were real:**
  - The critical-point estimate stayed at (0.41, −0.12) to within about 2 cm across separate runs, while the cluster moved over about 24 cm.
  - Its correlation with the centroid was −0.2 to −0.4. An estimate tied to the cluster would sit near +1.
  - It had sensor-like jitter of about 1 cm per step.

### 5.6 Old vs new kinematics, and the Python (Mac) code
- **The old Simulink kinematics:** inconsistent (the inverse kinematics are exact only when p = q).
- **The Mac Python `cluster_builder/src/control/kinematics.py`:** its θc in the forward kinematics is **π off** from its inverse kinematics. It should be `atan2(y1 − y2, x1 − x2)`. This is not fixed; `clusterbuilder/leaf_blocks.py` `TripleBlock` is the correct reference.
- **Python gains:** `position_gain` 1.0, `angle_gain` 0.1 (β), and no θc control. Simulink uses gain 1 on p, q and β; θc control was switched off on 2026-11-12 (5.2).

### 5.7 How the real robots follow a command, and what that means for the Python simulator
**Fit from the 2026-11-11 data** (three orbit runs, 1,194 samples, both axes):
- **Method:** I rebuilt the Orbiter's centroid command `u(k)` at every step from the logged data and fitted the measured centroid velocity `v(k)` (position difference / 0.1 s) to

  `v(k) = 0.40·v(k−1) + 0.47·u(k)`        (R² = 0.98; adding a one-sample delay fits no better)

- **Smoothing:** the robots have a "momentum" of about **0.40**.
- **Speed reached:** they only reach **0.78 ≈ 0.8** of the commanded speed (steady-state gain 0.47/(1 − 0.40)).
- **Validation:** fed back into the closed orbit loop, this model predicts every measured orbit radius to within 1 cm (0.103 vs 0.105, 0.116 vs 0.107, 0.048 vs 0.057 m).
- **Caveat:** the third run's settings were taken from its file name.

**Re-fit 2026-11-15 from the campaign logs: there's a one-step delay, and almost no momentum.** This supersedes the 0.40 / 0.44 "momentum" above. **The full write-up, including how it was found, is in `trunk/campaign_results/2026-11-15_robot_response_model.md`.**
- **Data and script:** the 2026-11-13/15 campaign (clean copies), per robot, 550 robot-runs. Runs where a robot reached less than 0.75 of its commanded speed were dropped (low battery). Measured velocity = pose difference / wall-clock step (median 0.100 s, p5–p95 0.094–0.109). Script: scratchpad `fit_coulomb_delay.m`.
- **Model:** the Python `Omnibot` "coulomb" structure plus an input delay d and a gain g:
  - `free = a·v(k−1) + (1−a)·g·u(k−d)`
  - `v(k) = free·max(0, 1 − (1−a)·v_k/|free|)`

| Fit | a (τ) | g | v_k (m/s) | 1-step RMSE | free-run RMSE (m/s) |
|---|---|---|---|---|---|
| **delay 1 step (0.1 s)** | **0.11 (0.045 s)** | **0.85** | **0.000** | **0.030** | **0.032** |
| no delay | 0.47 (0.13 s) | 0.89 | 0.005 | 0.037 | 0.037 |
| Python `OmniCluster` defaults (a 0.70, g 1, v_s = v_k 0.025, no delay, cap 0.4) | – | – | – | – | 0.045 |

- **Per robot (delay 1):**

  | Robot | a | g | v_k (m/s) |
  |---|---|---|---|
  | celeste | 0.17 | 0.85 | 0.009 |
  | tidal | 0.06 | 0.88 | 0 |
  | pacific-blue | 0.09 | 0.89 | 0.001 |
  | redwood | 0.09 | 0.88 | 0.006 |

- **Where the delay is:** the logged command at step k matches the Orbiter law evaluated on the same step's `cluster_state` and `field_cp` exactly, so there is no delay inside the controller. The step sits between the command and the motion: radio and Decabot latency, or OptiTrack latency, which is equivalent for the closed loop.
  - **Why the earlier "momentum ≈ 0.44" appeared:** without the delay in the model, the lag term absorbs it.
  - **Why the 11/11 fit missed it:** it used commands *reconstructed* offline, which blurred the timing.
- **Friction:** kinetic Coulomb friction is about **0** for moving robots. At steady commands, speed/command is flat at 0.80–0.86 from |u| = 0.10 to 0.25 m/s, which is a proportional gain, not an offset.
  - **Static stiction (breakaway) can't be identified from this campaign:** commands were almost never below 0.1 m/s. It needs a low-speed test: one robot, from rest, steps of 0.01, 0.02 … 0.08 m/s, 3 s each.
- **Speed cap:** no robot ever exceeded 0.41 m/s. Commands of 0.35–0.45 m/s gave median speeds of only 0.21–0.27, so the robots saturate at about **0.3 m/s**. That's the Decabot spec and the old Omnibot default; `OmniCluster` raised it to 0.4.
- **Proposed sim values (100 ms step):**
  - **a ≈ 0.1**, with a **one-step input delay**;
  - **gain ≈ 0.85**: battery-dependent, 0.8–0.93 healthy;
  - **v_k ≈ 0** (≤ 0.01);
  - **v_s:** keep 0.025 as a placeholder until the low-speed test;
  - **cap 0.3 m/s**.
  - `Omnibot` has neither a gain nor a delay parameter, so both would need adding.
  - **If a delay can't be added,** use the best no-delay compromise, a 0.47 / g 0.89.

**The Python simulator's robot model is different** (`from_mac/VF_Robot/VF_Robot/cluster_builder/src/robot/omnibot.py`): `momentum_alpha = 0.7`, and it reaches the full commanded speed (gain 1), with a 0.3 m/s cap and 0.025 m/s stiction.
- So the simulated robots lag more than the real ones but are faster in steady state.
- **Effect:** for the same orbit law, the simulator predicts a bigger orbit than the hardware gives (about 0.156 m vs 0.088 m at r_d = 0.01).

**Action: re-run the Python simulations with the measured robot model** once the data campaign is complete:
- **Proposed starting values:** `momentum_alpha ≈ 0.4` and a command gain ≈ 0.8, i.e. `v = 0.4·v + 0.6·(0.8·u)`. omnibot.py has no gain parameter, so one would need to be added.
- **Re-fit first, with all the campaign's data:** the new logs make it direct and per-robot. Fit `cmd_body_rN` (or `cmd_robots_selected`) against the measured velocity from `pose_rN`; `timing` gives the true step length.
- **Re-check:** that the 0.4 / 0.8 values hold at other speeds and radii, then update this section.
- **Same re-run: use the hardware's Orbiter gains.** Python's orbiter (`critical_point_orbiter_plane_fitting`) defaults to `base_orbital_speed` 0.2 and `radial_gain` 0.5. The hardware uses **k_t 0.15, k_r 1** since 2026-11-12 (5.4), with `max_radial_speed` 0.3 (same) and no θc control (same). Set them in the sim run configs, not as code changes, so the sims and the hardware campaign are directly comparable.
- **Battery state changes the robot model, so check it before fitting.** On 2026-11-13, redwood's speed reached (actual/commanded) fell from 0.81 to 0.34 over 7 runs (about 10 min) as its battery ran down, then jumped to 0.92 with new batteries. Healthy robots read 0.8–0.93.
  - **Before fitting,** compute each robot's ratio per run and drop or flag runs where any robot is below about 0.7.
  - **Affected so far:** 4-robot `vortex_4robots_radius_of_0_point_01_run3..7`.
  - **Details:** `trunk/robots_4/README_four_robot.md`, "First hardware runs".
- **Checklist for the post-campaign re-run (all sims, 3-robot and 4-robot):**
  1. re-fit momentum and gain from all the campaign logs (done 2026-11-15 for the vortex and saddle data so far: one-step delay, a ≈ 0.1, g ≈ 0.85, v_k ≈ 0, cap 0.3; see the re-fit above). Re-run `fit_coulomb_delay.m` when the campaign ends;
  2. set the robot model to those values;
  3. orbiter k_t 0.15, k_r 1, radial clip 0.3;
  4. no θc control;
  5. the same r_d values as the hardware runs;
  6. compare radius and speed with the logged `orbit_radius*` and `pose_*`.

### 5.8 The critical-point estimate gets worse far from the centre (kept raw on purpose)
**Decision (user, 2026-11-13):** the critical-point estimate is fed to the Orbiter **raw**: no gate, no filter. This is intentional, so the hardware results show the method as it is. The fixes below are written down for later; none is implemented.

**What the 2026-11-13 campaign shows** (29 runs; r_d 0.01, 0.1 and 0.2):

| Cluster distance from the centre | Median estimate error | Steps off by > 10 cm |
|---|---|---|
| 0–0.15 m | 2.4–3.0 cm | ≤ 1% |
| 0.15–0.20 m | 3.9 cm | 9% |
| 0.20–0.25 m | 5.1 cm | 8% |
| 0.25–0.30 m | 7.0 cm | 18% |

- At r_d 0.2, 11% of steps were more than 10 cm off, and a few were metres off (within-run std up to about 1 m).
- The per-run median is still repeatable to about 1 cm, and the orbit still averages 0.209 m, because the controller averages through the noise.
- **Distorted log:** the logged `orbit_radius` (measured to the *live* estimate) is inflated to about 0.25 m at r_d 0.2. Measure the radius against the run-median critical point instead.

**Why: extrapolation, not worse sensors.**
- **The method:** each step, `Criticalpointfinder` fits an exact linear field through the 3 readings and solves for where it is zero.
- **Near the centre:** the robots straddle the zero, so the fit **interpolates**. The error floor is about (direction noise ≈ 7°) × (robot spacing ≈ 0.17 m) ≈ 2–3 cm.
- **Far from the centre:** the zero is outside the triangle, so the fit **extrapolates**. Two effects grow with distance d:
  - in a vortex the vector grows with r, so a fixed direction error becomes a bigger vector error (∝ d);
  - the error in the fitted slope, measured across a 0.3 m triangle, is multiplied by d when projected back to the centre.

  So the error grows roughly like (direction error) × d × (1 + d/triangle size).
- **Checked:** the measured reading noise (6.9° median, 17° p90 direction error; 5 cm-equivalent magnitude noise), applied to an ideal vortex with the same triangle and fit, reproduces the trend (about 4 → 8 cm). Readings are not delayed relative to OptiTrack: a 0-step shift fits best.

**What protects against a bad estimate today:**
- **The Orbiter's radial-speed clip, ±0.3 m/s (since 2026-11-12).** A wild estimate can command at most √(0.15² + 0.3²) ≈ 0.34 m/s for that step, in a wrong direction. It limits *how hard* the cluster reacts, not *whether* it reacts.
- **The old Orbiter's clip, ±5 m on the radius error** (`e = max(min(r_d − r, 5), −5)`, before 2026-11-12). With K_p 1 that allowed up to 5 m/s, so in practice it only stopped the absurd cases.
- **What the finder itself checks:** only `det(J) < 1e-10` → NaN. A NaN estimate gives NaN commands; none occurred in the campaign so far.

**How each fix would help, if wanted later** (in order of cost):
1. **Distance gate** ("ignore estimates more than X from the cluster"):
   - **How:** in `Critical_point_predicter`, if ‖cp − centroid‖ > X (or the estimate is NaN), hold the last good estimate.
   - **What it fixes:** it removes the metres-away jumps, which are the dangerous ones.
   - **Choosing X:** for a 30 cm triangle, X ≈ 2·r_d + 0.3 m or a fixed 0.6–1 m would catch them.
   - **Limits:** it does nothing for the frequent 5–10 cm errors at 0.2–0.3 m.
2. **Median or low-pass filter on the estimate:**
   - **Why it fits:** the critical point is stationary, so a running median over about 10 steps (1 s) removes outliers.
   - **Effect:** it cuts the jitter roughly √10 ≈ 3× (about 2.5 cm → under 1 cm near the centre, about 7 → 2–3 cm at 0.3 m).
   - **Side effect:** it would also reduce the ~2 cm "estimate leads the cluster" bias that widens the small orbits (5.4 / campaign notes). A median preserves a real step change; a low-pass lags it.
3. **Bigger triangle for big orbits:**
   - **Why:** the amplified term shrinks roughly as 1/(triangle size). p = q = 0.45 instead of 0.3 cuts it by about a third.
   - **Cost:** the readings average over a larger patch of the map.
4. **Better sensor calibration:** celeste and pacific-blue read about 10° off, tidal about 4°. The whole error scales with the direction error, so re-tuning the two worst nets helps everywhere.
5. **Fit a vortex model instead of a general linear field:** a known field type uses the readings' directions only (each reading's perpendicular points at the centre). It ignores the noisier saturation and is better conditioned. It is the biggest change and is specific to vortices.

### 5.9 Batteries fade within one session, and it shows up as a slow, lagging robot (2026-11-13)
**What happened:** in the first 4-robot session, redwood got steadily weaker from run to run, and nothing else changed.
- **Speed reached:** about 81% of its commanded speed at 22:02, then 76 → 69 → 61 → 42 → 38 → **34%** by 22:11. The other three robots held steady at 80–93%.
- **Shape:** the formation drifted. The diagonal that includes redwood (d2) stretched from 0.31 to 0.34 m (setpoint 0.30) as redwood fell behind.
- **New batteries:** redwood went to **92%**, the best of the four, and d2 came back to 0.298 m.

**Lessons:**
1. **A start-of-day voltage check isn't enough.** The SOP's voltage check (logic 12–13.2 V, motor 6–7.2 V) is done once at setup. A pack can pass it and still fade noticeably within about 10 minutes of driving.
2. **The logs give you a battery gauge for free.** For each robot, the median of |actual velocity| / |commanded velocity| over a run:
   - **Inputs:** actual velocity from `diff(pose_rN(:,1:2))/0.1`; commanded from `cmd_robots_selected`, the robot's [vx vz] columns.
   - **Healthy:** 0.8–0.93.
   - **Swap batteries:** below about 0.7. A steady downward trend over consecutive runs means the battery is going.
   - **Script:** the scratchpad script `robot_health_4R.m` computes it per robot per run.
3. **A tired battery looks like a control problem.** The weak robot lags, the shape stretches along its diagonals, and the orbit gets sloppier. Before blaming gains or kinematics, check every robot's speed ratio.
4. **Battery fade is not a heading problem.** Redwood's 7–11° direction error was the same before and after the new batteries. A constant direction error is a Motive heading-zero issue (5.1); a falling speed ratio is the battery. The logs separate the two cleanly.
5. **Flag affected runs in the data.** Runs with a weak robot change the robot model (5.7), so leave them out of, or flag them in, robot-response fits and any speed-dependent comparison.
   - **So far:** `vortex_4robots_radius_of_0_point_01_run3..7`.
   - **Note the swap:** put it in `run_notes` when you change batteries (e.g. `'redwood new batteries'`), so the change is recorded with the run.

### 5.10 4-robot model: where each robot stands matters (2026-11-13)
**The rule:** the 4-robot kinematics treat robots 1→3 and 2→4 as the **diagonals**. Going around the square, place them celeste → tidal → pacific-blue → redwood.

**What went wrong:** in the first test the robots were placed redwood → pacific-blue → celeste → tidal.
- **Bad geometry:** the model's "diagonals" were sides, so its intersection ratios came out as r1 = −4.8 and r2 = −2.8.
- **Huge commands:** those shape errors produced 1–5 m/s commands, and the robots lurched around at their limit.
- **Wiring was fine:** each robot followed its own command; only the placement was wrong.

**Direction matters too.** Numbering the robots clockwise in Motive x–z gives φ = +π/2, which matches the setpoint. Counter-clockwise gives −π/2, and the controller would try to mirror the formation.

**Check:** in `cluster_state`, r1 and r2 should be in [0, 1] (≈ 0.5) and φ (column 8) ≈ +1.57. Details: `trunk/robots_4/README_four_robot.md`, "First hardware runs".

---

## 6. What gets logged (logging suite, added 2026-11-12)

Every logging block is named `LOG <variable>` and coloured **cyan**. They only tap existing signals; none of them feeds control. All are `Array` format with N rows (one per 0.1 s step, matching `tout`). **r1 = celeste, r2 = tidal, r3 = pacific-blue.**

**Load runs with `S = load_run(file)`** (`trunk/robots_3/load_run.m`). It works on old and new files and gives you both the old and the new names, so older plotting scripts work after replacing `load(f)` with `S = load_run(f)`. Column meanings are also stored in every new run as `run_config.log_legend`.

**Stale variables in campaign files (checked 2026-11-15).** The runs are saved with `save` of the whole workspace, so each file also carries leftovers from earlier sessions. A logged variable is fresh only if the run itself wrote it. Concretely:
- **Every 2026-11-13 file:** about 24 old-name variables from 2025, such as `robot1_pose`, `p`, `q` and `beta` at 601 rows, `hue` / `saturation` as timeseries, and `ans`. `load_run` keeps them as `<name>_STALE`.
- **4-robot vortex files:** stale **3-robot** variables: `field_cp`, `orbit_radius`, `simfield_*` and `cmd_*_vectorsum/ascent/descent`. For 4-robot runs, use `field_cp_jacobian` and `orbit_radius_jacobian`.
- **3-robot saddle files, r_d 0.01–0.3:** stale **4-robot** variables from 11/13: `pose_r4`, `field_cp_jacobian`, `orbit_radius_jacobian`, `FK_results`, `*_r4`, `*_jacobians`.
  - The r_d 0.4 runs, saved after MATLAB was restarted at about 22:10, don't have them.
- **Rules:**
  - Decide the robot count from `size(cmd_robots_selected, 2)/3`, or from the file name. **Never** decide it from whether `pose_r4` exists. `load_run` was fixed to do this on 2026-11-15; before the fix it would have treated the saddle files as 4-robot runs.
  - Read only the variables your model writes.
  - The real logs were always overwritten each run, so the data itself is fine.
- **Duplicate file:** `Vortex/three_robot/vortex_001/vortex_4robots_radius_of_0_point_01_run1.mat` is a misplaced copy of the 4-robot file of the same name. `campaign_report.m` deduplicates it by name.
- **Prevention:** run `clear` before each session, or save only the logged variables, for example with `save(f, '-regexp', '^(pose_|sensor_|field_|orbit_|cluster_state|setpoint|err_|cmd_|timing|tout|run_config|diag_)')`.

| Variable | Cols | What it is | Old name |
|---|---|---|---|
| `pose_r1/r2/r3` | 3 | OptiTrack [x z heading] | `robot1_pose` … |
| `sensor_rgbk_raw_r1..3` | 4 | **Raw** Decabot counts, in the Decabot's output order **[K(clear) R G B]** (`Decabot.m` `outputImpl`). The name says rgbk, but K comes first. | – |
| `sensor_rgbk_norm_r1..3` | 4 | After `(x−120)/1300` (K: `(x−800)/3500`) and clipping to [0, 1]; this is the neural-net input | `rgbk_robot_1..3` |
| `sensor_hue`, `sensor_sat` | 3 | After the nets and post-processing, 0..1. This is what control uses. | `hue`, `saturation` |
| `field_uv` | 6 | Decoded [u1 v1 u2 v2 u3 v3] going into the critical-point finder | – |
| `field_jacobian` | 8 | [J11 J12 J21 J22 trace det curl type]; type 1 vortex, 2 spiral, 3 node, −1 saddle, 0 degenerate/NaN | – |
| `field_cp` | 2 | Critical-point estimate [x z] | `detected_critical_point` |
| `orbit_radius` | 2 | Orbiter [current radius, desired − current] | – |
| `cluster_state` | 9 | [xc zc θc p q β φ1 φ2 φ3] | `cluster_pose` (`p q beta` = cols 4–6) |
| `setpoint` | 6 | Set point controller's desired [xc zc θc p q β] | – |
| `err_setpoint` | 6 | `Compute_error` outputs [xc zc θc p q β] (xc, zc clipped to ±0.3) | – |
| `err_jacobian` | 6 | `Jacobian_primitive` errors [xc zc θc p q β] | – |
| `cmd_cluster_<prim>` | 6 | What enters that primitive's inverse Jacobian: [v_xc v_zc ω_c v_p v_q v_β] | `desired_velocities` (vectorsum), `desired_velocities_gradient_descent` (actually ascent) |
| `cmd_robots_<prim>` | 9 | That primitive's per-robot global velocities [vx1 vz1 w1 … vx3 vz3 w3] | – |
| `cmd_robots_selected` | 9 | What the root switch chain passes on, i.e. what is actually sent | – |
| `cmd_body_r1..3` | 3 | Body-frame command sent to each Decabot | – |
| `simfield_hue`, `simfield_sat` | 3 | **Not sensor data.** The analytic simulator (5.5). Robot 3's columns were swapped before 2026-11-12 and are now fixed. | `target_*_adjusted`, earlier `hue_anticipated*` |
| `timing` | 2 | [sim time, wall-clock posix s]. `diff(timing(:,2))` is the real duration of each step. | – |
| `run_config` | struct | Written by the model's StopFcn (`log_run_config.m`): model file and SHA-256, whether there were unsaved edits, every switch position, every Constant and Gain, the full code of every MATLAB Function (K_p, orbit_speed, …), robot names, OptiTrack settings, your `run_notes`, and a column legend | – |

`<prim>` is one of `setpoint`, `jacobian`, `vectorsum`, `ascent`, `descent`. All five primitives run all the time, so all five are logged. `cmd_robots_selected` tells you which one was actually driving.

- **Before a run**, optionally set `run_notes = 'map rotated 90°, fresh batteries, orbit test r=0.05'` in the workspace; it gets stored in `run_config.notes`.
- **Save only what you need:** if you save the whole workspace, stale variables from earlier sessions come along. Check that every variable has `numel(tout)` rows.
- **Rebuilding the suite:** the logging was added by `tools/add_logging_suite.m`, which can be re-run safely (it skips blocks that already exist). The logging-only MATLAB Function code is in `tools/*_log.m`.
- **Compiling needs the hardware:** compiling or updating the model runs the hardware blocks' setup code, which connects to OptiTrack and DecabotHelper.
- **Pacing:** the model has a `Real-Time Pacer` block (simTimePerRealTime = 1). In an offline test with the hardware stubbed out, it held 0.096 s per step; `timing` shows what you get on the robots.

---

## 7. Checklist before a run
1. Follow the OptiTrack SOP: calibration, ground plane, rigid bodies created with the **blue wheel pointing +x**, names matching the model's `rigidBodyNames` (`celeste`, `tidal`, `pacific-blue`).
2. Run `neural_network_loader.m`, and `javaaddpath C:\DecaMATLAB\SocketShim.jar`.
3. **Manual Switches:** is the intended primitive selected?
4. **Setpoints and gains:**
   - shape (`Cluster Shape Variables`), θc (`Theta_Set Point1 `, and note the trailing space in the name), Orbiter radius (`Jacobian_primitive/Constant1`);
   - direction flag (`Jacobian_primitive/Constant`, +1 = Python's direction), Orbiter `k_t` / `k_r` / `max_radial_speed`, and `thC_gain` (0 = no spin).
5. **Hue decode:** matches how the map is laid (section 4).
6. Fixed step 0.1 s; set StopTime.
7. **Save the model before you run.** `run_config` records the settings automatically (and flags unsaved edits), but a saved model means the SHA-256 identifies exactly which file was used. Don't rely on the `.mat` file name.
8. Optionally set `run_notes` (map placement, batteries, purpose) before pressing Run.
9. **Batteries, between runs:** check each robot's speed ratio from the last run (5.9). Swap batteries on any robot below about 0.7 or trending down, and note the swap in `run_notes`.
10. **4-robot model:** robots go around the square in the order celeste → tidal → pacific-blue → redwood (5.10). On the first steps, check that r1 and r2 are ≈ 0.5 and φ ≈ +1.57.

---

## 8. Files
- `three_robot_cluster_all_primitives.slx`: the current model.
- `three_robot_cluster_all_primitives_og.slx`: before the 2026-11-11 kinematics port.
- `three_robot_cluster_all_primitives_BACKUP_pre_kinematics_port.slx`: another pre-port backup.
- `three_robot_cluster_all_primitives_BACKUP_pre_logging_20261111.slx`: just before the logging suite was added.
- `load_run.m`, `log_run_config.m`: load any run with both name sets; the run-configuration recorder (called by StopFcn).
- `tools/add_logging_suite.m` and `tools/*_log.m`: builds the logging suite; code of the logging-only MATLAB Function blocks.
- `../../data_hardware_campaign_11_11_2026/`: the 2026-11-11 runs.
- `../../Plots_to_print/`: the map generator notebook and the printed images.
- `../Optitrack_demo/OptiTrack_SOP.md`: the lab procedure.
