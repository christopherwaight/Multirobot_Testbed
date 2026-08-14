# 4-robot saddle seeking: a learned policy that beats the hand-derived law

**Result.** On 2500 paired held-out fields, PPO reaches **70.1%** success
against the gain-swept analytic controller's **66.4%**: **+3.7%, 95% CI
[+2.1%, +5.4%], P(better) = 100%.** The interval excludes zero.

The starting point was 34.5%, and the analytic law was winning 66.5% to 34.5%.
Almost all of that gap was **four defects in the RL environment**, not a
limitation of the method. See `HISTORY.md` for the wrong turns, including a
conclusion I reported and later had to retract.

| controller | success | vs analytic | 95% CI |
|---|---|---|---|
| tuned analytic (best of a 48-cell sweep) | 66.4% | reference | |
| **PPO E5** (BC warm start + fine-tune) | **70.1%** | **+3.7%** | **[+2.1%, +5.4%]** |
| either succeeds (oracle upper bound) | 76.8% | +10.4% | [+8.9%, +11.9%] |

## Why it wins, concretely

The analytic law takes a Newton step, `-H^-1 g`, which is attracted to **every**
critical point equally: the gradient vanishes at a hilltop just as it does at a
saddle, and the 4-robot Hessian estimate is always traceless, so `det < 0`
always and everything *looks* like a saddle. The law cannot tell it is being
captured by the wrong one.

Measured on 14 `log_sum_exp` fields, where PPO's margin is largest (+32.3%):

| | reaches the saddle | parks on a WELL | other |
|---|---|---|---|
| analytic | 7/14 | **2/14** | 5/14 |
| PPO | **11/14** | **0/14** | 3/14 |

PPO is not running Newton, so it is not bound by Newton's attraction to
stationary points. It never once parked on a well.

## Read in this order

| | |
|---|---|
| **`README.md`** (this file) | the result, why it wins, how to reproduce |
| `HISTORY.md` | the four defects, the retracted conclusion, superseded numbers |
| `compare_analytic_vs_ppo.ipynb` | **run both controllers on one field by hand** |
| `outputs/figures/fig9_paired_difference.png` | the headline, with confidence intervals |
| `outputs/figures/fig1_estimator_mechanism.png` | why the formation has to rotate |
| `python3 estimator.py` | every claim about the estimator, verified numerically |

## Quickest way to see it yourself

```bash
VENV=../../Vector_Fields/VF_Robot/venv/bin
$VENV/python3 estimator.py        # the estimator's limits, proven not asserted
jupyter notebook compare_analytic_vs_ppo.ipynb   # side by side, pick any field
```

The notebook's last cell lists the seeds where exactly one controller
succeeded. `SEED = 500011` is a good one: the analytic law drives out to the
8 m limit while PPO parks at 0.009 m on the identical field.

---

Can a learned policy do what the hand-derived rotating-Hessian law cannot once
momentum lag, stiction, and velocity limits are in the loop?

## Where this came from

`../saddle_point_4_robot.ipynb` navigates a 4-robot square to a scalar-field
saddle by estimating a Hessian from the four corner readings and servoing the
formation angle onto the Newton direction. Its own summary table:

| rotation mode | final distance | outcome |
|---|---|---|
| none | 9.81 | diverges |
| single, pi wrap | 0.20 | converges |
| quad, pi/2 wrap | 0.81 | limit cycle |

That notebook teleports the cluster. There are no actuator dynamics. The
production port, `scalar_newton_with_rotation` in
`../../Vector_Fields/VF_Robot/src/control/quad_primitives.py`, runs the same
idea through the real plant and does not converge.

## What the estimator actually recovers

Measured, not assumed. `python3 estimator.py` reproduces all of it.

Four readings cannot determine a 2D quadratic, which needs six coefficients.
The map from readings to (Hessian, offset) has **rank 3**: two degrees of
freedom go to the gradient, leaving exactly **one scalar** for curvature. The
estimate comes out as

```
H_est = m * Ref(theta_f),     m = M * cos(2 (theta_f - theta_field))
```

where `theta_f` is the formation angle, `theta_field` is the field's principal
axis, and `Ref(t)` is reflection about the line at angle `t`. Three
consequences:

1. The **gradient is exact** for a quadratic field, at every formation angle.
2. `H_est` is always traceless and always symmetric, and **its eigenvectors are
   the formation's own axes, not the field's**. Rotating the formation rotates
   the reported eigenframe with it. This is the part that is easy to get
   backwards: the estimate carries no direct information about where the
   field's eigenframe points. That reaches the controller only through `m`.
3. The Newton step is therefore exactly

   ```
   -H_est^-1 g  =  -(1/m) Ref(theta_f) g
   ```

   the negative gradient reflected about the formation's own axis. The step
   **direction** is set by the gradient and the formation angle alone.

So rotation is not merely a way out of a blind spot. Turning the formation is
the controller's only means of steering the step direction at all. It also
predicts that a *non*-rotating formation still converges whenever its fixed
angle happens to sit near a field eigen-axis, which is roughly half of random
initializations. `baselines.py` measures exactly that, and finds it.

## What the plant allows

Rotating a square of ring radius `R` at rate `omega` needs tangential speed
`omega * R`. That is bounded above by the velocity cap and below by an actual
dead zone, because one momentum-filter step realizes only `(1 - alpha) = 0.3`
of the command and anything under the stiction threshold is zeroed before it
can accumulate:

| R (m) | omega_min (rad/s) | omega_max (rad/s) |
|---|---|---|
| 0.08 | 1.042 | 3.750 |
| 0.15 | 0.556 | 2.000 |
| 0.25 | 0.333 | 1.200 |
| 0.40 | 0.208 | 0.750 |

Both bounds scale as `1/R`, so the usable band is only 3.6x wide at any size
and shrinking the formation shifts the whole band upward. A command below
`omega_min` produces **no** rotation, not slow rotation. Near convergence the
angle servo wants a small `omega`, which lands in the dead zone. That couples
the formation-size reward directly to the physics rather than making it just an
anti-cheat device.

## Files

| File | What it does |
|---|---|
| `saddle_fields.py` | Eight randomized scalar-field families, every saddle in closed form |
| `estimator.py` | The C(4,3) plane-fit estimator, and verification of everything above |
| `quad_saddle_env.py` | `gymnasium.Env` wrapping the repo's `QuadCluster` |
| `baselines.py` | The notebook law under both plants, plus a gain sweep |
| `train_ppo.py` | stable-baselines3 PPO driver (no curriculum by default; `--curriculum` opts into the old 3-stage schedule) |
| `evaluate.py` | Held-out rollouts, metrics, comparison tables |
| `visualize.py` | All figures |
| `bc_pretrain.py` | Behaviour-cloning warm start from the analytic law |
| `compare_analytic_vs_ppo.ipynb` | **Interactive side-by-side.** Run both controllers on the same field by hand, and list the fields where they disagree |
| `compare_helpers.py` | Thin wrappers the notebook uses |
| `outputs/` | Checkpoints, logs, figures. Not committed |

## The eight field families

The original six (`quadratic`, `log_sum_exp`, `gaussian_pair`, `cubic_perturbed`,
`streamfunction_quad`, `double_gyre_psi`) are documented in `saddle_fields.py`'s
module docstring. Two more were added specifically to stress "does the policy
arrive at the wrong critical point," each by a different closed-form mechanism
than anything else in the set:

- **`quartic_wells`**: a quadratic saddle plus a quartic double-well term on
  the weak eigen-axis, `eps*(u2^2-a^2)^2`. This term has no linear part, so
  its gradient still vanishes at the saddle and its Hessian contribution there
  is a single known scalar, both exact. It creates two more exact stationary
  points at `s +/- a*v_weak` (`a` random, known), and at each one the local
  Hessian works out to `diag(l_pos, +8*eps*a^2)`, both positive, i.e. genuine
  local minima flanking the saddle. Pure polynomial construction, verified
  symbolically (see the derivation in `saddle_fields.py`).
- **`rational_envelope`**: a quadratic saddle multiplied by a Lorentzian decay,
  `phi = (1/2 d'Md) / (1 + k|d|^2)`. Because the quadratic vanishes to second
  order at the saddle, the envelope's own Taylor expansion there doesn't
  contribute, so `grad(phi)(s)=0` and `Hess(phi)(s)=M` exactly, also verified
  symbolically. Far from the saddle the envelope forces `phi` toward a
  finite, direction-dependent limit, so it necessarily folds back and creates
  secondary critical points along most radial directions, a different
  mechanism (multiplicative decay) from every other family's additive or
  blended construction.

## Known limitation, accepted as such

Two families, `gaussian_pair` and `streamfunction_quad`, are hard enough that
even the **best gain-swept hand-derived controller** fails on them (median
final distance ~3, effectively domain-exit range; see the per-family table
below). This is not a PPO weakness: it shows up identically under the tuned
classical law. The 4-robot single-snapshot estimator's information content
appears to be the actual limit on those two shapes, not the controller
processing it. Treated here as an accepted, documented limitation rather than
a bug to keep chasing.

## Design decisions worth knowing

**The plant is not reimplemented.** The environment wraps `QuadCluster`
directly, so the momentum filter, stiction, velocity cap, and formation shape
servo are the same ones every other experiment in this repo runs against.

**The observation is a single frame with no privileged information.** No
gradient, no Hessian, no true saddle, no absolute position. 24 dimensions built
from the four readings and robot-relative geometry, split into direction and
log-magnitude so every channel is O(1) across six field families and a 5x range
of formation sizes. Per-robot velocities are included because the `alpha = 0.7`
filter makes position alone non-Markov.

**Episodes are not terminated on success.** An earlier version ended the
episode on reaching tolerance and paid a bonus, and PPO correctly learned to
*avoid* succeeding: ending at step 450 of 600 forfeits about 150 steps of
tracking reward worth far more than the bonus, so the policy hovered just
outside tolerance with an oversized formation. Reward 270, success rate 2
percent. Running the full horizon makes the reward and the metric agree, since
both then measure time parked on the saddle.

**The size bonus multiplies the tracking term rather than adding to it.** As an
additive term it pays out for shrinking anywhere in the domain, so the optimal
policy would collapse the formation and park far from the saddle. `R` is also
floored, with termination below it, because an unbounded size reward drives `R`
to zero where the readings become identical and the plane fits go singular.

**The dense shaping is potential-based**, `gamma * Phi(s') - Phi(s)` with
`Phi = -k e` (Ng, Harada, Russell 1999). That form provably leaves the optimal
policy unchanged, so it is a training aid rather than a change to the objective.

**The baseline is reported at its best swept cell, not its default gains.** The
defaults in `quad_primitives.py` were never retuned for the velocity form of
the law. Reporting only the default would confound "dynamics broke the law"
with "the gains were wrong", and the sweep shows that distinction is real: the
default `k_trans = 0.3626` is far from the best cell.

## Running it

```bash
VENV=../../Vector_Fields/VF_Robot/venv/bin

# one-time
$VENV/pip install gymnasium stable-baselines3

# verify the pieces
$VENV/python3 saddle_fields.py --draws 200     # closed-form saddles
$VENV/python3 estimator.py                     # the reflection law
$VENV/python3 quad_saddle_env.py --check --smoke

# baselines, with the gain sweep
$VENV/python3 baselines.py --episodes 40 --sweep

# train
$VENV/python3 train_ppo.py --timesteps 3000000 --envs 8 --tag ppo

# evaluate on held-out fields and draw everything
$VENV/python3 evaluate.py --n-eval 200
$VENV/python3 visualize.py --all
```

Throughput is about 3800 env steps/s on one core and plateaus near 2900
steps/s through PPO with 8 subprocess workers, so 3M steps is roughly 17
minutes.

### Ablations

Each answers one question. All are `train_ppo.py` flags.

| Flag | Question |
|---|---|
| `--obs-mode raw+est` | Does PPO beat the hand estimator, or just re-derive it? |
| `--action-mode no_rot` | Is rotation what the policy is actually exploiting? |
| `--action-mode fixed_size` | How much comes from the size knob? |
| `--rot-penalty 0.05` | Is there a low-excitation solution, or is max spin optimal? |
| `--sigma-z 0.02` | How does the blind spot interact with reading noise? |
| `--ideal-plant` | The 0-dynamics ceiling |
| `--curriculum` | Opt into the old 3-stage schedule; default is off (see below, it did not help) |


# Rebuilt environment: the corrected result

## The four defects, and what fixing them was worth

Each was confirmed against saved run data before being fixed, not inferred.

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | Domain exit cost `-10` against a ~120 return range, and the boundary lived in `c - saddle`, the one coordinate the observation hides | median PPO episode ended AT the wall (`e_final` 3.0 = `DOMAIN_HALF`) with `e_min == e_0`, i.e. zero progress; 41-47% early exit vs 25% for the baseline | exit is no longer terminal; only a far-field numerics guard at e>8, which truncates (bootstrapped) rather than terminates |
| 2 | Under `reward_mode='pure'`, driving out of bounds was **strictly optimal** above e ~ 1.04, by up to +200 return | closed-form discounted comparison, and `steps_mean` 300 vs 600 | follows from fix 1 |
| 3 | Potential shaping paid an idle agent `K(1-gamma)e = +0.02e` per step, more the further out it sat | do-nothing controller scored +21.9 on a field it never moved in | progress reward `K*(e_prev - e)`, exactly zero when stationary |
| 4 | The formation-size penalty did not clip R while the line above it did | live training episodes hit R ~ 1.9 m and scored -1.71/step against a -0.33 floor; the size term alone was ~1030 of a -1028 episode return | clip R, matching `size_gain` |

A fifth issue was introduced while fixing these and then removed:
`squash_output=True` with gSDE is numerically fragile in SB3 (recovering the
pre-squash action needs `atanh`, and saturated tanh is +/-inf), which NaN'd two
5M-step runs at ~1.8M. Plain diagonal Gaussian with `ent_coef` 0.02 -> 0.005
instead. `train_ppo.py` now aborts cleanly on non-finite policy weights so this
class of failure cannot silently burn a run again.

Two further reward problems surfaced during the rebuild and are worth recording
because both were counter-intuitive:

- A **pure state cost** (`reward = -e`) trains badly even with the termination
  fixed: episode return is dominated by the start distance, an uncontrollable
  draw from a 1.0-2.5 m annulus, and that variance swamps the advantage
  estimate. Measured at 1.35M steps: reward falling, success 1%, mean distance
  4.2. A progress term telescopes and removes that dependence.
- The historical tracking kernel `SIGMA_R = 0.35` is `exp(-e^2/0.1225) = 3e-4`
  at e = 1.0, i.e. **numerically zero across the entire start distribution**.
  A start-distance sweep of the partially-fixed policy showed exactly the
  predicted signature: final distance degrading 0.18 -> 3.42 as starts moved
  from 0.3-0.6 out to 2.0-2.5. `SIGMA_WIDE = 1.5` gives 0.64 at e = 1.0.

## Headline: 1000 held-out fields, paired

All controllers scored on the **same** 1000 fields, so the comparison is paired
and the reported interval is a 10,000-sample bootstrap over fields. Pairing
matters here: field difficulty varies far more than the controllers do, and at
n=200 a 3-point gap was inside noise.

| controller | success | diff vs analytic | 95% CI | P(better) |
|---|---|---|---|---|
| tuned analytic (best of 48-cell sweep) | 66.5% | reference | | |
| PPO E2, from scratch | 64.7% | -1.8% | [-4.7%, +1.1%] | 10.5% |
| BC clone of the analytic law, no PPO | 66.4% | -0.1% | [-2.1%, +1.9%] | 44.5% |
| PPO E4, BC + 1.5M fine-tune (128-wide) | 67.8% | +1.3% | [-1.5%, +4.0%] | 80.3% |
| **PPO E5, BC + 1M fine-tune (256-wide)** | **68.7%** | **+2.2%** | **[-0.5%, +4.9%]** | **94.7%** |

For scale: the same pipeline scored **34.5%** before the fixes. The four bugs
were worth roughly **30 points**.

### Confirmed at n = 2500: PPO beats the analytic law

The n=1000 result above put E5 at +2.2% with the interval grazing zero, which is
exactly the ambiguous case worth resolving rather than reporting. Re-scored on
2500 paired fields (seeds 500000-502499, a superset of the 1000):

| controller | success | diff | 95% CI | P(better) |
|---|---|---|---|---|
| tuned analytic | 66.4% | reference | | |
| **PPO E5** | **70.1%** | **+3.7%** | **[+2.1%, +5.4%]** | **100.0%** |

**The interval excludes zero.** On this benchmark, with the environment fixed,
a learned policy beats the tuned hand-derived law by 3.7 points. The n=1000
estimate was simply noisy; +3.7% sits comfortably inside its [-0.5%, +4.9%].

Per-family at n=2500, which shows where the win comes from:

| family | n | analytic | E5 | diff |
|---|---|---|---|---|
| log_sum_exp | 316 | 39.2% | **71.5%** | **+32.3%** |
| streamfunction_quad | 316 | 12.3% | **20.3%** | **+7.9%** |
| gaussian_pair | 303 | 24.4% | 25.4% | +1.0% |
| quartic_wells | 300 | 76.7% | 77.0% | +0.3% |
| double_gyre_psi | 325 | **96.9%** | 96.0% | -0.9% |
| rational_envelope | 297 | **80.8%** | 79.1% | -1.7% |
| quadratic | 321 | **99.1%** | 95.0% | -4.0% |
| cubic_perturbed | 322 | **99.4%** | 94.1% | -5.3% |

It is one big win (`log_sum_exp`, +32 points) plus a real gain on
`streamfunction_quad`, paid for with 4-5 points on the two easiest families.
Recovering those is the obvious remaining work and would put it near 74%.

**Caveat on selection.** Five configurations were tried (E1-E5) and the best is
reported, so the honest reading applies a multiple-comparison discount. The
margin survives it: even a Bonferroni-style widening over five comparisons
leaves the lower bound above zero. Checkpoint selection used validation seeds
900000+, disjoint from this 500000-range evaluation, so the held-out number
itself is clean.

E5 was not cherry-picked from a noisy screen: 7 of its 9 checkpoints scored
71-74.5% on 200 *validation* fields (seeds 900k+, disjoint from the 500k
evaluation range), so the whole fine-tuning trajectory sits above the analytic
law, not just the selected point. The gap between that 74.5% validation figure
and the 68.7% held-out figure is a reminder that a 200-field screen is still
optimistic when you select on it.

E5's recipe, for reproduction: behaviour cloning on 500 analytic-controller
episodes, 50 epochs, then PPO fine-tuning at `lr = 1.2e-4` with a 256-wide
network, checkpointing every 500k steps. The best checkpoint was at 1M steps;
fine-tuning past that slowly degrades success while raising time-in-tolerance.

## The more interesting result: they are complementary, not redundant

Per-family success at n = 1000:

| family | analytic | E2 (scratch) | E4 (BC+PPO) | E5 (best) | E5 - analytic |
|---|---|---|---|---|---|
| quadratic | **98.5%** | 88.7% | 88.7% | 94.0% | -4.5% |
| log_sum_exp | 40.8% | 64.2% | 65.0% | **69.2%** | **+28.3%** |
| gaussian_pair | 22.0% | **27.6%** | **29.9%** | 22.0% | 0.0% |
| cubic_perturbed | **100.0%** | 92.9% | 93.7% | 94.4% | -5.6% |
| streamfunction_quad | 14.3% | 17.5% | 16.7% | **21.4%** | **+7.1%** |
| double_gyre_psi | **97.5%** | 91.0% | **98.4%** | 96.7% | -0.8% |
| quartic_wells | **76.2%** | 72.3% | **76.2%** | **76.2%** | 0.0% |
| rational_envelope | **81.9%** | 62.9% | 74.1% | 75.9% | -6.0% |
| **overall** | 66.5% | 64.7% | 67.8% | **68.7%** | **+2.2%** |

The learned policy wins precisely where the analytic law fails and loses where
it is already near-perfect. Splitting the 1000 fields by who succeeds:

```
analytic only succeeds : 11.8%
PPO only succeeds      : 10.0%
both                   : 54.7%
neither                : 23.5%
                        -------
either (oracle)        : 76.5%   +10.0% over analytic, CI [+8.2%, +11.9%]
```

So **10 points are demonstrably on the table** for any rule that can tell,
online, which controller is failing. That is a stronger and more useful result
than either controller's headline number.

## Why the obvious way to collect those 10 points does not work

A handover controller is implemented (`baselines.handover`) that runs the
analytic law and switches to the policy once its own reading spread stops
improving. It does **not** help: 65.8% at every patience setting tried, against
68.3% for the analytic law alone on the same fields.

The reason is the same limitation `estimator.py` documents. At a wrong critical
point the gradient is also zero, so reading spread cannot distinguish "converged
on the saddle" from "stuck on an extremum". And the 4-robot estimate is always
traceless, so its Hessian is always indefinite and always *looks* like a saddle
regardless of the truth. A single 4-robot snapshot genuinely cannot tell the two
apart; the information is not there to trigger the switch. Collecting the 10
points needs either a fifth robot at the centroid, or memory across time, or an
explicitly curvature-aware objective.

## Honest caveats

- **Selection asymmetry remains.** The analytic law's gains came from a 48-cell
  sweep selected on `e_final_median`, the statistic the table reports. PPO runs
  were hand-configured, with checkpoints screened on 40 fields (E1-E4). That
  screen is itself noisy and optimistic: E3 screened at 77.5% on 40 fields and
  delivered 63.0% on 200.
- **`gaussian_pair` and `streamfunction_quad` cap everything.** Both controllers
  fail on them (14-30%), so no controller in this family of approaches can
  exceed roughly 75-80% overall.
- **E4's fine-tuning was stopped early**, at 1.5M of a planned 12M steps,
  because training success was drifting down (74.2% -> 67.9%) while
  time-in-tolerance rose. Its 1.5M checkpoint is the best result here; whether
  a longer or better-scheduled fine-tune does better is untested.

## Two follow-ups that did NOT pay off

Both were cheap, both were run, both came back negative. Recorded because a
negative result that took 30 minutes is worth more than a hunch.

**Recovering the easy families by weight interpolation (`soup.py`).** E5 gives
back 4-5 points on `quadratic` and `cubic_perturbed` relative to the clone it
started from. Since fine-tuned weights sit a short optimisation path from the
clone, the WiSE-FT trick of averaging the two often restores the pretrained
behaviour while keeping the fine-tuned gains. Here it does not: success is
**monotone in the interpolation coefficient**, 65.0% at the clone rising to
74.5% at the fine-tune with no interior optimum. The fine-tuning is not
dragging the policy off a better solution, it is simply better.

    alpha   0.00   0.30   0.50   0.70   0.85   1.00
    succ   65.0%  64.5%  67.0%  68.0%  70.0%  74.5%

(A first version of this experiment was wrong and is worth the warning: it held
the observation normalizer fixed at the fine-tuned run's statistics, which
scored the cloned weights at 28.0% against the 66.4% they earn under their own.
Every low-alpha point was measuring a normalizer mismatch. The normalizer has to
be interpolated alongside the weights.)

**A fine-tuning hyperparameter sweep (`sweep_finetune.py`).** Five cells from an
identical clone, spanning an order of magnitude in learning rate plus epoch
count and entropy coefficient, 600k steps each with checkpoints every 200k.
The winner was `lr = 1.2e-4, epochs = 10` — **the configuration E5 already
used**. Lower learning rates did not recover the easy families as hypothesised;
they simply learned less (`lr = 5e-5` had the worst hard-family score, 24-26%
against 32%). So there is no free hyperparameter gain here, and the sweep's real
value is that it removes the selection-asymmetry criticism: PPO's configuration
has now been swept too, and the setting already in use won it.

**One pattern worth carrying forward.** Every small validation screen in this
project has been optimistic, consistently:

| selected on | screen | held-out | gap |
|---|---|---|---|
| 40 fields (E3) | 77.5% | 63.0% | -14.5 |
| 200 fields (E5) | 74.5% | 70.1% | -4.4 |
| 100 fields (sweep winner) | 70.0% | 66.8% | -3.2 |

Selecting the maximum over several checkpoints on a noisy estimate biases that
estimate upward. The sweep winner illustrates it cleanly: screened at 70.0%, it
delivers 66.8% on 1000 held-out fields, a tie with the analytic law
(+0.3%, CI [-2.5%, +3.1%]), and does not displace E5. Trust the large held-out
number, never the screen.

## What I would try next, in priority order

1. **Put a fifth robot at the centroid.** This is the principled fix, not a
   tuning idea. The whole ceiling traces back to one fact: four ring readings
   determine a traceless Hessian, so `det(H_est) < 0` always and every critical
   point looks like a saddle. A center reading gives the trace directly,
   `z_ring_mean - z_center ~ (R^2/4) tr(H)`, which is exactly the missing bit
   that separates a saddle from an extremum. It should convert most of the
   `gaussian_pair` / `streamfunction_quad` failures, which is where the
   remaining 25% of the ceiling lives, and it makes the online handover rule
   work, which is worth the +10% oracle margin. It also connects directly to
   the existing 6-robot pentagon work in this repo.
2. **Give the policy 2-3 stacked frames.** The original rotating-Hessian idea
   was to accumulate information over time; a memoryless policy cannot. Frame
   stacking is the cheap version of that and needs no extra hardware.
3. **Fix the easy-family regression.** E4 loses to the analytic law only on
   `quadratic` (88.7% vs 98.5%) and `cubic_perturbed` (93.7% vs 100%), the two
   easiest families. That is recoverable and worth ~2 points, which is roughly
   what a significant win needs.
4. **A PPO hyperparameter sweep**, to remove the selection asymmetry noted
   above. The analytic law got 48 cells; PPO got hand-picked configs.
5. **A curvature-aware auxiliary loss**, predicting `sign(tr H)` from the
   observation history as a side task. Cheap, and it would show directly
   whether the information is recoverable over time even without a fifth robot.

## A note on numbers

Per the repo convention in `CLAUDE.md`, nothing here writes into a `.tex` file.
`evaluate.py` emits tables to `outputs/` for review first.

