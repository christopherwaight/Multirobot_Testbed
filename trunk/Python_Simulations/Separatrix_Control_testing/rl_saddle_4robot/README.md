# PPO for 4-robot saddle seeking under real actuator dynamics

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

## What actually happened (3M steps, 200 held-out fields)

Numbers from `outputs/evaluation.json`, produced by the commands above. This
is a real run, not a projection, and it did not go the way the pitch
suggested. Recorded here rather than smoothed over, per the repo's
no-numbers-without-provenance convention.

| controller | success | time in tol | e_final median [p25, p75] |
|---|---|---|---|
| do-nothing | 0% | 0% | 1.80 [1.36, 2.13] |
| rot-Hessian single, default gains | 1% | 0.2% | 1.20 [0.65, 2.42] |
| rot-Hessian none, default gains | 25% | 15% | 0.22 [0.15, 3.03] |
| **rot-Hessian single, best swept gains** | **58%** | **38%** | **0.033 [0.017, 3.03]** |
| rot-Hessian single, 0-dynamics ceiling | 30% | 5% | 0.26 [0.16, 3.00] |
| **PPO (3M steps, curriculum)** | **38%** | **26%** | **3.01 [0.040, 3.28]** |

Two things, read together:

**The properly tuned kinematic baseline is the strongest controller here**,
by a wide margin on the median. The gain sweep mattered exactly as much as
flagged above: `k_trans = 0.3626` (the repo default) gets 1 percent success;
`k_trans = 1.5` at a smaller `r0 = 0.10` gets 58 percent. Retuning three
numbers did more than switching to RL did.

**PPO's result is real but bimodal, and the rollouts explain why.** The p25
(0.040) shows it nails many episodes as well as the tuned baseline. The
median (3.01) shows it badly misses on most. `outputs/figures/fig5_rollouts.png`
makes the failure concrete: on 4 of 6 example episodes the formation walks
confidently to a nearby EXTREMUM (a local max or min, visible as a solid
red or blue blob) and parks there, not to the saddle. It gets `double_gyre_psi`
right (0.043 m) and does reasonably on `cubic_perturbed`, but fails on
`quadratic`, `log_sum_exp`, `gaussian_pair`, and `streamfunction_quad`.

The mechanism is legible from the reward, in hindsight: `w_track * exp(-e^2/sigma^2)`
depends only on distance to the true saddle and formation size. It does not
depend on the local curvature sign. A local extremum has zero gradient too,
so nothing in the observation or reward tells the policy that arriving at a
gradient-zero point with a definite (not indefinite) Hessian is the wrong
kind of stationary point. `outputs/figures/fig3_training.png` shows training
converging fast on the `quadratic`-only curriculum stage 1 (success up to
about 0.5) then plateauing near 0.35-0.40 once the other five families enter
in stages 2-3, and never climbing further in the steps used here. That is
consistent with capacity or training budget rather than an environment bug:
the curriculum stages transition cleanly and 5000 steps/s throughput was not
the bottleneck.

**Tested directly: is the shaped reward itself the cause?** The reward already
used the true saddle location (`e = |centroid - true_saddle|` is what the whole
thing is built from; only the *observation* withholds it, by the original
single-frame spec). The live question was whether the extra structure on top
of that, the multiplicative size gate, the potential-based shaping, the step
cost, was what steered the policy onto extrema instead of noise alone. So a
second policy was trained for the same 3M steps, same curriculum, same seeds,
with `reward_mode='pure'`: reward is exactly `-e` every step, nothing else
(`quad_saddle_env.py`, `--reward-mode pure`).

| controller | success | e_final median | quadratic | log_sum_exp | gaussian_pair | cubic_perturbed | streamfunction_quad | double_gyre_psi |
|---|---|---|---|---|---|---|---|---|
| PPO, shaped reward | 38% | 3.01 | 3.02 | 3.09 | 3.05 | 1.17 | 3.46 | 0.041 |
| PPO, pure `-e` reward | 24% | 3.04 | 3.03 | 3.11 | 3.21 | 0.029 | 3.10 | 0.024 |

Simplifying the reward to literally nothing but true-saddle distance did **not**
fix the failure. The same four families fail (`quadratic`, `log_sum_exp`,
`gaussian_pair`, `streamfunction_quad`), by nearly the same amount, and overall
success went down, not up (38% to 24%). `outputs/figures/fig4_comparison.png`
panel b shows the two bars sitting almost on top of each other across every
family. Pure reward did clearly help on `cubic_perturbed` (1.17 to 0.029) and
matched shaped on `double_gyre_psi`, so the shaping terms are not harmless
either, just not the dominant effect.

That rules out the reward's extra structure as the primary cause. What's left,
consistent with the rollout figure showing the formation walking to and
parking on extrema: the OBSERVATION cannot tell an extremum from a saddle
(both are gradient-zero to the estimator), and no amount of reshaping the
scalar reward changes what the policy can perceive. A reward can only steer
among behaviors the policy can already tell apart from its inputs.

## Tested next: does more exploration, no curriculum, and a longer budget fix it?

Same question the reward ablation asked, aimed at training procedure instead
of the reward. Two more full runs, `--tag ppo_v2` and `--tag ppo_v3`, both
15M steps (5x the original budget), both with the curriculum removed (full
8-family mix from step 0, since stage 1's `quadratic`-only warmup has no other
critical point to teach the policy to avoid), both with gSDE instead of i.i.d.
per-step exploration noise (a persistent, multi-step excursion is what
escaping a false attractor needs, not per-step jitter), entropy coefficient
raised 0.003 to 0.02, and the network widened 64 to 128.

**v2 found a real failure mode of its own: it collapsed.** Bucketing the
15M-step training curve into deciles:

| steps (millions) | success | time in tol |
|---|---|---|
| 1.5 - 9.0 | 37-43% | 18-24% |
| 9.0 - 10.5 | 37% | 9% |
| 10.5 - 15.0 | 12-22% | 0.7-1.6% |

It peaked in the middle of the run at roughly the same level as the original
curriculum run, then degraded hard in the last third. No periodic checkpoints
were saved for v2, an omission on my part, so the actual best policy from the
run was unrecoverable, only the collapsed final one. Fixed for v3: `train_ppo.py`
now checkpoints every million steps by default, anneals the learning rate
linearly (the standard stabilizer for exactly this kind of late-training PPO
collapse), and `--select-best` quick-screens every checkpoint afterward and
keeps the winner automatically.

**v3, same setup plus the fix, did not collapse** (success held 25-42% across
all 15 checkpoints) but did not clearly improve on v1 either:

| controller | success | e_final median | quadratic | log_sum_exp | gaussian_pair | cubic_perturbed | streamfunction_quad | double_gyre_psi | quartic_wells | rational_envelope |
|---|---|---|---|---|---|---|---|---|---|---|
| rot-Hessian, best gains | 65% | 0.027 | 0.020 | 0.025 | 3.057 | 0.022 | 3.165 | 0.016 | 0.027 | 0.031 |
| PPO v1, shaped, curriculum | 38% | 3.01 | 3.02 | 3.09 | 3.05 | 1.17 | 3.46 | 0.041 | - | - |
| PPO v1, pure `-e` reward | 24% | 3.04 | 3.03 | 3.11 | 3.21 | 0.029 | 3.10 | 0.024 | - | - |
| PPO v3, no curriculum, gSDE, 15M | 34.5% | 3.00 | **0.17** | 3.11 | 3.02 | 2.81 | 3.06 | **0.08** | 3.09 | 3.01 |

(v1's two runs predate the two newest field families, hence the dashes.)

**The result that survived three independent training regimes**, different
reward, different curriculum/exploration/network/budget: PPO reliably solves
exactly the two families whose accessible domain contains no critical point
besides the saddle, `quadratic` and `double_gyre_psi`, and sits at the
domain-exit ceiling (~3.0) on every family that has another one, `log_sum_exp`,
`gaussian_pair`, `streamfunction_quad`, and now also the two families added
specifically to test this, `quartic_wells` and `rational_envelope`, both of
which the tuned analytic law solves cleanly (0.027, 0.031) while every PPO
variant fails on both (3.09, 3.01). `cubic_perturbed` is the one family that
does not fit the pattern cleanly, it has no extra critical point by
construction, and PPO's result on it swings 1.17 / 0.029 / 2.81 across the
three runs, which reads as run-to-run variance rather than a real capability
either way. `outputs/figures/fig4_comparison.png` panel b shows this
side by side across all three PPO runs and eight families.

One more thing worth naming from v3's numbers: `|omega|` averaged 1.84 rad/s
against a 2.0 cap with 86% saturation, far higher and far more saturated than
v1's 1.05 / 22%. gSDE's persistent exploration noise seems to have pushed the
policy toward a near-constant high-rate spin rather than a more differentiated
strategy, another way of restating that more exploration did not translate
into better discrimination between critical-point types.

**Where this leaves it.** Three different levers, reward shape, and now
training procedure (curriculum, exploration, network size, budget,
stabilization), were each tested directly rather than assumed, and none of
them closed the gap to the tuned classical law. The failure tracks field
topology (does another critical point exist in reach) far better than it
tracks any training choice, which points at the 24-dimensional observation
itself as the limit: it gives the policy a gradient and a formation-frame
curvature scalar, and that is genuinely not enough to tell a saddle from a
nearby extremum, no matter how it is trained. The one lever that has not been
tested is giving the policy that missing information directly:
`--obs-mode raw+est` appends the hand estimator's own `(H, g)` to the
observation. If PPO with that extra input still walks to extrema, the limit
is something else entirely (likely the credit-assignment problem itself, escaping
a flat-gradient trap over a 600-step horizon); if it fixes the failure, the
raw 24-dim observation was the bottleneck all along, which would be the
cleanest possible resolution of everything measured above.

None of this is in the pitch's original failure mode (dynamics saturation
defeating a hand law). That failure mode is real and documented above and in
`estimator.py` and `fig1`/`fig6`, it is exactly why the untuned rotating-Hessian
law gets 1-2% success under real dynamics. It is just not what limited PPO:
PPO's problem was never reaching the saddle once close to the right field
region, it was reliably choosing the wrong critical point to approach in the
first place, and that held up under a reward, and then a training procedure,
with no room left to blame either one.

## A note on numbers

Per the repo convention in `CLAUDE.md`, nothing here writes into a `.tex` file.
`evaluate.py` emits tables to `outputs/` for review first.
