# How this got to the right answer, including the wrong turns

This is the working record: superseded results, a retracted conclusion, and the
defects that produced them. It is kept separate from `README.md` so the README
can state the finding without burying it, and kept at all because the wrong
turns are the most transferable part of the work.

The short version: everything measured before the environment rebuild was
distorted by four defects, and the conclusion drawn from it ("the reward is not
the cause") was **wrong**. `README.md` has the corrected result.

---

## RETRACTED: everything in this section was measured on a broken environment

> **Read this first.** The results below are left in place for provenance, but
> the environment they were measured on had four defects, and the central
> conclusion drawn from them was **wrong**. Specifically, the claim further
> down that testing `reward_mode='pure'` "rules out the reward's extra
> structure as the primary cause" does not hold: under the termination rule in
> force at the time, driving out of bounds was *strictly optimal* above
> e ~ 1.04, worth up to +200 return, and the entire start annulus (e in
> [1.0, 2.5]) sat above that threshold. Both reward modes shared that
> termination, so their agreeing failures were predicted by the bug and say
> nothing about reward shape. The comparison was never a controlled experiment.
>
> The four defects, all confirmed against saved run data:
>
> 1. **Domain exit cost -10** against a per-episode return range of ~120, and
>    the boundary was defined in `c - saddle`, the one coordinate the
>    observation deliberately hides. The median PPO episode ended by driving
>    into the wall with `e_min == e_0`: zero progress, straight out from step
>    one. ~41-47% of episodes exited early against ~25% for the baseline.
> 2. **Under `pure`, exiting was strictly optimal** (above).
> 3. **The potential shaping paid an idle agent** `K(1-gamma)e = +0.02e` per
>    step, i.e. +0.05/step for parking at e=3 and doing nothing, and paid more
>    the further out it sat. The do-nothing controller scored +21.9 on a field
>    it never moved in.
> 4. **The formation-size penalty did not clip R** while the line above it did.
>    Live training episodes reached R ~ 1.9 m and scored -1.71/step against a
>    -0.33/step floor, so the size term alone contributed about -1030 of a
>    -1028 episode return and drowned every other signal.
>
> Re-scored on the fixed environment the tuned baseline improves to **68.3%**,
> so the gap was *wider* than reported here, not narrower. See "Rebuilt
> environment" below for the corrected results.

## What was measured on the broken environment (3M steps, 200 held-out fields)

Numbers from `outputs/evaluation.json`. Superseded, see the retraction above.

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

~~That rules out the reward's extra structure as the primary cause.~~
**This inference was wrong; see the retraction at the top of this section.**
Both reward modes shared a termination rule that made leaving the domain
optimal, so their agreeing failures were caused by the termination, not by
anything about reward shape. The follow-up claim that the observation cannot
distinguish an extremum from a saddle was also not established: `(H, g)`
reconstructs from the 24-dim observation to 4e-15, so the observation is
information-sufficient and the difficulty is representational (the Newton step
carries a pole, `1/m -> infinity` at the 45-degree blind spot), not
informational.

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
law gets 1-2% success under real dynamics.

**All of the above was superseded once the environment was rebuilt.** The
conclusion drawn here, that PPO was reliably choosing the wrong critical point
and that neither reward nor training procedure could be blamed, was an artifact
of measuring on a broken environment. See below.

---

