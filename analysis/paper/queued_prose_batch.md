# Queued prose and math batch, 2026-11-16

**Applied 2026-11-16 by `apply_prose_batch.py`. Item F (now 5%) and the sweep sentence of item M (median 1.2 mm) were applied after the v4 sweep finished at 23:51.** Changes made while applying: the test plan states measured formation sizes (2025 vortex triangle 0.26 m, saddle 0.30 m; 4R diagonals 0.33 m in 2025 and 0.31 m in 2026). Items I and P sit together in Sec. V-D, the limit-cycle figure is Fig. 8 (the V2S figure is now Fig. 9), and the V-D practical-range sentence was merged to "in both fields" like the abstract.

Not applied. Every item below changes a claim, an equation, or wording about the robot model, so it waits for your yes.
Numbers that kept their sentence true are already in the tex as `\rev{}` (see `numbers_map.csv`).
The four items marked **[pending]** wait on the v4 sweep, which should finish around 23:45 on 2026-11-16.

Symbols: I kept `v_c` and the phrase "friction ring" throughout. The dead zone models static friction, so the ring is
still a friction ring. Keeping them limits the ripple to the four sentences that say "Coulomb".

---

## A. Robot model (Sec. III-A, from "Details of the hardware platform" to "not the robot dynamics.")

Current: a first-order lag with Coulomb friction, tau = 0.28 s, alpha = 0.70, v_max = 0.4 m/s.

Proposed:

> Details of the hardware platform are described in Section~V. In simulation, each robot is modeled as a single
> omnidirectional point mass that follows its command after a transport delay, through a first order response, with a
> dead zone and a speed limit. A command reaches the robot a time $L$ after it is computed, and the robot attains the
> fraction $g$ of it, so its velocity $\mathbf{v}_{r,i} = \dot{\mathbf{p}}_i$ follows
> \begin{equation}
>     \tau\,\dot{\mathbf{v}}_{r,i}(t) = -\mathbf{v}_{r,i}(t) + g\,\mathbf{u}_i(t - L),
>     \label{eq:continuous}
> \end{equation}
> where $\tau$ is the actuator time constant \cite{4,7}. The command passes through a dead zone that models static
> friction: $\mathbf{u}_i = \mathbf{v}_{\text{des},i}$ when $\|\mathbf{v}_{\text{des},i}\| \ge v_c$, and
> $\mathbf{u}_i = \mathbf{0}$ otherwise. Fitting (\ref{eq:continuous}) to the logged commands and tracked positions of
> the hardware runs of Section~V gave $g = 0.86$ and fixed the sum $L + \tau = 0.075$~s more tightly than either term.
> The robots reach their commanded speed within one cycle, so we take $\tau = 0.02$~s and $L = 0.055$~s. The robots are
> also modeled with a maximum speed of $v_{\max} = 0.3$~m/s and a dead zone of $v_c = 0.025$~m/s (remeasured for this
> work; an earlier characterization of the testbed reported 0.05~m/s \cite{22}).
>
> For digital implementation at control period $\Delta t$ and cycle $j$, each command is held over its period. A delay
> $L = f\Delta t$ with $0 \le f < 1$ means the robot spends the first fraction $f$ of each period on the previous
> command, so
> \begin{equation}\label{eq:momentum}
> \begin{split}
> &\bar{\mathbf{u}}_i[j] = (1-f)\,\mathbf{u}_i[j] + f\,\mathbf{u}_i[j-1],\\
> &\tilde{\mathbf{v}} = \alpha\,\mathbf{v}_{r,i}[j] + (1-\alpha)\,g\,\bar{\mathbf{u}}_i[j],\\
> &\mathbf{v}_{r,i}[j+1] = \tilde{\mathbf{v}}\,\min\{1,\ v_{\max}/\|\tilde{\mathbf{v}}\|\},
> \end{split}
> \end{equation}
> where $\alpha = e^{-\Delta t/\tau}$ and the dead zone acts on $\bar{\mathbf{u}}_i$. At a 10~Hz control rate,
> $f = 0.55$ and $\alpha \approx 0.007$, so the robot reaches $g$ times its command within one cycle, and the delay, not
> the lag, shapes its response. The position advances as $\mathbf{p}_i[j+1] = \mathbf{p}_i[j] + \Delta t\,\mathbf{v}_{r,i}[j+1]$.
> The sensed field enters only the estimate, not the robot dynamics.

Notes:
- The code (`omnibot.py`) applies the dead zone to the delayed, blended command. The text says so.
- The 0.28 s time constant came from an earlier characterization. The new fit says the robots are much faster. R1 or R3 may
  ask why it changed. The answer is that the 2026 runs log commands and positions at every cycle, so the fit can separate the delay from the lag.
- The `0.3~m/s` cap replaces 0.4. The hardware controller also clips the centroid command to 0.3 m/s per axis. That
  clip is already in Sec. IV as `\rev{0.3}`.

## B. Sampled loop (Sec. III-D, "Sampled Loop", from "With the formation held" to the pole sentence)

Pseudocode of the check (theory_check.py), for your reading only:

```
per axis, alpha ~ 0, L = f dt (d = 0):
  e[j+1] = e[j] - g k dt ((1-f)(e[j] - delta[j]) + f (e[j-1] - delta[j-1]))
  characteristic polynomial  z^2 - (1 - a(1-f)) z + a f,   a = g k dt
  Jury: a f < 1  and  a (1 - 2f) < 2  (and a > 0)
  f >= 1/4  =>  first condition binds  =>  k < 1/(g L)
  check: eigenvalues of the full model with lag: 21.14 /s; simulation stable at k = 21, unstable at 22
```

Proposed:

> With the formation held, and apart from the dead zone and the cap, the centroid follows (\ref{eq:momentum}) with
> $\alpha \approx 0$. The $x$ and $y$ axes then decouple, and for each axis $a$
> \begin{equation}\label{eq:cl}
> e_a[j{+}1] = e_a[j] - gk\Delta t\bigl((1-f)(e_a[j] - \delta_a[j]) + f(e_a[j-1] - \delta_a[j-1])\bigr).
> \end{equation}
> The Jury test on its characteristic polynomial $z^2 - (1 - gk\Delta t(1-f))z + gk\Delta t f$ gives exponential
> stability if and only if $gk\Delta t f < 1$ and $gk\Delta t(1-2f) < 2$, with $k > 0$. For $f \ge 1/4$ the first
> condition binds, and since $f\Delta t = L$,
> \begin{equation}\label{eq:kmax}
> 0 < k < 1/(gL).
> \end{equation}
> At $g = 0.86$ and $L = 0.055$~s the limit is 21.1~s$^{-1}$, and at $k = 1$~s$^{-1}$ the poles have magnitudes 0.91
> and 0.05. Including the lag changes the limit by less than 0.01~s$^{-1}$.
> A stable linear loop is input-to-state stable, so a bounded estimate error $\boldsymbol{\delta}$, and the dead zone
> acting as a command disturbance of size at most $v_c$, leave the centroid error bounded. [cap sentence unchanged]

Ripple: the intro (line ~70) says "despite actuator lag, sampled updates, and friction". Suggest "despite actuator delay
and lag, sampled updates, and friction". The section opener says "through the sampled lag and Coulomb friction". Suggest
"through the delay, the lag, and the dead zone". The delay cuts the old limit of 113 s^-1 by a factor of five. That is
worth one line in the R1 response on stability margins.

## C. Friction and estimate error (Sec. III-D), first two sentences

> The second effect is friction, and it sets where the cluster finally comes to rest. The dead zone stops each robot
> whenever its command is below $v_c$.

The rest of the paragraph stands. The ring is still $v_c/k$ = 0.025 m, and the analytical-field mean final error is
0.0226 m, inside it.

## D. Orbit closed form (Sec. III-D, "Orbits", second paragraph)

Proposed:

> Even with an exact estimate, the delay and lag leave a predictable offset in the steady orbit radius. The robots reach
> the fraction $g$ of each command, so the orbit speed is $V = gk_t$ and the radial correction has gain $gk_r$. The
> cluster follows the turning command a time $T$ late, so in a steady orbit of radius $r^*$ it drifts outward at
> $TV^2/r^*$. The orbit settles where the inward radial command balances this drift, giving
> \begin{equation}\label{eq:orbit_cf}
> r^* = \tfrac{1}{2}\bigl(r_d + \sqrt{r_d^2 + 4C}\bigr), \qquad C = TV^2/(gk_r).
> \end{equation}
> In the sampled loop $T \approx L + \tau + \Delta t/2$: the delay, the lag, and half a period of command hold.
> Equation (\ref{eq:orbit_cf}) accounts for two behaviors seen in Sections~IV and~V: the error $r^* - r_d$ cannot shrink
> below the floor $r_{\min} = \sqrt{C}$ as $r_d \to 0$, while at large radii it decays as $C/r_d$.

Check (theory_check.py, linear vortex, 60 s, mean radius after 20 s):

| r_d (m) | sim r* | closed form, T = L + tau + dt/2 | T = L + tau |
|---|---|---|---|
| 0.00 | 0.0469 | 0.0492 | 0.0381 |
| 0.01 | 0.0518 | 0.0544 | 0.0434 |
| 0.05 | 0.0772 | 0.0802 | 0.0706 |
| 0.10 | 0.1176 | 0.1201 | 0.1129 |
| 0.30 | 0.3067 | 0.3079 | 0.3048 |
| 0.50 | 0.5041 | 0.5048 | 0.5029 |

The largest gap is 3.0 mm. A fitted single T = 0.111 s brings it to 0.6 mm.

## E. Sec. IV-A, line "while at least \rev{95}\% of trials still end within 0.05~m."

> ... lowers success at 0.03~m, while every trial still ends within 0.05~m.

Table II: reconstructed vortex 91/100/100, saddle 68/100/100 at 3/4/5 cm.

## F. Sec. IV-A, covariance prediction "within \rev{4}\%" [pending]

The covariance check (`theory.noise_rms`, a discrete Lyapunov solve) is written for the old lag-only loop. It needs the
delay state added, then the E6 noise cells of the v4 sweep to compare against. I can do both after the sweep. If you
prefer, cut the sentence. The noise result in the sentence before it (3.0 mm) already makes the point.

## G. Sec. IV-B, orbit trend and practical range (last two sentences of the first paragraph)

> Mean radial errors had a positive bias, and decreased with increasing radius. The reconstructed fields share the bias
> at $r_d = 0.01$~m and nearly cancel it at 0.10~m. From 0.20~m their bias lies between 0.019 and 0.050~m, with
> within-run standard deviations that increase with the commanded radius. By the $0.25\,r_d$ criterion the practical
> range is $r_d \ge 0.10$~m on the analytical and reconstructed fields.

Data (summary.json, bias / within-run sd, m): reconstructed vortex 0.039/0.006, 0.006/0.009, 0.038/0.016, 0.048/0.021,
0.046/0.035, 0.027/0.046; reconstructed saddle 0.034/0.009, 0.000/0.016, 0.045/0.021, 0.050/0.035, 0.043/0.037,
0.019/0.041. The nonlinear fields also pass from 0.10 m wherever their orbits were kept.

## H. Sec. V-B, test plan

1. Orbit runs. Current: "allowing the cluster to complete several orbits around the critical point". This is not
   true at large radii, where the 3R runs complete one full lap from r_d = 0.30 m. Proposed:

   > Each orbiting controller experiment was run for 200 to 300 cycles at 10~Hz, long enough for at least one full orbit
   > at the largest radius and up to eight at the smallest.

2. Four robots (new sentence, end of the paragraph). **Check the Oct 2025 four-robot formation size before use.**

   > The vortex attraction trials and both fields' orbit trials were repeated with four robots in a square formation,
   > with $\hat{\mathbf{J}}$ from a least squares fit to the four readings; the four-robot orbits used five radii, from
   > 0.01 to 0.40~m.

3. Extras (new sentence after "yielding 80 trials per field").

   > Two saddle trials missing from the original set were rerun from the same starting locations, and their positions
   > were mapped into the original frame of the printed map.

## I. Sec. V-C, classification sentence (`\open{campaign}` still in the tex)

> The classifier labeled $\hat{\mathbf{J}}$ on every cycle of the orbit runs. It matched the printed saddle on 100\% of
> cycles. In the vortex, 99.9\% of cycles were labeled rotational, but only 45\% as a center, because the center's
> eigenvalues lie on the boundary between the two spiral types (Section~IV-C), so a small real part from distortion or
> noise moves the label to a spiral.

Data: hardware_summary.json, classification_2026_orbits: vortex 3R strict 44.7%, rotational 99.9%; vortex 4R 46.5% and
99.6%; saddle 3R 99.95%; saddle 4R 99.6%. The 2025 attraction files do not log J-hat, so these labels come from the 2026
orbit runs. The sentence may sit better in V-D.

## J. Sec. V-D, lost orbits (three `\syn{}` still in the tex)

> No orbit run left the map.

## K. Abstract

1. Practical range: "at radii of \rev{0.20}~m (vortex) and \rev{0.20}~m (saddle) and above" becomes "at radii of 0.20~m
   and above in both fields". The same change applies in the conclusion, which repeats the pair.
2. The abstract now counts the 3R orbits only (120), which matches its 3R attraction count (160). The 4R vortex orbits fail
   the quarter-radius test at 0.30 m (bias 0.098 m), so adding them would break the "and above" claim. A 4R sentence
   could read:

   > A four-robot formation converged within 0.05~m in all 79 vortex trials.

## L. Sec. VI-A, residual error (the "friction ring" paragraph)

No change is needed. "Friction ring" and "friction level $v_c$" still hold under the dead zone.

## M. Sec. VI-A, orbit floor paragraph [pending for the sweep sentence]

> Positive radial bias at small orbital radii is also a result of robot dynamics. At small radii, the commanded angular
> rate $k_t / r_d$ is high relative to the radial correction bandwidth $gk_r$, and the outward drift from the delay and
> lag sets the bias (\ref{eq:orbit_cf}). The closed form matches simulation within 3~mm at every commanded radius at the
> settings of Section~IV [sweep sentence: median gap across the E2 sweeps of gains, delay, lag, sample period and dead
> zone, after v4]. There $r_{\min} = 0.047$~m, so the error stays below 0.05~m at every commanded radius. Lowering the
> orbital speed $k_t$ to 0.1~m/s cuts $r_{\min}$ to 0.031~m, and removing the delay and lag cuts it to 0.032~m. Raising
> the radial gain $k_r$ to 2~s$^{-1}$ cuts it to 0.034~m, at the cost of increased sensitivity to noise in the critical
> point estimate.

Source: extra_numbers.json (rigidsim, linear vortex, r_d = 0, mean radius after 20 s of 60 s).

## N. Sec. VI-A, reconstructed-field orbit bias at r_d = 0.5 m (claim reversed)

> The measurement term of (\ref{eq:bound}) grows with the extrapolation distance $\|\mathbf{e}\|/R$, and the
> reconstructed field degrades near the edge of the mapped region. Together they set the localization error at the
> largest commanded radius. There the orbital bias on both reconstructed fields exceeds the controller-only bias,
> +0.027 against +0.013~m on the vortex and +0.019 against $-$0.003~m on the saddle, so the estimate lies closer to the
> cluster than the critical point does, and a cluster holding its estimated radius orbits outside $r_d$.

The old text had the opposite sign. With the 2025 nets, the estimate was inflated and pulled the cluster inward. The 2026 nets reverse it.

## O. Sec. VI-B, V2S comparison (three claims changed)

1. "every orbit spiraled off the map". No V2S-alone orbit left the map on the linear fields. Proposed:

   > In orbit, the primitive alone cannot correct radial error about a specific point, so centrifugal drift went
   > uncorrected and every orbit drifted outward, with a mean radial error of 0.14~m at $r_d = 0.3$~m
   > (Fig.~\ref{fig:effort}(b)).

2. "held the radius at a lower RMSE". The RMSE is now higher than the Jacobian's (0.075 against 0.018 m at
   r_d = 0.1 m). Proposed:

   > With its own speed estimate the primitive orbited inside the commanded radius, by 0.075~m at $r_d = 0.1$~m,
   > because the estimate's inward bias outweighs the lag drift of Section~III-D; during attraction, that estimate left
   > the cluster still moving at \rev{0.21}~m/s after 10~s.

3. Anisotropic node, "even with the oracle most orbits were lost". The oracle lost 88% at r_d = 0.1 m, 46% at 0.3 m and 30% at 0.5 m.

   > ... and even with the oracle 88\% of orbits at $r_d = 0.1$~m and 30\% at 0.5~m were lost, because the speed
   > contours are ellipses; ...

4. Gyre, "and with the oracle every orbit from $r_d = 0.3$~m". The oracle lost 92% at 0.3 m and all at 0.5 m.

   > The primitive lost every gyre orbit, and with the oracle 92\% at $r_d = 0.3$~m and all at 0.5~m.

## P. Proposed new figure: limit cycle (your decision)

`analysis/limit_cycle/figures/fig_limit_cycle.pdf`. It shows the vortex at r_d = 0.10 m, hardware and the reconstructed vortex in simulation.

> At $r_d = 0.10$~m in the vortex the cluster does not hold the commanded circle but settles onto a closed curve of its
> own (Fig.~X). Ten hardware runs, four started inside the curve and six outside, converged to the same curve, with a
> mean radius of 0.127~m, a peak-to-peak variation of 0.043~m around $\mathbf{p}^*$, and 2.9~mm between runs. On the
> reconstructed vortex in simulation, starts inside and outside meet on a curve of mean radius 0.111~m.

Shape agreement between the two curves is modest (correlation 0.38, median gap 13 mm), so the paragraph claims the
cycle, not a match of its shape.

## R. Response letter (Response_to_Reviewers.tex), claims that changed

Numbers in the letter are updated and marked `\rev{}`. `check_letter_quotes.py` reports 46 quotes and 0 mismatches. Five
items wait for you:

1. Opening, `\syn{All hardware experiments were repeated in a new campaign ...}`. Only the orbit experiments are
   new (2026). The attraction results are the 2025 runs (3R vortex and saddle, 4R vortex), re-scored. Proposed:

   > The hardware orbit experiments were repeated in a new campaign designed around the reviewers' requests, with ten
   > repeats per radius and with both three and four robots, and every hardware result is re-scored with the metrics
   > of Section~IV.

2. Opening, `\syn{... the new campaign uses the three-robot formation throughout.}` The revision now reports both. Proposed
   ending: "...; the revision reports three-robot and four-robot results in separate columns of Table~III."
3. R1.3(d), "to within `\rev{4}`\%". This waits on item F.
4. R1.7, `Hardware accuracy is in Section~V-C \open{campaign}.` This waits on item I. Once I is in, the letter can read: "Hardware
   accuracy is in Section~V-C: 100\% of cycles in the saddle, and in the vortex 99.9\% rotational, of which 45\% are
   labeled a center."
5. R1.4(e), new sentence on the trim (your request to say the earlier 10 s was too generous):

   > A settling analysis of the new orbit runs found no start-up transient on hardware, since each run starts near its
   > circle, and in simulation the radial error falls to 2~mm within 3~s. The submitted 10~s exclusion was therefore
   > more generous than needed, and the revision uses 3~s for simulation and hardware alike; the mean radial bias
   > changes by at most 4~mm between exclusions of 0, 2, 5 and 10~s.

   Source: analysis/hardware/outputs/transient_choice.md and settling_2026.csv.
