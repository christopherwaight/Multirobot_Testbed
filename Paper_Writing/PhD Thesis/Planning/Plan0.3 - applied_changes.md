# Plan 0.3: changes applied to the thesis

Date: 2026-11-19. Every edit made while working through Plan0, Plan0.1 and Plan0.2.
Edits go through `sources/assemble/*.py` and `thesis_main.tex`; `chapters/*.tex` are regenerated.
Each entry: item id, file, why, then old and new text. `git diff` of the scripts shows the same changes.

## E1, M4: `sources/assemble/ch01.py`
Novelty claim names this dissertation (Plan0.1 1a). Contributions: 169 to 459 hardware runs, 239 attraction trials plus 220 orbit runs (checked against `analysis/hardware/outputs/attraction_per_trial.csv`, 239 rows, and `orbit_per_run.csv`, 220 rows, and the revision paper's 459); 'distributed' to 'cooperative estimation' (Plan0 Sec. 1.5).

Old:
```
P('sys', 'Work that senses the physical field falls into two camps.'),
```
New:
```
P('sys', 'Work that senses the physical field falls into two camps.',
  subs=[('this paper gives', 'this dissertation gives')]),
```

Old:
```
(Chapter~\ref{ch:zeroth}). Third, it presents a distributed algorithm that
```
New:
```
(Chapter~\ref{ch:zeroth}). Third, it presents a cooperative estimation algorithm that
```

Old:
```
and orbital control laws built on that estimate in 169 hardware
experiments (Chapter~\ref{ch:first_order}).
```
New:
```
and orbital control laws built on that estimate in 459 hardware runs
with three and four robots (Chapter~\ref{ch:first_order}).
```

## E1, E6, E9, 1f: `sources/assemble/ch04.py`
Fig. 4.1 caption: 'new in this paper' to 'the contributions of this dissertation' (Plan0 Required). Table 4.1 column renamed (Plan0 Table 4.1). 'Dr. Chris Kitts' to 'Kitts and colleagues' (Plan0.1 1f, your veto). New paragraph states Cov(theta) with its assumptions (Plan0 optional, wording from Draft 14 Sec. II-C).

Old:
```
('width=0.98\\columnwidth', 'width=0.55\\textwidth')]),
```
New:
```
('width=0.98\\columnwidth', 'width=0.55\\textwidth'),
        ('are new in this paper.', 'are the contributions of this dissertation.')]),
```

Old:
```
Features reachable & Chapter
```
New:
```
Enabled navigation capabilities & Chapter
```

Old:
```
control is a technique developed by Dr. Chris Kitts of Santa Clara
```
New:
```
control is a technique developed by Kitts and colleagues at Santa Clara
```

Old:
```
chapters order the columns differently, which changes nothing in the fit.
```
New:
```
chapters order the columns differently, which changes nothing in the fit.

Every robot reads the field with its own sensor, so the measurement noise
on one reading is independent of the noise on another. Suppose each
reading carries zero-mean noise of variance $\sigma^2$ and
$\boldsymbol{\Phi}_k$ has full column rank. The least-squares coefficients
of one velocity component are then unbiased, with covariance
\begin{equation}
    \operatorname{Cov}(\hat{\boldsymbol{\theta}}) =
    \sigma^{2}\,(\boldsymbol{\Phi}_k^{\top}\boldsymbol{\Phi}_k)^{-1},
    \label{eq:ch4:cov}
\end{equation}
conditional on the robot positions being known.
```

## E4: `sources/assemble/ch05.py`
Drop the opener that repeated the new Sec. 5.1 text (Plan0.1 1h).

Old:
```
To evaluate our platform, we implemented two adaptive navigation
```
New:
```
We implemented two adaptive navigation
```

## E3, E5, E7: `sources/assemble/ch06.py`
Sec. 6.5.2 defines s_e; both result-table captions now say what it is (Plan0). The offline-classification sentence moves ahead of the closing sentence of the chapter summary (Plan0.1 1i). The lag coefficient e^(-dt/tau) is alpha_mom here as in Ch. 2 (Plan0.1 2b.2).

Old:
```
        (r'(\ref{eq:continuous})', r'(\ref{eq:ch2:continuous})')]),
```
New:
```
        (r'(\ref{eq:continuous})', r'(\ref{eq:ch2:continuous})'),
        (r'($\alpha \approx 0$)', r'($\alpha_{\text{mom}} \approx 0$)')]),
```

Old:
```
P('sys', r'\caption{Simulation results on eleven fields.', end=r'\end{table}', back=3),
```
New:
```
P('sys', r'\caption{Simulation results on eleven fields.', end=r'\end{table}', back=3,
  subs=[(r'A dash marks a radius where most orbits were lost.',
         r'A dash marks a radius where most orbits were lost. In (a), $\bar{e}_f \pm s_e$ is the mean and standard deviation of the final error $e_f$.')]),
```

Old:
```
subs=[(r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}')]),
```
New:
```
subs=[(r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}'),
        (r'with matched simulations of each attraction trial.',
         r'with matched simulations of each attraction trial. The final-error row gives the mean $\bar{e}_f$ and the standard deviation $s_e$ of the final error $e_f$.')]),
```

Old:
```
        ('approach uses instantaneous', 'approach used instantaneous')]),
T(r"""
% New text
Offline, the hardware attraction estimates also named the type of both printed
fields in every trial by majority vote, once each map's sensor frame was
registered to the motion-capture frame.
"""),
]
```
New:
```
        ('approach uses instantaneous', 'approach used instantaneous'),
        ('These results establish cooperative Jacobian estimation',
         "Offline, the hardware attraction estimates also named the type of both printed fields in every trial by majority vote, once each map's sensor frame was registered to the motion-capture frame. These results establish cooperative Jacobian estimation")]),
]
```

## E7: `sources/assemble/ch02.py`
One name for the lag coefficient e^(-dt/tau), so that alpha means only the real part of an eigenvalue (Plan0.1 2b.2).

Old:
```
        ('Fits to the hardware runs of Section~V', 'Fits to the hardware runs of Chapter~\\ref{ch:first_order}')]),
```
New:
```
        ('Fits to the hardware runs of Section~V', 'Fits to the hardware runs of Chapter~\\ref{ch:first_order}'),
        ('\\tilde{\\mathbf{v}} = \\alpha\\,\\mathbf{v}_{r,i}[j] + (1-\\alpha)\\,g', '\\tilde{\\mathbf{v}} = \\alpha_{\\text{mom}}\\,\\mathbf{v}_{r,i}[j] + (1-\\alpha_{\\text{mom}})\\,g'),
        ('where $\\alpha = e^{-\\Delta t/\\tau}$', 'where $\\alpha_{\\text{mom}} = e^{-\\Delta t/\\tau}$'),
        ('$\\alpha \\approx 0.007$', '$\\alpha_{\\text{mom}} \\approx 0.007$')]),
```

## E2: `sources/assemble/ch09.py`
Residual IEEE section numbers in the first-order limitations (Plan0 Sec. 9.1.1, Plan0.1 1d).

Old:
```
        ('(Section~\\ref{sec:hw_testbed})', '(Section~\\ref{sec:ch2:testbed})')]),
```
New:
```
        ('(Section~\\ref{sec:hw_testbed})', '(Section~\\ref{sec:ch2:testbed})'),
        ('(Section~II-B)', '(Section~\\ref{sec:ch6:estimation})'),
        ('Section~IV probes', 'Section~\\ref{sec:ch6:sim_attraction} probes')]),
```

## E7: `thesis_main.tex`
Glossary follows the refit model (Sec. 2.6), the only robot model any chapter now uses after the Ch. 7 port to Draft 14. tau was 0.28 s (the earlier momentum model); g, L, v_c added (Plan0.1 2b.1).

Old:
```
$\tau$ & Actuator time constant, 0.28~s on the Decabots \\
$\alpha_{\text{mom}}$ & Momentum coefficient, $e^{-\Delta t/\tau}$ \\
```
New:
```
$g$ & Share of a command that a robot attains, 0.86 (Section~\ref{sec:ch2:dynamics}) \\
$L$ & Transport delay of a robot command, 0.055~s \\
$\tau$ & Actuator time constant, 0.02~s \\
$v_c$ & Dead zone of a robot, 0.025~m/s \\
$\alpha_{\text{mom}}$ & Momentum coefficient of the sampled robot model, $e^{-\Delta t/\tau}$ \\
```

## B: `sources/assemble/apps.py`
Tier 5 B. App. D: title and opening state the three claims; the closing paragraph is rewritten as the three levels of Draft 14 Sec. III-C (features objective, command covariant where the sign test is decisive, trajectory equivariant only empirically), with p-dot' = Q p-dot + Q-dot p and the non-objective tangent seed (Plan0 3C, Sec. 7.7.4, App. D). App. E: scope paragraph (idealized assumptions, the 1/(gL) sampled-loop limit, no claim for the switched saturated controller) and a closing paragraph tying a_perp to Draft 14 Sec. III-D; the undefined reference to the old stop-test equation is replaced by the stop test of Sec. 7.5 (Plan0 App. E, Draft 14 III-D).

Old:
```
\chapter{Frame Equivariance of the $s_1$ Tracker}
```
New:
```
\chapter{Objectivity and Frame Equivariance of the $s_1$ Tracker}
```

Old:
```
and the strain eigenvalues unchanged. This appendix gives the algebra.
```
New:
```
and the strain eigenvalues unchanged. This appendix gives the algebra and
then separates three claims about the $s_1$ tracker that differ in
strength: its features are objective, its command is covariant, and its
trajectory is equivariant only under conditions.
```

Old:
```
The $s_1$ tracker's channels are built from $\hat{s}_1$, its gradient,
and the strain eigenframe, which all co-rotate with the observer. Its
tests are scalar inequalities in quantities a rotation leaves unchanged,
and its tangent reference is its own previous output, carried in state. Its command therefore co-rotates with the frame. The one
exception is the tangent seed on the first ride step, taken from the
measured flow, and Section~\ref{sec:ch7:rotating} gives the condition
under which the two frames agree on it.
```
New:
```
The $s_1$ tracker's channels are built from $\hat{s}_1$, its gradient,
and the strain eigenframe, which all co-rotate with the observer. Three
statements follow, in increasing strength, and they are the three levels
of Section~\ref{sec:ch7:s1_controller}.

First, the features are objective. The strain tensor co-rotates as
$\mathbf{S}' = \mathbf{Q}\mathbf{S}\mathbf{Q}^{\top}$ and its eigenvalues
are unchanged, so $\hat{s}_1$ is the same scalar field in every frame,
while its gradient and the eigenvectors turn with the axes. The capture
test is a ratio of quantities that a rotation leaves invariant, so it gives
the same answer. The determinant is not objective, by
(\ref{eq:appD:shift}).

Second, the command is rotationally covariant where its sign test is
decisive. Given co-rotated inputs and a co-rotated tangent reference, the
channels return the co-rotated command $\mathbf{u}' = \mathbf{Q}\mathbf{u}$.
The tangent reference is the previous output, which a rotating observer
sees turned by $\Omega\,\Delta t$ per step, so the sign rule returns the
same tangent in both frames except where
$|\mathbf{t}_{\mathrm{ref}}^{\top}\mathbf{t}_{\mathrm{raw}}| <
\sin\Omega\,\Delta t$, as at an eigenframe swap. The tangent seed on the
first ride step is the one input that is not objective, since it is taken
from the measured flow, which carries the transport velocity of the frame.

Third, the trajectory is not equivariant in general. The co-rotated
command $\mathbf{u}' = \mathbf{Q}\mathbf{u}$ is not the transformation law
of coordinate velocity in a rotating frame,
$\dot{\mathbf{p}}' = \mathbf{Q}\dot{\mathbf{p}} + \dot{\mathbf{Q}}\mathbf{p}$,
whose second term is the transport velocity of the frame. A cluster that
realizes the co-rotated command still follows the inertial path only if
its across-trench feedback absorbs that term, and the feedback can deliver
no more than the saturated command allows. Section~\ref{sec:ch7:rotating}
tests this in simulation and finds matching paths while the transport
velocity stays inside the delivered speed. This appendix does not prove it.
```

Old:
```
\chapter{Transverse Stability of the Trackers}
\label{app:stability}
```
New:
```
\chapter{Transverse Stability of the Trackers}
\label{app:stability}

% New text, reconciled with the closed-loop discussion of Chapter 7.
This appendix gives two arguments about the loop across the trench: a
Lyapunov argument for the Newton channel, and a bound on the saturated
across-trench loop of each tracker. Both are local and idealized. They
assume that the cluster realizes the commanded velocity exactly, that the
surrogate is quadratic across the trench, and that the estimate error is
bounded. The robot model of Section~\ref{sec:ch2:dynamics} is weaker than
the first assumption, since it attains a share $g$ of a command after a
delay $L$ and a lag $\tau$. For that model the sampled loop across the
trench is stable only for loop gains below $1/(gL)$, as
Section~\ref{sec:ch7:closed_loop} states, and nothing here covers the
switched and saturated controller as a whole.
```

Old:
```
        ('neighborhoods where (\\ref{eq:d_capture}) applies', 'neighborhoods where (\\ref{eq:ch7:d_capture}) applies')]),
])
```
New:
```
        ('neighborhoods where (\\ref{eq:d_capture}) applies', 'neighborhoods where the stop test of Section~\\ref{sec:ch7:sep_controller} applies')]),
T(r"""
% New text, reconciled with the closed-loop discussion of Chapter 7.
The proposition describes the unsaturated, delay-free loop across the
trench. With the robot model of Section~\ref{sec:ch2:dynamics}, the loop
gain $k\,a_\perp$ must also stay below the sampled-loop limit of
Section~\ref{sec:ch7:closed_loop}, and the field sets part of that gain:
$a_\perp = \kappa_\perp/\hat{\kappa}_\perp$ for the $D$ tracker and
$a_\perp = g_\perp\kappa_\perp$ for the $s_1$ tracker. Where the field
raises $a_\perp$ past the limit, the argument gives no bound. The
simulated $s_1$ path wiggles about the separatrix there
(Section~\ref{sec:ch7:disc_clean}).
"""),
])
```

## D (M6): `sources/assemble/ch08.py`
Sec. 8.1: closed-loop numbers rerun on the 2026-11-16 robot model with a 0.30 m triangle, same seed as the 2026-09-30 run (X1, `experiments/interesting/thesis_ch8_magnitude_saddle.py`, outputs `experiments/outputs/thesis_ch8/magnitude_saddle.json`; legacy results kept beside it). Old: canonical saddle 0.0061 m at the 95th percentile, 91.3% of anisotropic-saddle trials within 0.05 m, attraction 0.0147 m at the 95th percentile. New: magnitude descent on the canonical saddle and the vortex, max 0.0044 m (median 0.0031 m); anisotropic saddle, median 0.0475 m, max 0.0484 m, 100% within 0.05 m; attraction law, median 0.0226 m, max 0.0237 m on all three fields (friction ring 0.025 m; Table 6.1 analytic mean 22.6 mm). Fig. 8.1 regenerated. Also: differentiability qualifier (Sec. 8.1), exact vs approximate noise scaling (Sec. 8.3), and the link between the 3.6x fit-error ratio and Draft 14's retention ratios (blue, from Draft 14).

Old:
```
order $m$, a gradient at order $m + 1$, and a curvature at order $m + 2$.
A primitive needs the gradient
```
New:
```
order $m$, a gradient at order $m + 1$, and a curvature at order $m + 2$,
wherever it is differentiable. The magnitude is not differentiable where
the velocity vanishes, which matters at the critical points themselves,
and $s_1 = \mu - r$ is not differentiable at an isotropic point, where
the strain vanishes. A primitive needs the gradient
```

Old:
```
A closed-loop test confirms this with the three-robot simulator of
Chapter~\ref{ch:first_order}. The vector-to-scalar primitive of
Section~\ref{sec:ch5:vector_to_scalar} descended the magnitude from 1000
random starts in a 1~m box around a saddle. On the canonical saddle of
Appendix~\ref{app:fields} it finished within 0.006~m of the saddle in
95\% of trials, the same as on the vortex, whose magnitude landscape is
identical.
```
New:
```
A closed-loop test confirms this with the three-robot simulator of
Chapter~\ref{ch:first_order}, on the robot model of
Section~\ref{sec:ch2:dynamics}. The vector-to-scalar primitive of
Section~\ref{sec:ch5:vector_to_scalar} descended the magnitude from 1000
random starts in a 1~m box around a saddle. On the canonical saddle of
Appendix~\ref{app:fields} it finished within 0.005~m of the saddle in
every trial, the same as on the vortex, whose magnitude landscape is
identical.
```

Old:
```
the saddle, with 91\% of trials within 0.05~m. The attraction law of
Chapter~\ref{ch:first_order} finished within 0.015~m in 95\% of trials on
all three fields (Fig.~\ref{fig:ch8:magnitude}).
```
New:
```
the saddle, with every trial within 0.05~m. The attraction law of
Chapter~\ref{ch:first_order} finished within 0.024~m in every trial on
all three fields, at the edge of the friction ring $v_c/k = 0.025$~m of
Section~\ref{sec:ch6:friction} and in agreement with the 22.6~mm mean
final error of Table~\ref{tab:ch6:simulation_results}
(Fig.~\ref{fig:ch8:magnitude}).
```

Old:
```
where the row norm depends on the formation's shape alone.
```
New:
```
where the row norm depends on the formation's shape alone. This scaling is
exact for a formation of fixed shape, independent noise of equal variance
on every reading, and a full-rank $\boldsymbol{\Phi}_1$.
```

Old:
```
\label{eq:ch8:terminal_noise}
\end{equation}
The first-order law
```
New:
```
\label{eq:ch8:terminal_noise}
\end{equation}
Unlike (\ref{eq:ch8:coef_noise}), this relation is an approximation to
first order in the noise, since $f$ is nonlinear in the coefficients. The
first-order law
```

Old:
```
the $s_1$ tracker's failures under noise to the sign of its tangent,
which is selected from the same gradient.
```
New:
```
the $s_1$ tracker's failures under noise to the sign of its tangent,
which is selected from the same gradient. The ordering matches the
retention levels of Section~\ref{sec:ch7:disc_noise}, where the $D$
tracker withstood \prelim{4.4} to \prelim{5.6} times the measurement noise
of the $s_1$ tracker from the starts above the origin. The two ratios
measure different things. The first compares the fit errors at one point
of the separatrix at a fixed noise level. The second compares the noise
levels at which each tracker's retention falls to 50\%, and acquisition
and reversals enter only the second.
```

## M1: `sources/assemble/ch05.py`
Orbit drift described by its mechanism, not as 'momentum effects not captured in previous simulation models' (Plan0.1 3a, 3b; Plan0.2 Other updates). IDETC results stay as measured; the scope of 'not predicted in simulation studies' is limited to the simulations of the IDETC control-law source. No forward pointer in Ch. 5 (Plan0.1 3c).

Old:
```
The outward drift rate appears stable at all angles, a behavior not
predicted in simulation studies. This drift occurs because the robots,
following only their perceived direction using the vector-sum technique,
are always moving tangent to the circle while experiencing momentum
effects not captured in previous simulation models. This real-world
```
New:
```
The outward drift rate appears stable at all angles, a behavior not
predicted in the simulation studies of \cite{idc:4}. This drift occurs
because the robots, following only their perceived direction using the
vector-sum technique, are always moving tangent to the circle, but each
robot answers a new heading late, after a delay and a lag that those
simulations did not model, so every step leaves the cluster slightly
outside the circle it was on. This real-world
```

Old:
```
flow, and nothing in it measures the distance to the center, so momentum
carried the cluster steadily outward.
```
New:
```
flow, and nothing in it measures the distance to the center, so the late
turns carried the cluster steadily outward.
```

## M1, X3: `sources/assemble/ch06.py`
Sec. 6.2: one backward reference from the closed-form drift to the vector-sum drift of Ch. 5 (Plan0.1 3c). New paragraph in Sec. 6.5.4 (X3): the ten 2025 fixed-vortex vector-sum logs give T = 0.164 s median (quartiles 0.146 to 0.176 s; range 0.134 to 0.236 s) with the map center at the origin and 0.136 s with a common fitted center, against T = L + tau + dt/2 = 0.125 s from the refit; R^2 of r^2 against t from 0.79 to 0.93; mean centroid speeds 0.367 to 0.400 m/s. Source `experiments/thesis_ch5_orbit_drift_check.py` (runs 1 to 10 of `trunk/robots_3/Results_for TestBed_Paper/Fixed_vortex_Vector_Sum`). Plan0.1 3d.

Old:
```
P('sys', r'The orbital law (\ref{eq:11}) is undefined at $r = 0$',
  end=r'\end{equation}'),
```
New:
```
P('sys', r'The orbital law (\ref{eq:11}) is undefined at $r = 0$',
  end=r'\end{equation}',
  subs=[(r'it drifts outward at $TV^2/r^*$.',
         r'it drifts outward at $TV^2/r^*$. This is the outward drift that the vector-sum primitive showed in Chapter~\ref{ch:zeroth}, where nothing opposed it.')]),
```

Old:
```
  subs=[('fig_limit_cycle_2026.pdf', 'ch6_fig_limit_cycle_2026.pdf')],
  width=r'0.9\textwidth'),
```
New:
```
  subs=[('fig_limit_cycle_2026.pdf', 'ch6_fig_limit_cycle_2026.pdf')],
  width=r'0.9\textwidth'),
T(r"""
% New text. Analysis: VF_Robot experiments/thesis_ch5_orbit_drift_check.py,
% output experiments/outputs/thesis_ch5/orbit_drift_check.txt (2026-11-19).
The same delay and lag account for the outward drift of the vector-sum
primitive in Chapter~\ref{ch:zeroth}, which has no radial term to oppose
it. For a cluster of speed $V$ and no radial correction, $r^2$ grows at
the constant rate $2TV^2$. The ten 2025 vector-sum runs on the fixed
vortex grew linearly in $r^2$, with $R^2$ from 0.79 to 0.93 after the
first second, at mean centroid speeds of 0.37 to 0.40~m/s. Their slopes
give a median $T$ of 0.16~s, with quartiles of 0.15 and 0.18~s, when the
map center is taken as the origin of the motion capture frame, and 0.14~s
when a common center is fitted to the ten runs. The delay and lag of
Section~\ref{sec:ch2:dynamics} give 0.125~s.
"""),
```

## C (B1, B2, B3, S1, S2, S3, C1, C2): `sources/assemble/ch08.py`
Tier 4 content from Plan0.2, all in Ch. 8 so that references run backward only. NOTE: Plan0.2 placed B2's table in Ch. 4 and B3's column in Table 4.1; both would point forward to Chs. 6 and 7, so they live here. Numbers: ported Draft 14 numbers (blue where Draft 14 has them blue: 60%, and 4.4 to 5.6 in Sec. 8.4); friction-ring ratios 0.14 (0.025 m over 0.30/sqrt(3) = 0.173 m) and 0.11 (0.0083 m over 0.075 m) are arithmetic from stated numbers; the saddle-read-as-center claim (D' = Omega^2 - s^2) follows from Appendix D and was checked numerically. New figure `ch8_one_flow.png` from `experiments/thesis_ch8_one_flow.py` (X4).

Old:
```
geometry, and in noise. This chapter compares the rungs on those costs.
Three short numerical experiments supply the evidence where no chapter
already does.
```
New:
```
geometry, and in noise. This chapter compares the rungs on one flow, on
the landscapes each can read, and on those costs. Short numerical
experiments supply the evidence where no chapter already does.
```

Old:
```
\section{Vector Fields as Scalar Landscapes}
\label{sec:ch8:scalar}
```
New:
```
\section{One Flow at Three Orders}
\label{sec:ch8:one_flow}

% New text. Checked 2026-11-19 in experiments/thesis_ch8_one_flow.py: the
% shifted field equals the left gyre of Section~\ref{sec:ch3:dg_example} to
% machine precision, and det J vanishes on the diamond.
A cell of the double gyre is a bounded eddy: water circulates about a
center and is held between saddles at the corners of the cell. Two of the
benchmark fields of this dissertation are this one flow.
Chapter~\ref{ch:first_order} tests the first-order estimate on a single
cell centered on the origin,
\begin{equation}
    \mathbf{v} = 0.1\pi\,\bigl(\cos\pi X\,\sin\pi Y,\;
    -\sin\pi X\,\cos\pi Y\bigr),
    \label{eq:ch8:cell}
\end{equation}
and Chapter~\ref{ch:second_order} tracks the separatrix between the two
cells of the steady double gyre of Section~\ref{sec:ch3:dg_example}.
Shifting the left cell by $X = x + 0.5$, with $Y = y$, turns the
double-gyre field into (\ref{eq:ch8:cell}) term for term. The right wall
of the cell, $X = 0.5$, is then the separatrix $x = 0$, and the saddles at
its ends are the flow saddles of Chapter~\ref{ch:second_order}. On the
cell the determinant of the velocity gradient is
$D = \tfrac{1}{2}\pi^4 A^2(\cos 2\pi X + \cos 2\pi Y)$ with $A = 0.1$,
which vanishes on the diamond $|X| + |Y| = 1/2$.

Each rung of the ladder was run on this flow, and
Table~\ref{tab:ch8:one_flow} records what happened.

\begin{table}[htbp]
\centering
\caption{The gyre cell at each rung.}
\label{tab:ch8:one_flow}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{1.6cm} >{\raggedright\arraybackslash}p{4.0cm} >{\raggedright\arraybackslash}p{7.2cm}}
\hline
Rung & What the cluster reads & What it did on this flow \\
\hline
Zeroth & the speed & It could not tell the center from the saddles, since the speed vanishes at all of them, and it lost every orbit (Section~\ref{sec:ch6:comparison}). \\
First & $\hat{\mathbf{J}}$ at one point, from three robots & The sign of $\det\hat{\mathbf{J}}$ at the first step decided which critical point the cluster reached, and orbits were lost once the formation reached the diamond (Section~\ref{sec:ch6:sim_attraction}). \\
Second & the quadratic fit over the footprint, from six robots & Both trackers rode the separatrix, the right wall of the cell, to the saddle at its end (Section~\ref{sec:ch7:disc_clean}). \\
\hline
\end{tabular}
\end{table}

The determinant is the function behind the last two rows, read at two
orders. At first order the cluster reads one value of $D$ at a point, and
its sign sorts the starts into basins: the attraction law is a Newton step
toward a zero of the fitted flow, which cannot cross the diamond
$\det\hat{\mathbf{J}} = 0$. At second order the cluster reads the
landscape of $D$ over its footprint, and the trenches of that landscape
are separatrices (Section~\ref{sec:ch7:surrogates}). The two readings meet
at one point. On the separatrix $D$ is zero only at the crest, where the
diamond touches the wall (Fig.~\ref{fig:ch8:one_flow}). The crest is also
where the along-trench slope of $D$ vanishes, so it is where the Newton
step of the $D$ tracker stalls and the measured flow carries the cluster
over (Section~\ref{sec:ch7:sep_controller}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{ch8_one_flow.png}
    \caption{One gyre cell at three orders, in the cell coordinates of
    (\ref{eq:ch8:cell}). The shading is $D$, and the dotted diamond is
    $\det\mathbf{J} = 0$. Dots are the 1000 starts of the first-order gyre
    trials, blue where the cluster reached the center and orange where it
    reached a saddle. Lines are one noise-free ride of each second-order
    tracker from start S1 of Chapter~\ref{ch:second_order}, shifted into
    cell coordinates; both ride the separatrix, the right wall of the cell,
    to the saddle at its lower corner. Crosses mark the saddles of the cell,
    and the diamond marker is the crest.}
    \label{fig:ch8:one_flow}
\end{figure}

\section{Vector Fields as Scalar Landscapes}
\label{sec:ch8:scalar}
```

Old:
```
    \label{fig:ch8:radius}
\end{figure}

\section{Scalability}
```
New:
```
    \label{fig:ch8:radius}
\end{figure}

Each rung has a signal-to-noise ratio of its own. The signal is the next
derivative of the field times the footprint raised to that order, and
Table~\ref{tab:ch8:snr} sets it against the noise on it and gives the
threshold that each chapter found. The thresholds are not comparable one
to one: the first-order condition is a validity condition for a
covariance prediction and not a failure point, and the zeroth rung has no
measured threshold, so none is stated. The zeroth rung has a sharp
consequence all the same. A direction read from a vector becomes unreliable
where the vector vanishes, and the vector vanishes at the critical points
that the first rung was built to find.

\begin{table}[htbp]
\centering
\caption{The signal each rung reads, the noise on it, and the threshold
found for that rung.}
\label{tab:ch8:snr}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{1.5cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{5.7cm}}
\hline
Rung & Signal & Noise on it & Threshold found \\
\hline
Zeroth & $\lVert\mathbf{v}\rVert$ & sensing error & None measured. Small vectors gave large direction shifts on the RGB testbed (Section~\ref{sec:ch2:hsv}). \\
First & $\sigma_{\min}(\mathbf{J})R$ & $\sigma_v$ & The covariance prediction of $\bar{e}_f$ holds within 5\% when $\sigma_{\min}(\mathbf{J})R \geq 3\sigma_v$ (Section~\ref{sec:ch6:discussion}). \\
Second & $\pi^3 A\rho^2$, the largest second derivative of the gyre times $\rho^2$ & $2.53\,\sigma_{uv}$ & The two are equal near $\sigma_{uv} = 0.007$. By $\sigma_{uv} = 0.01$ neither tracker keeps the straddle in more than \prelim{60}\% of trials from any start (Section~\ref{sec:ch7:disc_noise}). \\
\hline
\end{tabular}
\end{table}

\section{Steady Offsets: Speed over Loop Gain}
\label{sec:ch8:offsets}

% New text. Every row of the table is a result of Chapters 5 to 7 or a
% statement of Chapter 9; nothing here was run for this chapter, except the
% fit of the vector-sum drift in Section~\ref{sec:ch6:limit_cycle}.
Every primitive of this dissertation closes a loop around a target, and
every one settles away from it for the same reason. When some speed
reaches the loop uncancelled, the cluster settles offset from its target by
that speed divided by the loop gain. If the speed exceeds the largest
command the loop can issue, the cluster loses the target.
Table~\ref{tab:ch8:offsets} lists the speeds that reached the loop in the
experiments.

\begin{table}[htbp]
\centering
\caption{Speeds that the loop does not cancel, the loop gain that opposes
them, and the offset or loss that follows. $T$ is the response delay of
Section~\ref{sec:ch6:orbits}.}
\label{tab:ch8:offsets}
\footnotesize
\begin{tabular}{>{\raggedright\arraybackslash}p{3.4cm} c >{\raggedright\arraybackslash}p{5.4cm} >{\raggedright\arraybackslash}p{2.4cm}}
\hline
Speed the loop does not cancel & Loop gain & Result & Where \\
\hline
Dead zone $v_c$ & $k$ & The cluster settles inside the friction ring $v_c/k$: 0.025~m in Chapter~\ref{ch:first_order} and 0.008 in the hold of Chapter~\ref{ch:second_order}. & Sec.~\ref{sec:ch6:friction}, \ref{sec:ch7:closed_loop} \\
Outward drift from late turns, $TV^2/r^*$ & $gk_r$ & Limit cycle, $r^* - r_d = (TV^2/r^*)/(gk_r)$. & Sec.~\ref{sec:ch6:orbits} \\
The same drift with no radial loop & $0$ & $r^2$ grows at $2TV^2$ without bound, as the vector-sum primitive drifted. & Sec.~\ref{sec:ch5:trials}, \ref{sec:ch6:limit_cycle} \\
A critical point moving at speed $V$ & $gk$ & The cluster trails it by $V/(gk)$ (stated, not tested). & Sec.~\ref{sec:ch9:limitations} \\
Frame transport, $0.5\lvert\Omega\rvert$ & $gk$ & The hold trails by $0.5\lvert\Omega\rvert/(gk)$. The $s_1$ hold is lost once $0.5\lvert\Omega\rvert > gkc_{\max}$. & Sec.~\ref{sec:ch7:rotating} \\
A separatrix moving faster than $gkc_{\max} = 0.103$ & saturated & Both channels lose it. & Sec.~\ref{sec:ch7:beyond} \\
\hline
\end{tabular}
\end{table}

The friction ring is the simplest case. The dead zone ignores any command
below $v_c$, so the cluster stops where $k\lVert\mathbf{p}_c -
\hat{\mathbf{p}}^*\rVert = v_c$. The orbit's limit cycle has the same
form. The late turn drifts the cluster outward at $TV^2/r^*$, and the
radial loop of gain $gk_r$ balances it at
$r^* - r_d = (TV^2/r^*)/(gk_r)$, which is (\ref{eq:ch6:orbit_cf})
rearranged. A primitive with no radial loop leaves the drift uncorrected.

All of these settle a loop whose gain is bounded above. The sampled loop
is stable only below $1/(gL) = 21.1$ per unit time, or 16.1 with the lag
included (Section~\ref{sec:ch7:closed_loop}). At first order the designer
sets the gain, which is $k$. At second order the field sets part of it: $k$
times the ratio of true to fitted curvature for the $D$ tracker, and
$k g_\perp\,\partial^2 s_1/\partial n^2$ for the $s_1$ tracker. This is why
the $s_1$ path wiggles below $y = -0.19$, where its gain passes the bound.

\section{Scalability}
```

Old:
```
carry the same limitation with more state to share.
```
New:
```
carry the same limitation with more state to share.

\section{Fitted Features and True Features}
\label{sec:ch8:fitted_true}

% New text.
A cluster steers on a feature of its fit, and the fit stands in for the
flow. Each rung therefore has two features: the one the cluster steers on,
which belongs to the fitted polynomial, and the one it wants, which belongs
to the flow. Table~\ref{tab:ch8:fitted_true} gives, for each rung, when the
two coincide and where they came apart.

\begin{table}[htbp]
\centering
\caption{The feature each rung steers on, the condition under which it is
the true feature, and where it was seen to fail.}
\label{tab:ch8:fitted_true}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{4.6cm}}
\hline
Rung & Steers to & It is the true feature when & It fails visibly on \\
\hline
Zeroth & the minimum of the fitted magnitude plane & the speed grows equally in every direction & the anisotropic node, where the primitive reached the critical point in 6\% of trials, and the gyre, where the speed vanishes at the center and the saddles alike (Section~\ref{sec:ch6:comparison}). \\
First & the zero of the fitted affine field & the field is affine from the cluster out to the critical point & the gyre and the rotational saddle, where the estimate stayed 2.6 to 4.9~mm from the true critical point and the sign of $\det\hat{\mathbf{J}}$ set the basin (Section~\ref{sec:ch6:sim_attraction}). \\
Second & the trench of the fitted $D$ or $s_1$ & the rotation on the separatrix is small against the strain, which is necessary but not sufficient & the bent gyre and the Bickley jet, where each tracker rides the trench of its own field (Section~\ref{sec:ch7:beyond}). \\
\hline
\end{tabular}
\end{table}

The condition is a statement about the flow, which the cluster cannot see.
What it can do is compute quantities from its own fit that signal when the
condition is likely to fail. Table~\ref{tab:ch8:indicators} lists them.

\begin{table}[htbp]
\centering
\caption{Quantities a cluster can compute about its own fit.}
\label{tab:ch8:indicators}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{3.3cm} >{\raggedright\arraybackslash}p{1.7cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{4.0cm}}
\hline
Indicator & Order & Signals & Tested \\
\hline
$\sigma_{\min}(\hat{\mathbf{J}})$ & first & a near-singular critical point estimate, since model error is amplified by $1/\sigma_{\min}$ & Stated in Section~\ref{sec:ch6:discussion}. The attraction law does not gate on it. \\
$\lvert\hat{\alpha}\rvert/\hat{\nu}$ against the center threshold & first & a type label near a boundary & The labels were scored at three thresholds (Table~\ref{tab:ch6:classification_hw}). \\
$\lvert\hat{\omega}\rvert/2\hat{r}$ & first & a trench unlikely to mark a separatrix & Yes: retention falls as the ratio rises (Section~\ref{sec:ch7:beyond}). It needs a heading referenced to an inertial frame. \\
The least-squares residual & any order with more than $n_k$ robots & measurement noise, where the truncation error is small & Not tested (Section~\ref{sec:ch9:higher_order}). \\
\hline
\end{tabular}
\end{table}

\section{Lower Rungs Inside Higher Ones}
\label{sec:ch8:nesting}

% New text. The decision rule below is a synthesis of the two trackers of
% Chapter 7, not a tested result.
The rungs also nest in use. The $D$ tracker is a composition of all three.
It makes its stop decision with the zeroth- and first-order coefficients,
the least noisy that the fit returns: once the fitted flow has a saddle
inside the formation, it latches the attraction law of
Chapter~\ref{ch:first_order}. It signs its direction against the measured
flow, a zeroth-order quantity. Only its directions and steps along and
across the trench, the eigenvectors and curvature of the fitted Hessian of
$D$, are of second order. The $s_1$ tracker nests the other way. It chooses
between the strain eigenvectors $\mathbf{e}_1$ and $\mathbf{e}_2$, a
first-order quantity, by the gradient of $s_1$, a second-order quantity,
and it takes the sign of that choice from memory. Its reversals therefore
concentrate where the along-trench slope goes to zero
(Section~\ref{sec:ch7:disc_noise}), and the future-work rule of
Section~\ref{sec:ch9:higher_order}, taking the eigenvector nearest the
previous tangent, would move that choice down to first order.

These two cases suggest a design rule: make each decision with the
lowest-order coefficients that can make it, since the noise on a
coefficient of order $q$ grows as $\rho^{-q}$ (Section~\ref{sec:ch8:noise_order}).
The rule accounts for the sign and stop decisions of the $D$ tracker. It
does not account for the whole noise gap between the trackers, because the
along-trench direction of the $D$ tracker is itself of second order.

The rungs meet in a second way. Critical points are the nodes of a flow,
and separatrices connect them. On the steady double gyre the separatrix is
a heteroclinic connection, since the flow runs down it from
$\mathbf{p}^*_2$ to $\mathbf{p}^*_1$. The $D$ tracker travels that
connection and captures the node at its end with the test and the law of
Chapter~\ref{ch:first_order}. Section~\ref{sec:ch9:features} lists
heteroclinic orbits as a feature that needs a second-order fit. On this one
steady flow Chapter~\ref{ch:second_order} has done it, and
Section~\ref{sec:ch7:beyond} shows that the time-periodic and jet cases
do not hold up as cleanly.

\section{Classification and Frames Across the Rungs}
\label{sec:ch8:class_frames}

% New text. The saddle-read-as-center statement follows from
% D' = D + Omega*omega + Omega^2 (Appendix D); with omega = 0 and a pure
% strain s, D' = Omega^2 - s^2, and J' has eigenvalues +/- i sqrt(Omega^2 - s^2)
% for Omega > s. Checked numerically 2026-11-19.
Every rung is asked what it is looking at. The zeroth rung cannot say: the
speed looks the same at a saddle and a center
(Section~\ref{sec:ch8:scalar}). The first names the type from the
eigenvalues of $\hat{\mathbf{J}}$ (Chapter~\ref{ch:first_order}). The
second types what it rides. At sea the tangent of the $s_1$ tracker
followed $\mathbf{e}_2$, an attracting OECS, on 34 of 42 steps from 7 to
14~h, and a backward FTLE field checked it (Section~\ref{sec:ch7:disc_ocean}).
The hard cases sit on the classifier's boundaries at both rungs. A center
has $\operatorname{tr}\mathbf{J} = 0$, and on hardware only 45\% and 47\%
of its cycles were labeled strictly as a center with three and four robots
(Section~\ref{sec:ch6:hw_class}). At an isotropic point, where the strain
vanishes, the tangent of the $s_1$ tracker swaps eigenvector and its
reversals begin.

Table~\ref{tab:ch8:scalar} marks which fields are objective, and the
primitives need the same column (Table~\ref{tab:ch8:objectivity}). A
first-order fit is not objective either. With no vorticity and a pure
strain $s$, an observer rotating at $\Omega$ reads $D' = \Omega^2 - s^2$
(Appendix~\ref{app:frame_equivariance}), so one that rotates faster than
the strain rate reads an irrotational saddle as a center.

\begin{table}[htbp]
\centering
\caption{Objectivity of the primitives. A primitive is objective when
observers rotating relative to each other agree on what it steers on.}
\label{tab:ch8:objectivity}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.8cm} >{\raggedright\arraybackslash}p{5.2cm}}
\hline
Primitive & Reads & Objective \\
\hline
Vector-sum, vector-to-scalar & the sensed vector and its magnitude & no \\
Attraction, orbital & the fitted affine field & no \\
$D$ tracker & $\hat{D}$ and its Hessian & no \\
$s_1$ tracker & $\hat{s}_1$, its gradient, the strain eigenframe & the features, yes; the command, where its sign test is decisive; the path, only empirically (Appendix~\ref{app:frame_equivariance}) \\
\hline
\end{tabular}
\end{table}

``Instantaneous'' also needs one scope sentence. Estimation is memoryless,
but control is not: the $D$ tracker latches its stop test, and the $s_1$
tracker carries one sign bit. That bit buys its objectivity and is the
source of its reversals.

\section{Design Guidelines}
\label{sec:ch8:guidelines}

% New text. The ratios 0.14 and 0.11 are arithmetic on stated numbers:
% 0.025 m over the center-to-robot distance of the 0.30 m triangle,
% 0.30/sqrt(3) = 0.173 m, and 0.008 m over the pentagon radius 0.075 m.
Every result of the previous chapters is stated in formation radii or in
ratios of speeds, and that is why one controller ran on a 1.6~m map and, by
a uniform dilation, on a 51.4~km channel. Table~\ref{tab:ch8:groups} collects
the dimensionless groups behind the results.

\begin{table}[htbp]
\centering
\caption{Dimensionless groups behind the results of this dissertation.}
\label{tab:ch8:groups}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{4.4cm} >{\raggedright\arraybackslash}p{4.0cm} >{\raggedright\arraybackslash}p{4.0cm}}
\hline
Group & Governs & Where \\
\hline
$\rho$ over the length on which the field curves & truncation bias & Sections~\ref{sec:ch4:size}, \ref{sec:ch8:noise_order} \\
Signal over noise, by rung & noise failure & Table~\ref{tab:ch8:snr} \\
$gkL$ & stability of the sampled loop & Section~\ref{sec:ch7:closed_loop} \\
$v_c/(k\rho)$ & the friction ring in formation radii, 0.14 of the center-to-robot distance of the triangle in Chapter~\ref{ch:first_order} and 0.11 of the pentagon radius in Chapter~\ref{ch:second_order} & Table~\ref{tab:ch8:offsets} \\
$\lvert\omega\rvert/2r$ & whether a trench marks a separatrix & Section~\ref{sec:ch7:beyond} \\
Feature or frame speed over $gkc_{\max}$ & tracking a moving feature or holding in a rotating frame & Table~\ref{tab:ch8:offsets} \\
\hline
\end{tabular}
\end{table}

Four guidelines follow, each from a result above. First, size the
formation to the field. A larger formation suppresses the noise on a
coefficient of order $q$ as $\rho^{-q}$ and raises the truncation error with
$\rho$, so the radius belongs where the two curves cross
(Section~\ref{sec:ch8:noise_order}). Second, keep the robots off the
degenerate curve and watch the condition number: the pentagon plus center
gives 8.26 for a quadratic fit, and the best ten-robot layout found for a
cubic fit gives 44.3 (Section~\ref{sec:ch8:geometry}). Third, keep the loop
gain below the sampled-loop limit and the friction ring small against the
formation radius, remembering that at second order the field sets part of the
gain (Section~\ref{sec:ch8:offsets}). Fourth, choose the tracker by the
constraint: rotation of the observer favors the $s_1$ tracker, while noise
and directional reliability favor the $D$ tracker, and a cluster with an
absolute heading reference gives up little by carrying the $D$ tracker
(Chapter~\ref{ch:second_order}).
```

## C (S3 fix): `sources/assemble/ch08.py`
Opening of the design-guidelines section no longer implies that the six-robot trackers ran on a 1.6 m map; they ran in simulation on the nondimensional double gyre and the dilated channel.

Old:
```
Every result of the previous chapters is stated in formation radii or in
ratios of speeds, and that is why one controller ran on a 1.6~m map and, by
a uniform dilation, on a 51.4~km channel. Table~\ref{tab:ch8:groups} collects
the dimensionless groups behind the results.
```
New:
```
The thresholds of the second-order trackers are stated in formation radii,
which is why the same trackers ran on the nondimensional double gyre and,
after a uniform dilation, on the Santa Barbara Channel, where one length
unit is 51.4~km (Section~\ref{sec:ch7:ocean_sim}).
Table~\ref{tab:ch8:groups} collects the dimensionless groups behind the
results of this dissertation.
```

## J2 (thesis statement, contributions): `sources/assemble/ch01.py`
Thesis statement rewritten as Plan0 Sec. 3A suggests (instantaneous local reconstruction determines the field information available; scoped to one set of simultaneous measurements; fit order defined as the order of vector-field reconstruction, Plan0 3B). Contributions: seven, the Ch. 7 evidence named as simulation and Ch. 8 added as the last (Plan0 Sec. 1.5 and optional list; Plan0.1 4b). Three-robot minimum scoped as unrestricted affine fit from one set of measurements (Plan0 Sec. 6.2.1).

Old:
```
The order of the local fit a cluster makes to its measurements determines
which features of a vector field it can reach. A heading read directly
from the samples supports only direction-following behaviors. A
first-order fit
```
New:
```
The order of the instantaneous local fit that a cluster makes to its
measurements determines the field information available to it, and so the
feature-based navigation it can perform. The statement concerns what one
set of simultaneous measurements supports; methods that add temporal
measurements or prior knowledge lie outside it. Fit order here means the
order to which the vector field itself is reconstructed. A heading, or the
slope of the sensed magnitude, read directly from the samples supports
direction-following behaviors, which reach a target that the flow carries
the cluster to and drift when asked to orbit. A first-order fit
```

Old:
```
This dissertation makes six contributions. First,
```
New:
```
This dissertation makes seven contributions. First,
```

Old:
```
measurements alone, and proves that three robots are the minimum for this
estimate (Chapter~\ref{ch:first_order}).
```
New:
```
measurements alone, and proves that three robots are the minimum for an
unrestricted affine fit from one set of measurements
(Chapter~\ref{ch:first_order}).
```

Old:
```
common conic (Chapter~\ref{ch:second_order}). Finally, it builds two
primitives on that fit, the $D$ tracker and the $s_1$ tracker, which
acquire and ride a separatrix from instantaneous measurements without
pre-straddling (Chapter~\ref{ch:second_order}). The $D$ tracker tolerates
more measurement noise, while the $s_1$ tracker, being objective, holds
its path under a rotating observer.
```
New:
```
common conic (Chapter~\ref{ch:second_order}). Sixth, it builds two
primitives on that fit, the $D$ tracker and the $s_1$ tracker, which
acquire and ride the trench of their field from instantaneous measurements
without pre-straddling, and it tests them in simulation on a double gyre,
on three further fields, and on measured ocean currents
(Chapter~\ref{ch:second_order}). The $D$ tracker tolerates more
measurement noise, while the $s_1$ tracker, being built from objective
quantities, holds its path under a rotating observer. Finally, it compares
the three rungs on one flow, on formation geometry, on noise, and on steady
offsets, and it states for each rung when the feature that it steers to is
the feature of the flow (Chapter~\ref{ch:crosscutting}).
```

## J3: `thesis_main.tex`
Abstract: thesis statement; 'in simulation' at the first second-order result (Plan0.1 4a); Draft 14 results replace '1.1 to 5.8 times' and 'successfully follow' (Draft 14 abstract; Plan0.2 'What Draft_14 forces' 2); which calibration the R^2 of 0.96/0.91 belongs to, and the recalibration (6.1 deg, 3.8%, Sec. 2.4.4); one sentence for the Ch. 8 findings. No blue numbers in the abstract.

Old:
```
dissertation applies that process to vector fields and shows that the
order of the local fit a cluster makes to its measurements sets which
features of the field it can reach. An indoor testbed encodes direction in
hue and magnitude in saturation on printed floor maps, and a supervised
learning neural network calibration achieves an encoding-decoding $R^2$ of
0.96 in direction and 0.91 in magnitude. On it, direction-only primitives
```
New:
```
dissertation applies that process to vector fields and shows that the
order of the instantaneous local fit a cluster makes to its measurements
determines the field information available to it, and so the feature-based
navigation it can perform. An indoor testbed encodes direction in hue and
magnitude in saturation on printed floor maps, and a cross-robot neural
network calibration achieves an encoding-decoding $R^2$ of 0.96 in direction
and 0.91 in magnitude. After maintenance, a recalibration with pooled
training reads direction to within 6.1$^\circ$ and magnitude to within
3.8\% for the trained robots. On the testbed, direction-only primitives
```

Old:
```
second-order fit from six robots recovers the local quadratic model of a
flow from one synchronized set of measurements. Two primitives built on it
locate the separatrix and navigate to and along it with no
integration horizon and no pre-straddling. Monte Carlo sweeps show the
determinant tracker withstands 1.1 to 5.8 times more measurement noise
than the strain tracker, depending on the start, while in a rotating-frame trial only the strain
tracker, which is objective, holds the same path. On Santa
Barbara Channel radar currents, both primitives successfully follow the dominant
transport corridor.
```
New:
```
second-order fit from six robots recovers the local quadratic model of a
flow from one synchronized set of measurements. In simulation, two
primitives built on it navigate to and along the trenches of two fields
computed from that fit, the determinant of the velocity gradient and the
smaller strain eigenvalue, which lie closest to a separatrix where strain
dominates rotation, with no integration horizon and no pre-straddling. The
determinant tracker withstands more measurement noise than the strain
tracker from every start tested, while in a rotating-frame trial only the
strain tracker, which is built from objective quantities, holds the same
path. On Santa Barbara Channel radar currents, both primitives acquire the
dominant transport corridor, but under small changes to the start only the
determinant tracker keeps its downstream direction. Across the three rungs,
the noise on a fitted coefficient grows as the inverse formation radius
raised to the order of the coefficient, and the steady offset of a cluster
from its target is the speed that its loop does not cancel divided by the
loop gain.
```

## M7: `sources/assemble/ch04.py`
Ch. 4: coefficients are estimates of Taylor derivatives, with truncation error O(rho^(k+1-q)) under stated conditions; 'algebraic curve of degree at most k'; fit order defined as the order of vector-field reconstruction, with the vector-to-scalar primitive named a first-order scalar fit that sits on the zeroth rung; 'cannot follow a curve' scoped (Plan0 Secs. 4.2.1, 4.2.2, 4.3 and 3B, 3D). Consistent with Ch. 8: noise-free fits grew as rho^0.95 and rho^0.98 for the second-order trackers (k = 2, q = 2 gives rho^1).

Old:
```
measured from the cluster centroid. The scaling makes each fitted
coefficient a Taylor derivative of the field at the centroid. The basis
```
New:
```
measured from the cluster centroid. The scaling makes each fitted
coefficient an estimate of the corresponding Taylor derivative of the field
at the centroid. The basis
```

Old:
```
lie on a common algebraic curve of degree $k$. For $k = 1$ that curve is
a line, and for $k = 2$ it is a conic.
```
New:
```
lie on a common algebraic curve of degree at most $k$. For $k = 1$ that
curve is a line, and for $k = 2$ it is a conic, which may be a pair of
lines.
```

Old:
```
$k + 1$, so the bias they leave grows with $\rho$. A larger formation
therefore trades truncation bias for noise suppression.
```
New:
```
$k + 1$, so the bias they leave grows with $\rho$. Under smoothness and a
fixed, full-rank normalized formation, the truncation error in an
order-$q$ derivative estimated by an order-$k$ fit generally behaves as
$O(\rho^{\,k+1-q})$, and leading terms can cancel by symmetry or by the
structure of the field. A larger formation therefore trades truncation
bias for noise suppression.
```

Old:
```
Table~\ref{tab:ch4:ladder} organizes this dissertation by fit order. Each
rung fixes a failure of the one below it. The zeroth-order primitives of
```
New:
```
Table~\ref{tab:ch4:ladder} organizes this dissertation by fit order, the
order to which the vector field itself is reconstructed from one set of
simultaneous measurements (Section~\ref{sec:ch1:thesis}). The
vector-to-scalar primitive fits a plane to the sensed magnitudes, which is
a first-order fit of a scalar field, but it reconstructs the vector field
only to zeroth order, so it sits on the lowest rung. Each rung fixes a
failure of the one below it. The zeroth-order primitives of
```

Old:
```
it gives the velocity gradient at one point only, so it cannot follow a
curve.
```
New:
```
it gives the velocity gradient at one point only, so one instantaneous
first-order fit does not by itself identify a curve to follow.
```

## M7: `sources/assemble/ch08.py`
Same 'degree at most k' wording as Ch. 4.

Old:
```
robots lie on a common algebraic curve of degree $k$, a line for the
```
New:
```
robots lie on a common algebraic curve of degree at most $k$, a line for the
```

## M6: `sources/assemble/ch06.py`
Sec. 6.2.1: the three-robot minimum is stated for an unrestricted affine fit from one set of instantaneous measurements (Plan0 Sec. 6.2.1).

Old:
```
P('sys', r'each contain three coefficients. From', end=r'\end{IEEEproof}', subs=IEEEPROOF),
```
New:
```
P('sys', r'each contain three coefficients. From', end=r'\end{IEEEproof}',
  subs=IEEEPROOF + [('Therefore: three robots are the minimum required for critical point estimation.',
                     'Therefore: three robots are the minimum required to recover an unrestricted affine model, and with it the critical point, from one set of instantaneous measurements. A prior on the field, or measurements over time, could reduce the count.')]),
```

## J4 (M6): `sources/assemble/ch10.py`
Ch. 10: Ch. 7 scoped as simulation in the first sentence of its paragraph (Plan0.1 4c); Table 10.1 stability column stated as local and conditional, with a 'Uses' column for the lower rungs (Plan0.2 B3) and a caption sentence on what the robot counts mean (Plan0 Ch. 10); Final Thoughts restated with the thesis statement, the three-study trend (Plan0.2 'Trends', closed-form agreement 3 mm from Sec. 6.5.1) and future work d) from Plan0 optional; recalibration sentence to match the abstract.

Old:
```
subs=[('This paper has shown', 'Chapter~\\ref{ch:second_order} showed')]),
```
New:
```
subs=[('This paper has shown that six robots', 'Chapter~\\ref{ch:second_order} showed, in simulation, that six robots')]),
```

Old:
```
the quantity it steers on, its stability result, and how it was verified.
```
New:
```
the quantity it steers on, its stability result, how it was verified, and
the lower rungs it calls on. The robot counts are the smallest clusters
with which each primitive was run and, for the fits, the minimum for an
unrestricted fit of that order from one set of instantaneous measurements;
they are not minimums for navigation in general.
```

Old:
```
\begin{table}[htbp]
\centering
\footnotesize
\caption{Primitive library. Stability entries are as stated in the
chapter that defines each primitive.}
\label{tab:ch10:library}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{2.0cm} c c >{\raggedright\arraybackslash}p{2.7cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.0cm}}
\hline
Primitive & Order & Robots & Steers on & Stability & Verification \\
\hline
Vector-sum & 0 & 3 & mean sensed vector & none stated & hardware, fixed and sinking vortex, 10 runs each \\
Vector-to-scalar & 0 & 3 & slope of the sensed magnitude & none stated & hardware, fixed vortex, 10 runs plus randomized starts \\
Attraction & 1 & 3 & $\hat{\mathbf{p}}^* = -\hat{\mathbf{J}}^{-1}\hat{\mathbf{h}}$ & exponential convergence to a stationary estimate, $\tau = 1/k$ & simulation, 11 fields; hardware, 239 trials \\
Orbital & 1 & 3 & $\hat{\mathbf{p}}^*$ with radial and tangential terms & radial error $\dot{e}_r = -k_r e_r$ & simulation, 11 fields; hardware, 220 runs \\
$D$ tracker & 2 & 6 & $\hat{D}$ and $\hat{\mathbf{H}}_{D,0}$ & transverse practical stability, $\limsup|n| \leq \delta/a_\perp$ & simulation, double gyre and Santa Barbara Channel \\
$s_1$ tracker & 2 & 6 & $\hat{s}_1$ and $\nabla\hat{s}_1$ & same transverse bound, frame equivariant past its seed step & simulation, double gyre and Santa Barbara Channel \\
\hline
\end{tabular}
\end{table}
```
New:
```
\begin{table}[htbp]
\centering
\footnotesize
\caption{Primitive library. Stability entries are as stated in the
chapter that defines each primitive, and each is local and conditional on
the assumptions stated there.}
\label{tab:ch10:library}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{1.7cm} c c >{\raggedright\arraybackslash}p{2.3cm} >{\raggedright\arraybackslash}p{3.1cm} >{\raggedright\arraybackslash}p{2.9cm} >{\raggedright\arraybackslash}p{2.1cm}}
\hline
Primitive & Order & Robots & Steers on & Stability & Verification & Uses \\
\hline
Vector-sum & 0 & 3 & mean sensed vector & none stated & hardware, fixed and sinking vortex, 10 runs each & none \\
Vector-to-scalar & 0 & 3 & slope of the sensed magnitude & none stated & hardware, fixed vortex, 10 runs plus randomized starts & none \\
Attraction & 1 & 3 & $\hat{\mathbf{p}}^* = -\hat{\mathbf{J}}^{-1}\hat{\mathbf{h}}$ & exponential convergence to a stationary estimate, $\tau = 1/k$ & simulation, 11 fields; hardware, 239 trials & none \\
Orbital & 1 & 3 & $\hat{\mathbf{p}}^*$ with radial and tangential terms & radial error $\dot{e}_r = -k_r e_r$ & simulation, 11 fields; hardware, 220 runs & none \\
$D$ tracker & 2 & 6 & $\hat{D}$ and $\hat{\mathbf{H}}_{D,0}$ & across-trench error bounded by $\delta/a_\perp$ for an ideal robot; local and linearized for the refit model (Appendix~\ref{app:stability}) & simulation: double gyre, three further fields, Santa Barbara Channel & measured flow (order 0); stop test and attraction law (order 1) \\
$s_1$ tracker & 2 & 6 & $\hat{s}_1$ and $\nabla\hat{s}_1$ & same bound; features objective, command covariant where the sign test is decisive (Appendix~\ref{app:frame_equivariance}) & same & strain eigenvectors (order 1); memory of the last tangent \\
\hline
\end{tabular}
\end{table}
```

Old:
```
show that the order of a cooperative fit sets which features of a flow a
cluster can reach, from a heading, to a critical point, to a separatrix.
Ongoing and future work targets a) hardware trials of the second-order
trackers on the Decabot testbed, b) control that accounts for a flow that
advects the robots, and c) a third-order fit from ten robots.
```
New:
```
show that the order of the instantaneous cooperative fit determines the
field information a cluster has, and so the features of a flow that it can
navigate, from a heading, to a critical point, to a separatrix.

The evidence changed in kind from one study to the next. The testbed study
reported an outward drift that simulation had not predicted. The
first-order study explained it with a closed form that matches simulation
within 3~mm. The second-order study stated where both of its trackers must
fail, near $\sigma_{uv} = 0.007$, before it ran the Monte Carlo sweeps that
confirm it. The test conditions widened in the same order, from ten runs
per case, to 459 hardware runs with matched simulations, to sweeps over
rotating frames, three further fields, and perturbations of the start of
the ocean run. Each claim also carries its scope condition, such as an
affine field out to the critical point, strain that dominates rotation, or
the tested environments. Only the second rung has reached measured ocean
currents, and it has done so in simulation.

Ongoing and future work targets a) hardware trials of the second-order
trackers on the Decabot testbed, b) control that accounts for a flow that
advects the robots, c) a third-order fit from ten robots, and d) switching
between estimator orders from the indicators of
Section~\ref{sec:ch8:fitted_true}.
```

Old:
```
confirming behaviors previously demonstrated only in
simulation.
```
New:
```
confirming behaviors previously demonstrated only in
simulation. After preventive maintenance, a pooled recalibration read
direction to within 6.1$^\circ$ and magnitude to within 3.8\% for the
trained robots (Section~\ref{sec:ch2:calibration}).
```

## J1, M3, M5 (1c, 1e): `sources/assemble/ch02.py`
Ch. 2. One testbed description: IDETC '3 rovers, 2.3 m^2 maps' now '3 or 4 rovers, 1.6 m by 1.6 m maps' (your answer: same maps; 2.3 m^2 was approximate), and the revision paragraph gives each campaign's robot count (Plan0.1 1c). Sec. 2.1: Arduino to Intel Edison on an Arduino breakout board with a Teensy 3.2 (Plan0.1 'also found'; Hart thesis description). Calibration: Sec. 2.4.4 states the metric conversion (6.1 deg = 0.017 turns; 3.8% = 0.038; arithmetic). Dead zone: Sec. 2.6 says once that the fitted 0.025 m/s and the 2025 0.05 m/s are different measurements and neither is direct (SIM_RERUN_PLAN_2026-11-16; the X3 drift check supports the refit timing, not the floor) (Plan0.1 2c). Sec. 2.9: '0.05 to 0.3 m/s' attributed to the 2025 tests. New closing section 'Verification by Substitution' with the substitution table and the IDETC-to-refit arc (Plan0.2 M1, M2); the IDETC digital-twin future work is described as delivered.

Old:
```
that point. Each is powered by an Arduino and communicates over TCP/IP to
```
New:
```
that point. Each is powered by an Intel Edison on an Arduino breakout board,
with a Teensy 3.2 that reads the encoders and the light sensor, and
communicates over TCP/IP to
```

Old:
```
consistent sensing capabilities. It includes three omnidirectional mobile
rovers (Decabots), 2.3~m$^2$ multicolored floor maps representing
```
New:
```
consistent sensing capabilities. It includes three or four omnidirectional
mobile rovers (Decabots), 1.6~m by 1.6~m multicolored floor maps representing
```

Old:
```
subs=[('The testbed used for validation is', 'The same testbed, used for the critical point experiments of Chapter~\\ref{ch:first_order}, is')]),
```
New:
```
subs=[('The testbed used for validation is', 'The testbed of Chapters~\\ref{ch:zeroth} and~\\ref{ch:first_order}, with three robots in the first and three or four in the second, is')]),
```

Old:
```
The robots were recalibrated after preventive maintenance, before the
critical point experiments of Chapter~\ref{ch:first_order}. The
recalibrated models use the four raw channels as inputs and are the ones
the later hardware results rely on.
```
New:
```
The robots were recalibrated after preventive maintenance, before the
critical point experiments of Chapter~\ref{ch:first_order}. The
recalibrated models are the ones the later hardware results rely on.
Their errors are reported as RMS errors in degrees of hue and percent of
saturation, where Table~\ref{tab:ch2:calibration} gives the circular $R^2$
and the RMSE of hue in turns and of saturation on a unit scale. An RMS error
of 6.1$^\circ$ in hue is an RMSE of 0.017 on the scale of that table, and
3.8\% in saturation is 0.038.
```

Old:
```
life per robot, and a limited velocity range of 0.05 to 0.3~m/s.
Additionally
```
New:
```
life per robot, and a limited velocity range of 0.05 to 0.3~m/s in the 2025
tests (Section~\ref{sec:ch2:dynamics}).
Additionally
```

Old:
```
\section{Field Reconstruction as a Noise Model}
```
New:
```
% New text. The fitted dead zone and the 0.05 m/s of the 2025 tests.
The dead zone of 0.025~m/s is a fitted value. The 2025 tests place the
floor near 0.05~m/s: lanes below a saturation of 0.3, a command of about
0.09~m/s, did not overcome static friction (Section~\ref{sec:ch2:saturation}).
The two numbers measure different things. A lane test combines the decoding
of a low saturation with the friction of the robot, while the fit uses the
commanded and the measured velocity directly. The runs of both campaigns
come to rest where the attraction command falls to about 0.03 to
0.05~m/s, and no run commanded less than about 0.06~m/s, so neither number
is a direct measurement of the threshold.

\section{Field Reconstruction as a Noise Model}
```

Old:
```
significant operational costs.
"""),
]
```
New:
```
significant operational costs.

\section{Verification by Substitution}
\label{sec:ch2:substitution}

% New text. The studies of this dissertation earn their claims one way: swap
% one part of the system, hold the rest fixed, and see what changes.
The claims of this dissertation rest on one habit: swap one part of the
system, hold the rest fixed, and see what changes.
Table~\ref{tab:ch2:substitution} lists the substitutions, and each isolates
a single question.

\begin{table}[htbp]
\centering
\caption{Substitutions used to isolate one question at a time.}
\label{tab:ch2:substitution}
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{5.2cm} >{\raggedright\arraybackslash}p{7.0cm} c}
\hline
Substitution & What it isolates & Chapter \\
\hline
Robot R4 held out of the calibration training & whether the sensor calibration generalizes to a new robot & \ref{ch:tools} \\
Analytical field against reconstructed field & the fixed distortion of the printed map against measurement noise that changes in time & \ref{ch:first_order} \\
A simulation matched to each hardware run & what the model reproduces and what it omits & \ref{ch:first_order} \\
The vector-to-scalar primitive alone, with an oracle target, and with a speed estimate & the missing critical point coordinate against the steering direction & \ref{ch:first_order} \\
Measurement noise and position noise swept separately & which noise limits each tracker & \ref{ch:second_order} \\
Scoring against the separatrix and against the tracker's own trench, with analytic trenches & the field against the tracker & \ref{ch:second_order} \\
A rotating observer & objectivity & \ref{ch:second_order} \\
A replay of the ocean start with small perturbations & whether the headline run was selected & \ref{ch:second_order} \\
\hline
\end{tabular}
\end{table}

The studies also form a sequence. The testbed paper found an outward drift
that simulation had not predicted, and it promised digital twins of the
Decabots that capture their dynamics for simulation, together with
controller substitution on a fixed lower architecture. The first-order
study delivered both. It fitted the robot model of
Section~\ref{sec:ch2:dynamics}, used it to explain the drift as a limit
cycle, and substituted the vector-to-scalar primitive for the Jacobian
method with the lower layers held fixed. The second-order study runs its
six-robot simulations on the same model.
"""),
]
```

## 1e: `sources/assemble/ch02.py`
Sec. 2.4.4: the revision's description of the palette and the two networks, already given in Secs. 2.4.1 to 2.4.3, is cut to the two things that changed, the four raw input channels and pooled training (Plan0.1 1e). The sentence 'The encoding and calibration palette follows [22]' goes with it; the same citation is in Sec. 2.4.1.

Old:
```
effects. ', '')]),
```
New:
```
effects. ', ''),
        ("The robots' RGB sensors are calibrated against a printed palette spanning hue $0$ to $2\\pi$ and saturation $0$ to $1$ at 24 intervals in each dimension. The encoding and calibration palette follows~\\cite{22}. Each robot records raw RGB and clear-channel counts over every palette cell. The readings are normalized and stored as a four-dimensional feature vector $(R,G,B,K)$. Two feedforward networks, each with two hyperbolic-tangent hidden layers, are trained to recover the target hue and saturation from this vector. The saturation network has a single output, while the hue network outputs the sine and cosine of hue to respect its circularity. Training pools data across robots so that a single model can be applied to an uncalibrated unit without per-robot tuning.",
         "The recalibration repeats the palette and the two networks of the previous sections with two changes. The input is the four normalized raw channels $(R,G,B,K)$ in place of seven inputs, and training pools data across robots so that a single model can be applied to an uncalibrated unit without per-robot tuning.")]),
```

## M2: `sources/assemble/ch05.py`
Ch. 5 Sec. 5.5: formation precision stated as Table 5.1 shows it (mean sides within 1 cm of 0.35 m, mean angle within 0.03 rad of 1.05 rad, standard deviations 0.9 to 3.2 cm and 0.045 to 0.12 rad), and the OptiTrack-resolution clause dropped (Plan0.1 2e). Summary: one backward reference from the drift to the fitted delay and lag (Plan0.2 M2).

Old:
```
precise formation control, as evidenced by
Table~\ref{tab:ch5:formation}. The values stayed within 1~cm of desired
parameters ($p = 0.35$~m, $q = 0.35$~m, and $\beta = 1.05$~radians) across
all test environments. This formation precision, within the resolution
limits of the OptiTrack motion capture system, demonstrates the
```
New:
```
formation control, as evidenced by
Table~\ref{tab:ch5:formation}. The mean side lengths stayed within 1~cm of
the commanded values ($p = 0.35$~m and $q = 0.35$~m) and the mean angle
within 0.03~rad of $\beta = 1.05$~rad across all test environments, with
standard deviations of 0.9 to 3.2~cm in the sides and 0.045 to 0.12~rad in
the angle. This demonstrates the
```

Old:
```
so the late
turns carried the cluster steadily outward.
```
New:
```
so the late
turns carried the cluster steadily outward. Those turns are late because of
a delay and a lag that Section~\ref{sec:ch2:dynamics} fits to the hardware.
```

## J6, B1: `sources/assemble/ch03.py`
Ch. 3: the trenches of D and s1 named as surrogates for Lagrangian structures, with the strain-over-rotation condition (Plan0 3E; Draft 14 Sec. VI wording, citing Haller 2015). One sentence that the Ch. 6 gyre cell is the left cell of the double gyre (Plan0.2 B1); the equation and the numerical check are in Sec. 8.1.

Old:
```
\section{Benchmark Fields}
\label{sec:ch3:benchmarks}
```
New:
```
% New text. Instantaneous Eulerian fields are surrogates for material
% transport structures (Draft 14 Sec. VI; Plan0 Sec. 3E).
Both fields are instantaneous and Eulerian, so they are surrogates for the
Lagrangian structures that carry material, and not the structures
themselves \cite{sep:19}. A trench of $D$ or $s_1$ marks a separatrix of
the flow only where the rotation on it is small against the strain, and
even there it need not match the Lagrangian structure. Chapter~\ref{ch:second_order}
states each result for the trench, and it names a separatrix only where the
benchmark supports it.

\section{Benchmark Fields}
\label{sec:ch3:benchmarks}
```

Old:
```
\subsection{The Steady Double Gyre}
\label{sec:ch3:dg_example}
```
New:
```
\subsection{The Steady Double Gyre}
\label{sec:ch3:dg_example}

% New text. The nonlinear gyre cell of Chapter 6 is one cell of this flow.
The gyre cell that tests the first-order estimate in
Chapter~\ref{ch:first_order} is the left cell of this flow, shifted so that
its center sits at the origin.
```

## J5: `sources/assemble/ch09.py`
Ch. 9: what the fitted model still misses, from Ch. 6 (estimate lead 1.7 to 2.8 cm, vortex attraction 0.4 to 1.5 cm farther, four-robot orbit shortfall) and the unmeasured low-speed response (Plan0.2 Other updates); time-periodic gyre result from Draft 14 Sec. IV-E (Plan0.2 forced 5); new subsection 'Choosing the Fit Order' (Plan0 optional, built from the Ch. 8 indicators). The sentences 'one simulator cannot host both models' and the [26] veering claim are already gone with the Draft 14 port (Plan0.2 forced 4).

Old:
```
T(r"""
\subsection{Second-Order Estimation and Tracking}
\label{sec:ch9:second_order}
"""),
```
New:
```
T(r"""
% New text. What the fitted robot model still misses (Chapter 6).
The fitted robot model of Section~\ref{sec:ch2:dynamics} closes part of the
gap between simulation and hardware that Chapter~\ref{ch:zeroth} found.
Chapter~\ref{ch:first_order} reports three differences that it leaves. The
live estimate led the cluster along its motion by 1.7 to 2.8~cm in small
orbits, a lead that the model omits. The hardware vortex ended 0.4 to
1.5~cm farther from the critical point than the matched simulations. The
orbits of the four-robot square fell short of the triangle's, which the
matched simulations did not predict. The shape of the low-speed response
is also unmeasured, since no run commanded less than about 0.06~m/s.

\subsection{Second-Order Estimation and Tracking}
\label{sec:ch9:second_order}
"""),
```

Old:
```
The Santa Barbara Channel trial of Section~\ref{sec:ch7:ocean} ran the
second-order trackers on a field that changes in time, but on one record
and with no analytic guarantee.
```
New:
```
The Santa Barbara Channel trial of Section~\ref{sec:ch7:ocean} ran the
second-order trackers on a field that changes in time, but on one record
and with no analytic guarantee. On a time-periodic double gyre,
Section~\ref{sec:ch7:beyond} found that both trackers follow the separatrix
at the smaller amplitude, and that at the larger the separatrix outruns the
speed that either channel can deliver.
```

Old:
```
\subsection{Three-Dimensional Fields}
```
New:
```
\subsection{Choosing the Fit Order}
\label{sec:ch9:switching}

% New text. Built from the indicators of Table~\ref{tab:ch8:indicators}; none of
% this was built or tested.
The indicators of Table~\ref{tab:ch8:indicators} give a cluster the inputs
for choosing its own fit order. A cluster could hold a first-order fit while
$\lvert\hat{\omega}\rvert/2\hat{r}$ stays small and its type labels sit away
from their thresholds, and move to a second-order fit when a curve is worth
following. A fit from more robots than $n_k$ could signal, through its
residual, when the measurement noise is too large for either tracker. None
of this was built or tested.

\subsection{Three-Dimensional Fields}
```

Old:
```
\subsection{Testbed Extensions}
\label{sec:ch9:testbed}
```
New:
```
\subsection{Additional Topological Features, Continued}

% New text. Placed with the features subsection above in the compiled text.
```

## J5 (repair): `sources/assemble/ch09.py`
Repairs the previous Ch. 9 edit, which had replaced the 'Testbed Extensions' heading by a placeholder. Heading restored; the heteroclinic paragraph now ends the features subsection (Plan0.2 B3, Sec. 9.2.5); the IDETC digital-twin and controller-substitution plans are marked as done in part, and the low-speed test is named (SIM_RERUN_PLAN_2026-11-16, Sec. 1).

Old:
```
\subsection{Additional Topological Features, Continued}

% New text. Placed with the features subsection above in the compiled text.
```
New:
```
% New text. Chapter 7 followed one heteroclinic connection on the steady flow.
On the steady double gyre the separatrix between the two saddles is a
heteroclinic connection, and the $D$ tracker travels it
(Section~\ref{sec:ch8:nesting}). The bent gyre and the Bickley jet of
Section~\ref{sec:ch7:beyond} show that a trench of a fitted surrogate
follows such a connection only where strain outweighs rotation, so
following heteroclinic connections in general fields remains future work.

\subsection{Testbed Extensions}
\label{sec:ch9:testbed}
```

Old:
```
larger scales and real-world deployment challenges.
```
New:
```
larger scales and real-world deployment challenges.

% New text. The IDETC digital-twin and controller-substitution plans, now done in part.
Part of this has been done. Section~\ref{sec:ch2:dynamics} fits the dynamics
of the Decabots for simulation, and Section~\ref{sec:ch6:comparison}
substituted the vector-to-scalar primitive for the Jacobian method with the
cluster and robot layers held fixed. One further test would settle the dead
zone of Section~\ref{sec:ch2:dynamics}: a single robot stepped through
commands from 0.01 to 0.08~m/s, from rest and from motion.
```

## M8 (CI): `sources/assemble/ch06.py`
Sec. 6.5.1 gives 95% Wilson intervals for the hardware convergence rates at 0.05 m, computed from the counts in `analysis/hardware/outputs/hardware_summary.json`: saddle 3R 79/80, [93.3, 99.8]%; vortex 3R 73/80, [83.0, 95.7]%; vortex 4R 79/79, [95.4, 100]% (Plan0 optional). The counts reproduce the 99, 91 and 100% already in the text.

Old:
```
subs=[(r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}'),
        (r'with matched simulations
```
New:
```
subs=[(r'and in 80\% and 39\% at 0.03~m.',
         r'and in 80\% and 39\% at 0.03~m. At the 0.05~m threshold the 95\% Wilson intervals are 93 to 100\% for the saddle and 83 to 96\% for the vortex.'),
        (r'ended within 0.05~m and 65\% within 0.03~m.',
         r'ended within 0.05~m (95 to 100\%, 95\% Wilson interval) and 65\% within 0.03~m.'),
        (r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}'),
        (r'with matched simulations
```

## M8 (availability): `sources/assemble/ch01.py`
New Sec. 1.7, a reproducibility statement (Plan0 optional). No URL, since whether the repository is public is your call (as in Draft 14).

Old:
```
\section{Dissertation Organization}
\label{sec:ch1:organization}
```
New:
```
\section{Availability of Code and Data}
\label{sec:ch1:availability}

% New text. Plan0 optional reproducibility statement. Each campaign folder
% has a README that lists its files, robots, and maps.
The code for every simulation, the scripts behind each figure, the
calibration data, and the logs of the three hardware campaigns are
available from the author on request. Each hardware campaign has a README
that lists its files, its robots, and its maps.

\section{Dissertation Organization}
\label{sec:ch1:organization}
```

## M8 (assumptions table): `sources/assemble/ch10.py`
New Table 10.2: the assumptions behind each primitive in one place, each taken from the chapter that states it (Plan0 optional).

Old:
```
\section{Final Thoughts}
\label{sec:ch10:final}
```
New:
```
Table~\ref{tab:ch10:assumptions} states, for each primitive, the
assumptions its results rest on.

\begin{table}[htbp]
\centering
\footnotesize
\caption{Assumptions behind each primitive. Every primitive also assumes a
static field that does not advect the robots, and the second-order trackers
assume a centralized cluster with ample communication.}
\label{tab:ch10:assumptions}
\begin{tabular}{>{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{12.0cm}}
\hline
Primitive & Assumptions \\
\hline
Vector-sum, vector-to-scalar & A sensed vector or magnitude that is decoded accurately away from zero speed; for the vector-to-scalar primitive, a speed that grows equally in every direction about the target. \\
Attraction, orbital & Three robots not on a line; $\det\mathbf{J} \neq 0$; a field that is affine from the cluster out to the critical point; an estimate that is stationary over the analysis; a robot that realizes the command, with a loop gain below the sampled-loop limit. \\
$D$ tracker & Six robots not on a conic; a fitted quadratic that is adequate across the footprint; a trench of $D$ that marks the separatrix, which needs strain to outweigh rotation on it; a measured flow that gives the direction downstream. \\
$s_1$ tracker & The same fit and the same trench condition; an along-trench slope of $\hat{s}_1$ large enough to select the tangent; a tangent seed taken from the measured flow, which is the one input that is not objective. \\
\hline
\end{tabular}
\end{table}

\section{Final Thoughts}
\label{sec:ch10:final}
```

## M8 (CI, anchor fix): `sources/assemble/ch06.py`
The interval sentences belong to the text passage of Sec. 6.5.1, not to the table passage that the previous edit touched.

Old:
```
subs=[(r'and in 80\% and 39\% at 0.03~m.',
         r'and in 80\% and 39\% at 0.03~m. At the 0.05~m threshold the 95\% Wilson intervals are 93 to 100\% for the saddle and 83 to 96\% for the vortex.'),
        (r'ended within 0.05~m and 65\% within 0.03~m.',
         r'ended within 0.05~m (95 to 100\%, 95\% Wilson interval) and 65\% within 0.03~m.'),
        (r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}'),
```
New:
```
subs=[(r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}'),
```

Old:
```
P('sys', r'Following the test plan,',
  end='matched the localization error'),
```
New:
```
P('sys', r'Following the test plan,',
  end='matched the localization error',
  subs=[(r'and in 80\% and 39\% at 0.03~m.',
         r'and in 80\% and 39\% at 0.03~m. At the 0.05~m threshold the 95\% Wilson intervals are 93 to 100\% for the saddle and 83 to 96\% for the vortex.'),
        (r'ended within 0.05~m and 65\% within 0.03~m.',
         r'ended within 0.05~m (95 to 100\%, 95\% Wilson interval) and 65\% within 0.03~m.')]),
```

## E8: `thesis_main.tex`
Front matter (Plan0 Required). The signature page spilled onto a second page, and the titlepage environment reset the page counter so that Roman numerals repeated (i, ii, i, ii). The approval page is now unnumbered and uncounted, the title page is page i without a number, and the copyright page is ii, with no repeats. Spacing on the approval page reduced to fit one page. NOT done: the date (still 'Month Day, 2026') and the reader names, which need your input, and the department's own numbering convention, which I could only infer from the SCU template and the example thesis in this folder.

Old:
```
% Front matter uses roman numerals
\pagenumbering{roman}
\setcounter{page}{1}
```
New:
```
% Front matter. The approval page carries no number and is not counted; the
% title page is page i and carries no number. The titlepage environment is not
% used because it resets the page counter, which repeated the numerals.
\pagenumbering{gobble}
```

Old:
```
\begin{titlepage}
\begin{center}
  \large
  Santa Clara University \\[6pt]
  Department of \thesisdepartment

  \vspace{1.5in}
```
New:
```
\thispagestyle{empty}
\begin{center}
  \large
  Santa Clara University \\[6pt]
  Department of \thesisdepartment

  \vspace{0.7in}
```

Old:
```
  Date: \thesismonth\ \thesisday, \thesisyear

  \vspace{1in}
```
New:
```
  Date: \thesismonth\ \thesisday, \thesisyear

  \vspace{0.6in}
```

Old:
```
\vspace{0.8in}

\noindent
\begin{tabular}{p{2.6in} p{0.2in} p{2.6in}}
```
New:
```
\vspace{0.4in}

\noindent
\begin{tabular}{p{2.6in} p{0.2in} p{2.6in}}
```

Old:
```
  & & \hrulefill \\[2pt]
  & & Thesis Reader
\end{tabular}

\end{titlepage}
```
New:
```
  & & \hrulefill \\[2pt]
  & & Thesis Reader
\end{tabular}

\clearpage
\pagenumbering{roman}
\setcounter{page}{1}
```

Old:
```
\begin{titlepage}
\begin{center}
  \vspace*{1in}
```
New:
```
\thispagestyle{empty}
\begin{center}
  \vspace*{1in}
```

Old:
```
  Santa Clara, California \\
  \thesisyear
\end{center}
\end{titlepage}
```
New:
```
  Santa Clara, California \\
  \thesisyear
\end{center}
\clearpage
```

## E8 (spacing): `thesis_main.tex`
Approval page spacing tightened further so that the signature block stays on the page.

Old:
```
  \vspace{0.7in}

  \normalsize
  Date: \thesismonth\ \thesisday, \thesisyear

  \vspace{0.6in}

  I HEREBY RECOMMEND THAT THE THESIS PREPARED UNDER MY SUPERVISION BY

  \vspace{0.5in}
  \MakeUppercase{\thesisauthor}
  \vspace{0.5in}

  ENTITLED

  \vspace{0.4in}
  \textit{\thesistitle}
  \vspace{0.4in}

  BE ACCEPTED IN PARTIAL FULFILLMENT OF THE REQUIREMENTS FOR THE DEGREE

  \vspace{0.3in}
  \textbf{DOCTOR OF PHILOSOPHY IN \MakeUppercase{\thesisdegreefield}}
\end{center}

\vspace{0.4in}
```
New:
```
  \vspace{0.3in}

  \normalsize
  Date: \thesismonth\ \thesisday, \thesisyear

  \vspace{0.3in}

  I HEREBY RECOMMEND THAT THE THESIS PREPARED UNDER MY SUPERVISION BY

  \vspace{0.2in}
  \MakeUppercase{\thesisauthor}
  \vspace{0.2in}

  ENTITLED

  \vspace{0.15in}
  \textit{\thesistitle}
  \vspace{0.15in}

  BE ACCEPTED IN PARTIAL FULFILLMENT OF THE REQUIREMENTS FOR THE DEGREE

  \vspace{0.15in}
  \textbf{DOCTOR OF PHILOSOPHY IN \MakeUppercase{\thesisdegreefield}}
\end{center}

\vspace{0.25in}
```

Old:
```
  Thesis Advisor & & Thesis Reader \\[24pt]
  \hrulefill & & \hrulefill \\[2pt]
  Chairman of Department & & Thesis Reader \\[24pt]
  & & \hrulefill \\[2pt]
  & & Thesis Reader \\[24pt]
```
New:
```
  Thesis Advisor & & Thesis Reader \\[20pt]
  \hrulefill & & \hrulefill \\[2pt]
  Chairman of Department & & Thesis Reader \\[20pt]
  & & \hrulefill \\[2pt]
  & & Thesis Reader \\[20pt]
```

## J4 (wording): `sources/assemble/ch10.py`
Ch. 10 uses the thesis-statement wording of the abstract and Sec. 1.4 (same phrase in all places).

Old:
```
show that the order of the instantaneous cooperative fit determines the
field information a cluster has,
```
New:
```
show that the order of the instantaneous local fit that a cluster makes
determines the field information it has,
```

## K1g: `sources/assemble/ch01.py`
Sec. 1.2.2: the IDETC sentence 'only validated in simulation' contradicted the next paragraph, which cites a flow-tank experiment [Michini et al., IROS 2014]. Scoped to 'validated mainly in simulation', with the flow-tank extension named, as Draft 14 describes it (Plan0.1 1g). I did not open the cited works; the characterization is Draft 14's.

Old:
```
comprehensive real-world validation and are only validated in simulation
\cite{idc:4}.
```
New:
```
comprehensive real-world validation: they were validated mainly in
simulation \cite{idc:4}, and the PIM-triple approach of \cite{sep:26} was
extended to robots in a flow tank \cite{sep:28}.
```

