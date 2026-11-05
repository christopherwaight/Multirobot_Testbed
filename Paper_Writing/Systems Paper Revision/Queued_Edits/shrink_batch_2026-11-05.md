# Shrink-and-fix batch, 2026-11-05

Target: `cwaight_systems_paper/cwaight_systems_paper_with_revisions2.tex`. Apply by exact-text match; abort the batch on any miss.
Fixes are F1 to F5 (approved plan). S- edits are shrink or survival. A deleted line shows NEW as (deleted).

Result on the candidate (Queued_Edits/candidate_revisions2.tex): 12 pages, both bios on page 12 with about 40 pt to spare, no errors, no undefined references, no warnings; survival.py 61.2% (was 56.5%); abstract 241 words; no new data numbers (only figure widths and table padding).

Re-apply against the live file with: python3 Queued_Edits/shrink_batch_2026-11-05.py <live.tex> <out.tex> <out.md> (every old string must match exactly once, or it stops).

## F1c
Fix 1. Equation moved to III-D

OLD:
```latex
\begin{equation}\label{eq:orbit_cf}
```
NEW:
```latex
(deleted)
```

## F1d
Fix 1. Equation moved to III-D

OLD:
```latex
r^* = \tfrac{1}{2}\bigl(r_d + \sqrt{r_d^2 + 4C}\bigr), \qquad C = \tau V^2/k_r.
```
NEW:
```latex
(deleted)
```

## F1e
Fix 1. Equation moved to III-D (the end{equation} line right after it)

OLD:
```latex
\end{equation}
```
NEW:
```latex
(deleted)
```

## F1f
Fix 1. Moved into F1b (numbers) and F1a (limits)

OLD:
```latex
The error $r^* - r_d$ approaches the minimum radius $r_{\min} = \sqrt{C}$ as $r_d \to 0$ and falls as $C/r_d$ at large radii. Using the effective lag of the sampled loop in place of $\tau$, the closed form matches simulation to a median of \rev{0.03}~mm across sweeps of gains, lag, sample period, and friction. At the settings of Section~IV, $r_{\min} = \rev{0.133}$~m, and the error falls below 0.05~m only for $r_d \ge \rev{0.31}$~m. Lowering the orbital speed $k_t$ to 0.1~m/s cuts $r_{\min}$ to \rev{0.058}~m, and removing the lag cuts it to \rev{0.059}~m. Raising the radial gain $k_r$ to 2~s$^{-1}$ cuts it to \rev{0.067}~m, at the cost of increased sensitivity to noise in the critical point estimate. % [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3)
```
NEW:
```latex
(deleted)
```

## F1a
Fix 1. Orbit closed form stated as the code computes it (theory.orbit_C); derivation moved here from VI-A

OLD:
```latex
Lag and friction set the steady orbit radius even with an exact estimate. Friction takes $v_c$ off the orbit speed, $V = k_t - v_c$. In steady circular motion at radius $r^*$ the commanded velocity turns at the rate $V/r^*$, and the lag makes the cluster follow it about $\tau$ late, which tilts its velocity outward by the angle $\tau V/r^*$. The cluster therefore drifts outward at $\tau V^2/r^*$.
```
NEW:
```latex
Lag and friction set the steady orbit radius even with an exact estimate. Friction takes $v_c$ off the orbit speed, $V = k_t - v_c$, and passes only the fraction $V/k_t$ of the radial command. The lag makes the cluster follow the turning command a time $T$ late, so in a steady orbit of radius $r^*$ it drifts outward at $TV^2/r^*$. The radial term of (\ref{eq:11}) cancels this drift at
\begin{equation}\label{eq:orbit_cf}
r^* = \tfrac{1}{2}\bigl(r_d + \sqrt{r_d^2 + 4C}\bigr), \qquad C = TVk_t/k_r.
\end{equation}
In the sampled loop $T = \alpha'\Delta t/(1-\alpha') + \Delta t/2$, where $\alpha' = \alpha V/(\alpha V + (1-\alpha)k_t)$ is the lag pole after friction. Without sampling or friction, $T = \tau$ and $C = \tau V^2/k_r$. The error $r^* - r_d$ approaches the minimum radius $r_{\min} = \sqrt{C}$ as $r_d \to 0$ and falls as $C/r_d$ at large radii.
```

## F1b
Fix 1. VI-A keeps the submitted sentences and the numbers; the derivation now lives in III-D

OLD:
```latex
Positive radial bias at small orbital radii is also a result of robot dynamics. There the commanded angular rate $k_t/r_d$, set by the orbital speed $k_t$, is high relative to the radial gain $k_r$, and the outward drift from the actuator lag of Section~III-D sets the bias. With time constant $\tau$ and orbit speed $V = k_t - v_c$, the radial term of (\ref{eq:11}) cancels that drift when $k_r(r^* - r_d) = \tau V^2/r^*$, whose positive root is
```
NEW:
```latex
Positive radial bias at small orbital radii is also a result of robot dynamics. At small radii, the commanded angular rate $k_t / r_d$ is high relative to the radial correction bandwidth $k_r$, and the outward drift from the actuator lag sets the bias (\ref{eq:orbit_cf}). The closed form matches simulation to a median of \rev{0.03}~mm across sweeps of gains, lag, sample period, and friction. At the settings of Section~IV, $r_{\min} = \rev{0.133}$~m, and the error falls below 0.05~m only for $r_d \ge \rev{0.31}$~m. Lowering the orbital speed $k_t$ to 0.1~m/s cuts $r_{\min}$ to \rev{0.058}~m, and removing the lag cuts it to \rev{0.059}~m. Raising the radial gain $k_r$ to 2~s$^{-1}$ cuts it to \rev{0.067}~m, at the cost of increased sensitivity to noise in the critical point estimate. % [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3); C = T V k_t / k_r is theory.orbit_C
```

## F2
Fix 2. r = 0 guard phrased as what the controller could do

OLD:
```latex
When $r < 10^{-6}$~m the implementation sets $\mathbf{n}$ to an arbitrary unit vector, so the command stays defined and its tangential step $k_t\mathbf{t}$ carries the cluster off the singular point on the next cycle.
```
NEW:
```latex
When $r < 10^{-6}$~m, the controller could simply take a step in a random direction, which carries the cluster off the singular point on the next cycle.
```

## F3a
Fix 3. Field count (Table II has three nonlinear columns)

OLD:
```latex
across six canonical linear vector fields and two nonlinear fields using
```
NEW:
```latex
across six canonical linear vector fields and three nonlinear fields using
```

## F3b
Fix 3. Field count

OLD:
```latex
across six vector field types and two nonlinear fields. It also states
```
NEW:
```latex
across six vector field types and three nonlinear fields. It also states
```

## F3d
Fix 3. Field count

OLD:
```latex
Two nonlinear fields test how the estimate extrapolates.
```
NEW:
```latex
Three nonlinear fields test how the estimate extrapolates.
```

## F3e
Fix 3. Field count

OLD:
```latex
Simulations were therefore run on ten fields in total.
```
NEW:
```latex
Simulations were therefore run on eleven fields in total.
```

## F3f
Fix 3. Field count

OLD:
```latex
For each of the ten fields, the cluster
```
NEW:
```latex
For each of the eleven fields, the cluster
```

## F3g
Fix 3. Field count

OLD:
```latex
\caption{Simulation results on ten fields.
```
NEW:
```latex
\caption{Simulation results on eleven fields.
```

## F3h
Fix 3. Field count

OLD:
```latex
evaluated on the same ten fields at different
```
NEW:
```latex
evaluated on the same eleven fields at different
```

## F4
Fix 4. Amplification is 1/sigma_min(J) per (8), not kappa(J); edge-of-map bias comes from the extrapolation term of (8)

OLD:
```latex
Poor conditioning of the estimated Jacobian, measured by $\kappa(\hat{\mathbf{J}})$ in (\ref{eq:8}), amplifies coefficient errors into errors in the critical point estimate (\ref{eq:9}). Unlike the condition number of $\mathbf{A}$, it is a property of the field and is not under the designer's control. It accounts for the orbital bias at the largest commanded radius falling below the controller-only bias on both reconstructed fields, to \rev{+0.026} against \rev{+0.035}~m on the vortex and \rev{+0.001} against \rev{+0.017}~m on the saddle. Near the edge of the mapped region the reconstructed field degrades, the estimated distance to the critical point is inflated, and the radial controller pulls the cluster inward. Where $\det\hat{\mathbf{J}}$ approaches zero, $\kappa(\hat{\mathbf{J}})$ grows without bound, and Section~III-D describes how the controller handles it. A center is non-hyperbolic yet has $\det\mathbf{J} > 0$, so (\ref{eq:9}) locates it as well as any other critical point, and only its type is fragile (Section~IV-C). % [REVIEW] revision_metrics/results/summary.json, reconstructed orbit r_d 0.50 (bias_w1, ctrl_bias_w1)
```
NEW:
```latex
Poor conditioning of the estimated Jacobian amplifies coefficient errors into errors in the critical point estimate (\ref{eq:9}) by the factor $1/\sigma_{\min}(\hat{\mathbf{J}})$ in (\ref{eq:bound}). Unlike the condition number of $\mathbf{A}$, it is a property of the field and is not under the designer's control. Where $\det\hat{\mathbf{J}}$ approaches zero, so does $\sigma_{\min}(\hat{\mathbf{J}})$, and Section~III-D describes how the controller handles it. A center is non-hyperbolic yet has $\det\mathbf{J} > 0$, so (\ref{eq:9}) locates it as well as any other critical point, and only its type is fragile (Section~IV-C).

The measurement term of (\ref{eq:bound}) grows with the extrapolation distance $\|\mathbf{e}\|/R$, and the reconstructed field degrades near the edge of the mapped region. Together they account for the orbital bias at the largest commanded radius falling below the controller-only bias on both reconstructed fields, to \rev{+0.026} against \rev{+0.035}~m on the vortex and \rev{+0.001} against \rev{+0.017}~m on the saddle. There the estimated distance to the critical point is inflated, and the radial controller pulls the cluster inward. % [REVIEW] revision_metrics/results/summary.json, reconstructed orbit r_d 0.50 (bias_w1, ctrl_bias_w1)
```

## F5a
Fix 5. Saturation (R1.2) addressed in one sentence

OLD:
```latex
leave the centroid error bounded.
```
NEW:
```latex
leave the centroid error bounded. The speed cap shortens the velocity without turning it, and it is inactive within $v_{\max}/k$ of the estimate, where this linear analysis applies.
```

## F5b
Fix 5. Unverified tangential estimate-error paragraph cut (nothing used it)

OLD:
```latex
To first order, an estimate error adds its radial component to the radial error. A tangential component $\delta_t$ tilts the tangent command by $\delta_t/r_d$, and the radial term cancels the drift $k_t\delta_t/r_d$ at an offset $k_t\delta_t/(k_r r_d)$, largest at small radii.
```
NEW:
```latex
(deleted)
```

## S-ab1
Survival. Submitted opening restored

OLD:
```latex
A variety of interesting phenomena exist at
```
NEW:
```latex
There are a variety of interesting phenomena that exist at
```

## S-ab2
Shrink and survival. Last sentence split and shortened, submitted opening words restored

OLD:
```latex
While these findings establish the estimation method under lab conditions where printed fields do not advect the robots, the present controller does not yet compensate for the disturbance in oceanic and atmospheric flows where the field physically pushes the platform.
```
NEW:
```latex
These findings establish this estimation method under lab conditions, where printed fields do not advect the robots. The present controller does not yet compensate for oceanic and atmospheric flows that push the platform.
```

## S-I1
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
systems enhance redundancy, throughput, coverage, and reconfigurability in
```
NEW:
```latex
systems enhance redundancy, throughput, coverage, and reconfigurability of robotic systems in
```

## S-I2
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
environmental features \cite{2,3,4}.
```
NEW:
```latex
environmental features across a region \cite{2,3,4}.
```

## S-I3
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
collects a scalar measurement as input
```
NEW:
```latex
collects a single scalar measurement as the input
```

## S-I4
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
and several have been field-deployed
```
NEW:
```latex
and several such systems have been field-deployed
```

## S-I5
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
eddy acts as a center, trapping a body of water and the plankton
```
NEW:
```latex
eddy serves as a center, trapping a body of water, along with the plankton
```

## S-I6
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
a fixed station would not
```
NEW:
```latex
a ground-fixed station would not
```

## S-I7
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
These applications need the coordinate
```
NEW:
```latex
These applications require the coordinate
```

## S-I8
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
drive the team onto
```
NEW:
```latex
drive the multirobot team onto
```

## S-I9
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
reconstructed on hardware from
```
NEW:
```latex
reconstructed on physical hardware from
```

## S-I10
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
The reconstruction's conditioning degrades
```
NEW:
```latex
The conditioning of this reconstruction degrades
```

## S-I11
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
Both the estimator and the cluster's mechatronics limit
```
NEW:
```latex
Both the estimator and the mechatronics of the cluster limit
```

## S-I12
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
are the minimum required for this estimation, and
```
NEW:
```latex
are the minimum number required for this estimation, and
```

## S-I8
Shrink and survival. Submitted paragraph restored without "distributed" and with "indicate the type" (round-1 R1.7 wording); centralized-computation sentence kept; nothing new

OLD:
```latex
This paper introduces a cooperative Jacobian-inversion framework that shifts multirobot vector field navigation from local directional guidance to explicit coordinate inference and feedback control. Fitting a local affine model to three non-collinear measurements is algebraically compact, but closing a cluster control loop on its zero addresses three limitations of existing flow navigation methods. Directional policies \cite{23,33} and magnitude descent \cite{48} guide the robots along a heading or a manifold, which cannot by itself hold a standoff distance. Inverting the assembled Jacobian $\hat{\mathbf{J}}$ instead yields the coordinate $\hat{\mathbf{p}}^*$, which both brings the cluster to rest at the critical point and holds circular orbits at any standoff radius $r_d$. Because the full $2 \times 2$ Jacobian is preserved, one estimator also locates and classifies all six nondegenerate critical point types in a single time step, without the temporal aliasing of a single moving sensor. The paper also links formation size to closed-loop accuracy under real robot dynamics. It proves the three-robot minimum, quantifies the trade-off between field curvature and measurement error, and derives the conditions for sampled-loop stability, the settling error under stiction, and the orbit radius that actuator lag allows. In the present implementation the robots sense and move, and a central computer forms the estimate and computes the commands.
```
NEW:
```latex
This paper introduces a cooperative estimation strategy that recovers the local linear structure of the field. Using a minimum of three robots, the cluster estimates the local Jacobian in a single time step and treats critical point navigation as a local-to-global inference. Inverting the Jacobian predicts the location of the nearby critical point that organizes the surrounding flow, and its eigenvalues indicate the type, with no global map required. The same estimate supports two control laws, a proportional law that drives the cluster centroid onto the critical point, and an orbital law that holds the centroid at a commanded standoff radius. In the present implementation the robots sense and move, and a central computer forms the estimate and computes the commands.
```

## S-II1
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
to each point $\mathbf{p} \in \mathbb{R}^n$ in a given domain
```
NEW:
```latex
to each point $\mathbf{p} \in \mathbb{R}^n$ within a given domain
```

## S-II2
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
For the planar fields of this work, this takes the form
```
NEW:
```latex
For the planar fields analyzed in this work, this takes the specific form
```

## S-II3
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
determined by the Jacobian $\mathbf{J} = 
```
NEW:
```latex
determined by the Jacobian matrix $\mathbf{J} = 
```

## S-II4
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
The eigenvalues of $\mathbf{J}$ at $\mathbf{p}^*$
```
NEW:
```latex
The eigenvalues of $\mathbf{J}$ evaluated at $\mathbf{p}^*$
```

## S-II5
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
degenerate scenarios such as a zero determinant.
```
NEW:
```latex
degenerate scenarios such as cases where the determinant is zero.
```

## S-II6
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
The coefficients are found by solving
```
NEW:
```latex
The coefficients are determined by solving
```

## S-II7
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
are the collected vector field measurements.
```
NEW:
```latex
are the collected field measurements.
```

## S-II8
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
Two control primitives drive the cluster centroid
```
NEW:
```latex
We develop two control primitives that drive the cluster centroid
```

## S-II9
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
The mapping from cluster velocity commands to robot velocities is handled by the cluster space controller of Section~III.
```
NEW:
```latex
The mapping from cluster-level velocity commands to individual robot velocities is handled by the cluster space controller described in Section~III.
```

## S-IIB1
Shrink. Scope paragraph tightened; submitted sentences and math unchanged

OLD:
```latex
Equation (\ref{eq:9}) returns the zero of the fitted affine field $\hat{\mathbf{v}}(\mathbf{p}) = (\hat{U}(\mathbf{p}), \hat{V}(\mathbf{p})) = \hat{\mathbf{J}}\mathbf{p} + \hat{\mathbf{h}}$, which need not be a zero of the true field. To relate the two, let $\mathbf{p}^*$ be any critical point of the true field, so that $\mathbf{v}(\mathbf{p}^*) = \mathbf{0}$. Substituting $\hat{\mathbf{h}} = -\hat{\mathbf{J}}\hat{\mathbf{p}}^*$ writes the fitted field about its own zero, $\hat{\mathbf{v}}(\mathbf{p}) = \hat{\mathbf{J}}(\mathbf{p} - \hat{\mathbf{p}}^*)$, and evaluating it at $\mathbf{p}^*$ gives
```
NEW:
```latex
Equation (\ref{eq:9}) returns the zero of the fitted affine field $\hat{\mathbf{v}}(\mathbf{p}) = \hat{\mathbf{J}}\mathbf{p} + \hat{\mathbf{h}} = \hat{\mathbf{J}}(\mathbf{p} - \hat{\mathbf{p}}^*)$, which need not be a zero of the true field. Evaluating it at any critical point $\mathbf{p}^*$ of the true field gives
```

## S-IIB2
Shrink. Scope paragraph tightened

OLD:
```latex
The true field vanishes at $\mathbf{p}^*$, so the term $\hat{\mathbf{v}}(\mathbf{p}^*)$ is the error of the affine model at the true critical point, and the estimate coincides with $\mathbf{p}^*$ exactly when that error is zero. If the true field is affine over a region containing both the cluster and $\mathbf{p}^*$, the models (\ref{eq:2})--(\ref{eq:3}) reproduce it without error and the recovery is exact at any formation size. On a smooth nonlinear field, the fit carries a first-order linearization error that scales with the local curvature of the field; this error decreases as the formation contracts toward the critical point. The inference from $\hat{\mathbf{p}}^*$ to a true critical point is therefore valid only where the affine model remains accurate from the cluster out to $\mathbf{p}^*$.
```
NEW:
```latex
The true field vanishes at $\mathbf{p}^*$, so $\hat{\mathbf{v}}(\mathbf{p}^*)$ is the error of the affine model there. If the true field is affine over a region containing both the cluster and $\mathbf{p}^*$, the models (\ref{eq:2})--(\ref{eq:3}) reproduce it without error and the recovery is exact at any formation size. On a smooth nonlinear field, the fit carries a first-order linearization error that scales with the local curvature of the field; this error decreases as the formation contracts toward the critical point. The inference from $\hat{\mathbf{p}}^*$ to a true critical point is therefore valid only where the affine model remains accurate from the cluster out to $\mathbf{p}^*$.
```

## S-IIB3
Shrink. Correlation paragraph tightened, same content

OLD:
```latex
Both $\hat{\mathbf{J}}$ and $\hat{\mathbf{h}}$ are computed from the same measurements, so their errors are correlated. Because (\ref{eq:scope}) evaluates the fitted field at $\mathbf{p}^*$, the two errors act jointly through the vector $\hat{\mathbf{v}}(\mathbf{p}^*)$ and need not be bounded separately. Taking norms in (\ref{eq:scope}) gives $\|\hat{\mathbf{p}}^* - \mathbf{p}^*\| \le \|\hat{\mathbf{v}}(\mathbf{p}^*)\|/\sigma_{\min}(\hat{\mathbf{J}})$. The singular value $\sigma_{\min}(\hat{\mathbf{J}})$ is the slowest rate at which the fitted field changes in any direction, so a poorly conditioned Jacobian amplifies the model error.
```
NEW:
```latex
The errors in $\hat{\mathbf{J}}$ and $\hat{\mathbf{h}}$ come from the same measurements and are correlated, but (\ref{eq:scope}) carries them jointly through $\hat{\mathbf{v}}(\mathbf{p}^*)$. Taking norms gives $\|\hat{\mathbf{p}}^* - \mathbf{p}^*\| \le \|\hat{\mathbf{v}}(\mathbf{p}^*)\|/\sigma_{\min}(\hat{\mathbf{J}})$. The singular value $\sigma_{\min}(\hat{\mathbf{J}})$ is the slowest rate at which the fitted field changes in any direction, so a small value amplifies the model error.
```

## S-IIB4
Shrink. Bound derivation prose tightened; math unchanged

OLD:
```latex
Let $\mathbf{p}_c = \frac{1}{3}\sum_i\mathbf{p}_i$ be the cluster centroid, $\mathbf{e} = \mathbf{p}_c - \mathbf{p}^*$ its displacement from the critical point, and $R$ the distance from $\mathbf{p}_c$ to each robot of the equilateral formation. Let $M$ bound the second derivatives of the field, the largest Hessian norm of $u$ or $v$ over the region holding the robots and $\mathbf{p}^*$, and let $\bar{\varepsilon}$ bound the measurement error of each robot. Expanding the field to second order about $\mathbf{p}_c$ leaves each robot's reading off the linearization by at most $MR^2/\sqrt{2} + \bar{\varepsilon}$. The fit reproduces the linearization exactly and carries these deviations to $\mathbf{p}^*$ through the barycentric weights $w_i = \frac{1}{3} - \frac{2}{3}(\mathbf{p}_i - \mathbf{p}_c)^T\mathbf{e}/R^2$ of $\mathbf{p}^*$, whose magnitudes sum to at most $1 + \sqrt{2}\|\mathbf{e}\|/R$. Adding the linearization's own error $M\|\mathbf{e}\|^2/\sqrt{2}$ at $\mathbf{p}^*$ and bounding the curvature terms by $\tfrac{M}{\sqrt{2}}(R + \|\mathbf{e}\|)^2$ gives
```
NEW:
```latex
Let $\mathbf{p}_c = \frac{1}{3}\sum_i\mathbf{p}_i$ be the cluster centroid, $\mathbf{e} = \mathbf{p}_c - \mathbf{p}^*$ its displacement from the critical point, and $R$ the distance from $\mathbf{p}_c$ to each robot of the equilateral formation. Let $M$ bound the Hessian norms of $u$ and $v$ over the region holding the robots and $\mathbf{p}^*$, and let $\bar{\varepsilon}$ bound each robot's measurement error. Each reading then lies within $MR^2/\sqrt{2} + \bar{\varepsilon}$ of the linearization about $\mathbf{p}_c$. The fit carries these deviations to $\mathbf{p}^*$ through the barycentric weights $w_i = \frac{1}{3} - \frac{2}{3}(\mathbf{p}_i - \mathbf{p}_c)^T\mathbf{e}/R^2$, whose magnitudes sum to at most $1 + \sqrt{2}\|\mathbf{e}\|/R$. Adding the linearization's own error $M\|\mathbf{e}\|^2/\sqrt{2}$ at $\mathbf{p}^*$ and merging the curvature terms gives
```

## S-IIB5
Shrink. Trade-off paragraph tightened; the collinearity sentence (which the equilateral formula cannot reach) dropped

OLD:
```latex
Formation size moves the two terms in opposite directions. The curvature term grows with $(R + \|\mathbf{e}\|)^2$, the square of the largest distance from a robot to $\mathbf{p}^*$, so a large formation raises it. The measurement term grows with $\|\mathbf{e}\|/R$, the distance the fit extrapolates measured in formation sizes, so a small formation raises it. This extrapolation grows without bound as the robots approach collinearity. The bound measures how far the estimate sits from a true critical point. It assumes one exists nearby and does not establish that it does.
```
NEW:
```latex
Formation size moves the two terms in opposite directions. A large formation raises the curvature term, which grows with the square of the largest distance from a robot to $\mathbf{p}^*$. A small formation raises the measurement term, which grows with the extrapolation distance $\|\mathbf{e}\|/R$ measured in formation sizes. The bound assumes a true critical point exists nearby and does not establish that it does.
```

## S-IIC
Shrink. Proof prose tightened, same construction; "Suppose" replaced by choosing a field with p* off L (one counterexample proves necessity)

OLD:
```latex
Any configuration of $n \le 2$ robots, or any number of collinear robots, lies on a line $L = \{\mathbf{p}_0 + s\mathbf{d}\}$, with robot $i$ at $s = s_i$. The robots measure $\mathbf{m}_0 + s_i\mathbf{m}_1$, where $\mathbf{m}_0 = \mathbf{J}(\mathbf{p}_0 - \mathbf{p}^*)$ and $\mathbf{m}_1 = \mathbf{J}\mathbf{d}$. Suppose $\mathbf{p}^* \notin L$, and choose any $\mathbf{q} \notin L$. The alternative field $\mathbf{J}_q(\mathbf{p} - \mathbf{q})$ with $\mathbf{J}_q = [\,\mathbf{m}_0\ \ \mathbf{m}_1\,][\,\mathbf{p}_0 - \mathbf{q}\ \ \mathbf{d}\,]^{-1}$ is nonsingular and reproduces every measurement on $L$. Because $\mathbf{q}$ is arbitrary, the measurements cannot determine $\mathbf{p}^*$. Furthermore, $\det(\mathbf{J}_q) = \det(\mathbf{J})\det[\,\mathbf{p}_0 - \mathbf{p}^*\ \ \mathbf{d}\,]/\det[\,\mathbf{p}_0 - \mathbf{q}\ \ \mathbf{d}\,]$ changes sign as $\mathbf{q}$ crosses $L$, so the type of the critical point is also undetermined. Conversely, for three non-collinear robots $\mathbf{A}$ is invertible, so $\hat{\mathbf{J}} = \mathbf{J}$ and $\hat{\mathbf{p}}^* = \mathbf{p}^*$. The same holds for least squares with $n > 3$ robots that are not all collinear.
```
NEW:
```latex
Any $n \le 2$ robots, or any collinear robots, lie on a line $L = \{\mathbf{p}_0 + s\mathbf{d}\}$ and measure $\mathbf{m}_0 + s_i\mathbf{m}_1$, with $\mathbf{m}_0 = \mathbf{J}(\mathbf{p}_0 - \mathbf{p}^*)$ and $\mathbf{m}_1 = \mathbf{J}\mathbf{d}$. Take a field with $\mathbf{p}^* \notin L$. For any $\mathbf{q} \notin L$, the field $\mathbf{J}_q(\mathbf{p} - \mathbf{q})$ with $\mathbf{J}_q = [\,\mathbf{m}_0\ \ \mathbf{m}_1\,][\,\mathbf{p}_0 - \mathbf{q}\ \ \mathbf{d}\,]^{-1}$ is nonsingular and reproduces every measurement on $L$, so the measurements cannot determine $\mathbf{p}^*$. Its determinant $\det(\mathbf{J}_q) = \det(\mathbf{J})\det[\,\mathbf{p}_0 - \mathbf{p}^*\ \ \mathbf{d}\,]/\det[\,\mathbf{p}_0 - \mathbf{q}\ \ \mathbf{d}\,]$ changes sign as $\mathbf{q}$ crosses $L$, so the type is also undetermined. Conversely, for three non-collinear robots $\mathbf{A}$ is invertible, so $\hat{\mathbf{J}} = \mathbf{J}$ and $\hat{\mathbf{p}}^* = \mathbf{p}^*$, and the same holds for least squares with $n > 3$ robots that are not all collinear.
```

## S-III1
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
also samples the local field $\mathbf{v}(\mathbf{p}_i)
```
NEW:
```latex
also samples the local vector field $\mathbf{v}(\mathbf{p}_i)
```

## S-III2
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
inputs to the estimation framework of Section~II.
```
NEW:
```latex
inputs to the estimation framework described in Section~II.
```

## S-III3
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
vector field measurements (Section~II-B), a navigation controller
```
NEW:
```latex
vector field measurements as described in Section~II-B, a navigation controller
```

## S-III4
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
using the control laws of Section~II-D, and a formation
```
NEW:
```latex
using the control laws described in Section~II-D, and a formation
```

## S-III5
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
three-layer control architecture of Section~III was developed
```
NEW:
```latex
three-layer control architecture described in Section~III was developed
```

## S-IIIB
Shrink and survival. Submitted III-B sentences restored, attribution to [50] kept

OLD:
```latex
A cluster space controller layer is used to execute centroid velocity commands from the adaptive navigation layer while maintaining formation geometry. The cluster space controller regulates the three-robot triangular formation through a Side-Angle-Side (SAS) parameterization $(p, \beta, q)$, where $p$ is the distance between robots 1 and 2, $\beta$ is the interior angle at robot 2, and $q$ is the distance between robots 2 and 3 \cite{50}. Following the derivation in \cite{50}, differentiating the forward and inverse kinematic mappings between robot coordinates and these cluster variables gives analytic forms for the cluster space Jacobian and its inverse. The inverse Jacobian translates cluster velocity commands into robot velocity commands, and the forward kinematics transform measured robot positions into cluster variables for shape control. The forward kinematics and the cluster space Jacobian close the feedback loop.
```
NEW:
```latex
The cluster space controller maintains the cluster's formation geometry while executing centroid velocity commands from the adaptive navigation layer. It translates cluster-level velocity commands into individual robot velocity commands using an inverse kinematic Jacobian, and transforms robot position data back into cluster variables for feedback control. In this study, we consider a cluster of three robots forming a triangular configuration. The shape can be parameterized using Side-Angle-Side (SAS) representation with parameters $(p, \beta, q)$, where $p$ is the distance between robots 1 and 2, $\beta$ is the interior angle at robot 2, and $q$ is the distance between robots 2 and 3. Differentiating the forward and inverse kinematic equations gives analytic forms for the cluster space kinematic Jacobian and its inverse \cite{50}.
```

## S-IVm1
Shrink. itemize replaced by run-in paragraphs

OLD:
```latex
\begin{itemize}\setlength{\itemsep}{0pt}\setlength{\topsep}{2pt}\setlength{\parsep}{0pt}
```
NEW:
```latex
(deleted)
```

## S-IVm2
Shrink. Attraction metrics condensed; definitions unchanged

OLD:
```latex
\item \emph{Attraction.} A trial ends at time $T$ and succeeds when the critical point lies inside the triangle of the three robots and the centroid never left the map. Its final error is the distance $e_f = \|\mathbf{p}_c(T) - \mathbf{p}^*\|$. The tables give the mean $\bar{e}_f$ and standard deviation $s_e$ of $e_f$ over trials, its median and 95th percentile (p95), and the share of trials with $e_f \le \varepsilon$ for $\varepsilon = 0.03$, $0.04$, and $0.05$~m. Averaging the final positions rather than their distances separates a systematic offset from scatter. The bias is the distance of the mean final position from $\mathbf{p}^*$, and the precision $\sigma$ is the RMS distance of the final positions from that mean, so that bias$^2 + \sigma^2$ is the mean of $e_f^2$. The convergence time $T_{\mathrm{conv}}$ is the median time after which the centroid stays within 0.05~m. Localization error is the mean of $\|\hat{\mathbf{p}}^* - \mathbf{p}^*\|$ over the last 20\% of a trial, and control effort is $E_u = \sum_j \|\mathbf{p}_c[j+1] - \mathbf{p}_c[j]\|^2/\Delta t$. In a field with several critical points, a trial can succeed at a critical point other than the target $\mathbf{p}^*$. Success there is also scored against the critical point nearest the final centroid, and the metrics above use only the trials that succeeded at $\mathbf{p}^*$.
```
NEW:
```latex

\emph{Attraction.} A trial ends at time $T$ and succeeds when the critical point lies inside the triangle of the three robots and the centroid never left the map. Its final error $e_f = \|\mathbf{p}_c(T) - \mathbf{p}^*\|$ is reported by its mean $\bar{e}_f$, standard deviation $s_e$, median, 95th percentile (p95), and the share of trials with $e_f$ within 0.03, 0.04, and 0.05~m. The bias is the distance of the mean final position from $\mathbf{p}^*$, and the precision $\sigma$ is the RMS distance of the final positions from that mean, so that bias$^2 + \sigma^2$ is the mean of $e_f^2$. The convergence time $T_{\mathrm{conv}}$ is the median time after which the centroid stays within 0.05~m. Localization error is the mean of $\|\hat{\mathbf{p}}^* - \mathbf{p}^*\|$ over the last 20\% of a trial, and control effort is $E_u = \sum_j \|\mathbf{p}_c[j+1] - \mathbf{p}_c[j]\|^2/\Delta t$. In a field with several critical points, success is also scored at the critical point nearest the final centroid, and the other metrics use only the trials that succeeded at the target $\mathbf{p}^*$.

```

## S-IVm3
Shrink and Fix 5 (orbit RMSE defined). Triangle-based orbit success rule and the orphan (Ctrl.) label dropped; controller-only bias is now defined in VI-A where it is used

OLD:
```latex
\item \emph{Orbit.} An orbit run is scored on its radial error $\|\mathbf{p}_c - \mathbf{p}^*\| - r_d$ after the first 10~s. Its mean is the radial bias and its standard deviation the within-run spread, and the standard deviation of run means across repeats is the between-trial spread. The controller-only bias (Ctrl.) measures the radius about $\hat{\mathbf{p}}^*$ instead, the radius $r$ that (\ref{eq:11}) regulates, so its gap from the radial bias is the part due to estimation. A run succeeds when the triangle touches the commanded circle of radius $r_d$ about the critical point on at least 95\% of the scored samples and the centroid never left the map. The triangle sets the tolerance, between its inradius of 0.095~m and its circumradius of 0.19~m at the 0.33~m side. A negative bias means the cluster orbited inside $r_d$. A radius is practically achievable when the magnitude of its radial bias is at most $0.25\,r_d$. A run is lost if the centroid leaves the map or turns less than a quarter revolution about $\mathbf{p}^*$ after the first 10~s. The statistics use the runs that were not lost, and a radius at which most runs were lost is not scored.
```
NEW:
```latex
\emph{Orbit.} An orbit run is scored on its radial error $\|\mathbf{p}_c - \mathbf{p}^*\| - r_d$ after the first 10~s. Its mean is the radial bias, its RMS value the orbit RMSE, and its standard deviation the within-run spread. The standard deviation of run means across repeats is the between-trial spread. A radius is practically achievable when the magnitude of its radial bias is at most $0.25\,r_d$. A run is lost if the centroid leaves the map or turns less than a quarter revolution about $\mathbf{p}^*$ after the first 10~s, and the statistics use the runs that were not lost.

```

## S-IVm4
Shrink. Run-in paragraph

OLD:
```latex
\item \emph{Classification.} Classification accuracy is the fraction of cycles on which the type of $\hat{\mathbf{J}}$ matches the type of the critical point the trial reached.
```
NEW:
```latex
\emph{Classification.} Classification accuracy is the fraction of cycles on which the type of $\hat{\mathbf{J}}$ matches the type of the critical point the trial reached.
```

## S-IVm5
Shrink. itemize removed

OLD:
```latex
\end{itemize}
```
NEW:
```latex
(deleted)
```

## S-IV1
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
the final centroid position in each trial was used
```
NEW:
```latex
the final centroid position of the cluster in each trial was used
```

## S-IVB
Shrink. Triangle-rule orbit sentences cut with the rule itself

OLD:
```latex
 By the triangle rule, no orbit succeeds at $r_d = 0.01$~m and every analytical orbit succeeds from 0.1~m up. The reconstructed saddle succeeds in \rev{43} to \rev{100}\% of runs from 0.1~m up, and the reconstructed vortex in \rev{0}\% at 0.1 and 0.2~m rising to \rev{100}\% at 0.5~m. % [REVIEW] revision_metrics/tables/table_III_revised.tex; triangle rule: revision_metrics/results/triangle_*.csv
```
NEW:
```latex
 % [REVIEW] revision_metrics/tables/table_III_revised.tex
```

## S-IVC
Shrink. Coarse-class clause cut

OLD:
```latex
the star nodes on \rev{95}\%, or \rev{100}\% in coarse classes that merge nodes and spirals of the same stability, against
```
NEW:
```latex
the star nodes on \rev{95}\%, against
```

## S-V1
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
OptiTrack positions and RGB readings, computes
```
NEW:
```latex
OptiTrack positions and RGB sensor readings, computes
```

## S-V2
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
minimally affected by overhead lighting.
```
NEW:
```latex
minimally affected by overhead lighting variations.
```

## S-V3
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
interpolation models, used in Section~IV as a fixed
```
NEW:
```latex
interpolation models, referenced in Section~IV as a fixed
```

## S-V4
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
previous studies in our group~
```
NEW:
```latex
previous studies in our research group~
```

## S-V5
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
the only canonical field type with separatrices
```
NEW:
```latex
the only canonical field type that contains separatrices
```

## S-V6
Survival. Cosmetic edit reverted to submitted wording

OLD:
```latex
In the vortex, systematic bias was
```
NEW:
```latex
In the vortex field, systematic bias was
```

## S-VIA1
Survival. Cosmetic edit reverted

OLD:
```latex
with its errors propagating directly
```
NEW:
```latex
with sensing errors propagating directly
```

## S-VIA2
Controller-only bias defined where it is used (label dropped from IV)

OLD:
```latex
The hardware orbit metrics separate estimation error from control error. At $r_d = 0.01$~m the controller-only bias in the vortex was \syn{0.177}~m against a true bias of \syn{0.166}~m, so the small-radius offset arises in the closed loop. At larger radii the localization error grows with distance from the critical point, and the within-run spread grows with it. % [SYNTHETIC] V_VIA numbers.tex
```
NEW:
```latex
The hardware orbit metrics separate estimation error from control error. The controller-only bias measures the radius about $\hat{\mathbf{p}}^*$, which (\ref{eq:11}) regulates, instead of about $\mathbf{p}^*$. At $r_d = 0.01$~m it was \syn{0.177}~m in the vortex against a true bias of \syn{0.166}~m, so the small-radius offset arises in the closed loop. At larger radii the localization error grows with distance from the critical point, and the within-run spread grows with it. % [SYNTHETIC] V_VIA numbers.tex
```

## S-VIB1
Shrink. Variant description tightened

OLD:
```latex
The vector-to-scalar primitive \cite{48} reduces the flow to its magnitude, the speed $S_i = \|\mathbf{v}(\mathbf{p}_i)\|$ at each robot, and a plane fit to the three speeds gives their gradient $\mathbf{g}$. For attraction the cluster descends $\mathbf{g}$. For orbits the primitive was modified to move tangentially to the direction of steepest descent. It was run in simulation with the robot model, gains, and formation of Section~IV in three forms (Table~\ref{tab:baselines}). The first is the primitive alone. The second adds the oracle, the true critical point, as the target of (\ref{eq:10}) and the center of the radial term of (\ref{eq:11}), which isolates the primitive from any estimate. The third uses only the sensed speeds. It places the critical point where the fitted speed reaches zero, $\hat{\mathbf{p}}_m = \mathbf{p}_c - S_c\mathbf{g}/\|\mathbf{g}\|^2$ with $S_c$ the fitted speed at the centroid, and feeds that point to (\ref{eq:10}) and (\ref{eq:11}).
```
NEW:
```latex
The vector-to-scalar primitive \cite{48} reduces the flow to its magnitude, the speed $S_i = \|\mathbf{v}(\mathbf{p}_i)\|$ at each robot, and a plane fit to the three speeds gives their gradient $\mathbf{g}$. For attraction the cluster descends $\mathbf{g}$. For orbits the primitive was modified to move tangentially to the direction of steepest descent. It was run in simulation with the robot model, gains, and formation of Section~IV in three forms (Table~\ref{tab:baselines}). The first is the primitive alone. The second, the oracle, takes the true critical point as the target of (\ref{eq:10}) and the center of the radial term of (\ref{eq:11}). The third places the critical point where the fitted speed reaches zero, $\hat{\mathbf{p}}_m = \mathbf{p}_c - S_c\mathbf{g}/\|\mathbf{g}\|^2$ with $S_c$ the fitted speed at the centroid, and feeds it to (\ref{eq:10}) and (\ref{eq:11}).
```

## S-VIB2
Shrink. Numbers that Table IV already gives dropped from the prose

OLD:
```latex
On the six analytical fields the speed grows equally in every direction, so its gradient points directly away from $\mathbf{p}^*$, and the primitive reached $\mathbf{p}^*$ in every trial. Its dynamics differ from those of the Jacobian estimate (Fig.~\ref{fig:effort}(a)). The fitted gradient keeps the same length at every distance, so the command holds the robots at their speed limit until the cluster passes $\mathbf{p}^*$, overshoots by a median of \newnum{0.031}~m, and rebounds. The Jacobian estimate scales the command with distance, and the cluster slows along a smooth exponential. The primitive ends closer to $\mathbf{p}^*$, \newnum{0.003} against \rev{0.025}~m, because friction stops a proportional command on the ring $v_c/k$, and it settles in \newnum{1.1} against \rev{2.1}~s at \newnum{1.9} times the control effort. % [REVIEW] baselines_v2s_summary.json linear: v2s overshoot_med 0.031, e_mean 0.003, t_conv 1.1, E_u 0.137 against 0.074
```
NEW:
```latex
On the six analytical fields the speed grows equally in every direction, so its gradient points directly away from $\mathbf{p}^*$, and the primitive reached $\mathbf{p}^*$ in every trial (Fig.~\ref{fig:effort}(a)). The fitted gradient keeps the same length at every distance, so the command holds the robots at their speed limit until the cluster passes $\mathbf{p}^*$, overshoots by a median of \newnum{0.031}~m, and rebounds. The Jacobian estimate scales the command with distance, and the cluster slows along a smooth exponential. The primitive settles faster and closer to $\mathbf{p}^*$, because friction stops a proportional command on the ring $v_c/k$, but at \newnum{1.9} times the control effort. % [REVIEW] baselines_v2s_summary.json linear: v2s overshoot_med 0.031, e_mean 0.003, t_conv 1.1, E_u 0.137 against 0.074
```

## S-VIB3
Shrink. Orbit paragraph tightened; lag-drift reference now points to III-D

OLD:
```latex
In orbit, the primitive alone has no mechanism to detect or correct radial error with respect to a specific point, and so centrifugal drift goes uncorrected. Every orbit spiraled outward off the map (Fig.~\ref{fig:effort}(b)). Given the oracle, the same tangent with the radial term of (\ref{eq:11}) matched the Jacobian method within \newnum{0.002}~m RMSE at every radius. The orbit law therefore holds the radius about whatever point it is given, and the Jacobian estimate supplies that point from the same three readings the primitive uses. With its own speed estimate the primitive also held the radius on these fields, at a lower RMSE than the Jacobian method, because the estimate's inward bias offsets the lag drift of Section~VI-A. That estimate never lets the cluster rest during attraction, however. The cluster still moved at \newnum{0.12}~m/s after 10~s, with \newnum{four} times the control effort. % [REVIEW] baselines_v2s_summary.json linear: v2s lost 100% at every r_d (all left the map); v2s_oracle RMSE 0.092/0.052/0.035 against 0.093/0.051/0.033; v2s_est RMSE 0.068/0.016/0.020, final_speed 0.116, E_u 0.306
```
NEW:
```latex
In orbit, the primitive alone has no mechanism to detect or correct radial error with respect to a specific point, and so centrifugal drift goes uncorrected. Every orbit spiraled outward off the map (Fig.~\ref{fig:effort}(b)). Given the oracle, the same tangent with the radial term of (\ref{eq:11}) matched the Jacobian method within \newnum{0.002}~m RMSE at every radius. The orbit law therefore holds the radius about whatever point it is given, and the Jacobian estimate supplies that point from the same three readings the primitive uses. With its own speed estimate the primitive held the radius at a lower RMSE, because the estimate's inward bias offsets the lag drift of Section~III-D. That estimate never lets the cluster rest during attraction, however, and the cluster still moved at \newnum{0.12}~m/s after 10~s. % [REVIEW] baselines_v2s_summary.json linear: v2s lost 100% at every r_d (all left the map); v2s_oracle RMSE 0.092/0.052/0.035 against 0.093/0.051/0.033; v2s_est RMSE 0.068/0.016/0.020, final_speed 0.116, E_u 0.306
```

## S-VIB4
Shrink. Anisotropic node and gyre paragraph tightened; "the speed points" corrected to the speed gradient

OLD:
```latex
The speed points at the critical point only where it grows equally in every direction. The anisotropic node $\mathbf{J} = \operatorname{diag}(-1, -0.2)$ is affine, but its speed grows five times more slowly along one axis. There the primitive reached $\mathbf{p}^*$ in \newnum{7}\% of trials, and most of the rest ended on the slow axis, a median \newnum{0.67}~m from $\mathbf{p}^*$. Its speed estimate erred by \newnum{0.46}~m on average during orbits, and even with the oracle most orbits were lost, because the speed contours are ellipses. The Jacobian estimate is exact on any affine field, and on the node it reproduced its results on the analytical fields. In the gyre, the primitive reached the same critical point as the Jacobian method in \newnum{96}\% of trials. The speed vanishes at the center and at the saddles alike, however, so it cannot tell which it has reached, while the Jacobian method labeled the type on every cycle (Section~IV-C). The primitive lost every gyre orbit, and with the oracle every orbit from $r_d = 0.3$~m, where the gyre's speed contours no longer circle the center. The Jacobian method held its orbits at 0.1 and 0.3~m. % [REVIEW] baselines_v2s_summary.json node_aniso: v2s success 6.6% (467 failures, median |y| 0.67 m, |x| 0.03 m), v2s_est est_orb 0.455, v2s_oracle lost 90/82/100%; gyre: same point as jacobian 96.4%, jacobian type_acc 100%, v2s lost 100%, v2s_oracle lost 0/100/100%, jacobian lost 0/0/100%
```
NEW:
```latex
The speed gradient points away from the critical point only where the speed grows equally in every direction. The anisotropic node $\mathbf{J} = \operatorname{diag}(-1, -0.2)$ is affine, but its speed grows five times more slowly along one axis. There the primitive reached $\mathbf{p}^*$ in \newnum{7}\% of trials, and most of the rest ended on the slow axis, a median \newnum{0.67}~m from $\mathbf{p}^*$. Its speed estimate erred by \newnum{0.46}~m on average during orbits, and even with the oracle most orbits were lost, because the speed contours are ellipses. The Jacobian estimate is exact on any affine field, including the node. In the gyre, the primitive reached the same critical point as the Jacobian method in \newnum{96}\% of trials. The speed vanishes at the center and at the saddles alike, however, so it cannot tell which it has reached, while the Jacobian method labeled the type on every cycle (Section~IV-C). The primitive lost every gyre orbit, and with the oracle every orbit from $r_d = 0.3$~m, where the gyre's speed contours no longer circle the center. % [REVIEW] baselines_v2s_summary.json node_aniso: v2s success 6.6% (467 failures, median |y| 0.67 m, |x| 0.03 m), v2s_est est_orb 0.455, v2s_oracle lost 90/82/100%; gyre: same point as jacobian 96.4%, jacobian type_acc 100%, v2s lost 100%, v2s_oracle lost 0/100/100%, jacobian lost 0/0/100%
```

## S-VIC1
Survival. Submitted First paragraph restored

OLD:
```latex
The estimator fits a planar affine model across the formation, which limits the fields it can address. It applies only to planar vector fields, and a 3D field would require at least four non-coplanar robots for Jacobian estimation. It also assumes the affine model holds from the cluster out to the critical point (Section~II-B). On fields that are highly nonlinear at the scale of the formation, the fit degrades and biases the recovered critical point location. Section~IV probes this regime on a gyre cell and a curved saddle, but fields with stronger curvature, or with critical points closer together than one formation side, were not tested.
```
NEW:
```latex
Several limitations constrain the current implementation and experimental validation. First, the approach is restricted to planar (2D) vector fields, though many real-world applications involve three-dimensional flows. Extension to 3D fields would require at minimum four non-coplanar robots for Jacobian estimation.

Second, the testbed validates performance only on static fields where critical points remain fixed during operation. The static printed field approach enables precise ground truth validation but does not capture dynamic field behavior, and the printed fields do not push the robots. A physical flow advects the cluster and adds its own velocity $\mathbf{J}\mathbf{e}$ to the centroid near $\mathbf{p}^*$. For an ideal centroid with an exact estimate, the error then obeys $\dot{\mathbf{e}} = (\mathbf{J} - k\mathbf{I})\mathbf{e}$, which is unstable when an eigenvalue of $\mathbf{J}$ has real part above $k$. A physical critical point can also drift, and the attraction law trails one moving at constant speed by that speed divided by $k$.

Third, robot team sizes were limited to three. While the theoretical framework supports arbitrary team sizes through the overdetermined system approach, larger formations present practical implementation challenges including communication bandwidth, computational complexity, and formation control stability. With $n > 3$ robots the least-squares fit averages measurement errors across robots and survives the loss of one. However, a wider footprint asks the affine model to hold over more of the field's curvature.

Fourth, the estimator assumes the field is well approximated by an affine model from the cluster out to the critical point (Section~II-B). On fields that are highly nonlinear at the scale of the formation, this local linear fit degrades, biasing the recovered critical point location. Section~IV probes this regime on a gyre cell and a curved saddle, but fields with stronger curvature, or with critical points closer together than one formation side, were not tested.
```

## S-VIC2
Folded into Third

OLD:
```latex
All experiments used three robots, the minimum. With $n > 3$ robots the least-squares fit averages measurement errors across robots and survives the loss of one. However, a wider footprint asks the affine model to hold over more of the field's curvature, and each added robot brings more communication load and two more shape variables for the formation controller to regulate.
```
NEW:
```latex
(deleted)
```

## S-VIC3
Fix 5. Folded into Second; unverified "Actuator lag lowers this limit in rotating flow" cut

OLD:
```latex
The printed fields give exact ground truth, but they are static and do not push the robots. A physical flow advects the cluster and adds its own velocity $\mathbf{J}\mathbf{e}$ to the centroid near $\mathbf{p}^*$. For an ideal centroid with an exact estimate, the error then obeys $\dot{\mathbf{e}} = (\mathbf{J} - k\mathbf{I})\mathbf{e}$, which is unstable when an eigenvalue of $\mathbf{J}$ has real part above $k$. Actuator lag lowers this limit in rotating flow. A physical critical point can also drift. Even without advection, the attraction law trails one moving at constant speed by that speed divided by $k$.
```
NEW:
```latex
(deleted)
```

## S-VIC4
Survival. Finally paragraph with the submitted closing sentence restored

OLD:
```latex
The computation is centralized. A central computer receives OptiTrack positions and field readings from every robot, forms the estimate, and returns velocity commands. The estimate needs only the positions and readings of one cycle in a common frame, so a robot receiving them from its neighbors could compute it. Onboard relative localization and inter-robot communication were not tested. In either architecture, every fit depends on the communication that delivers its readings. A dropout can leave fewer than three readings, and the fit then has no unique solution. Latency mixes readings from different times into one fit.
```
NEW:
```latex
Finally, the computation is centralized. A central computer receives OptiTrack positions and field readings from every robot, forms the estimate, and returns velocity commands. The estimate needs only the positions and readings of one cycle in a common frame, so a robot receiving them from its neighbors could compute it. Onboard relative localization and inter-robot communication were not tested. A dropout can leave fewer than three readings, and the fit then has no unique solution, while latency mixes readings from different times into one fit. The robustness of the approach to lower update rates and communication dropouts was not tested.
```

## S-VID1
Shrink and survival. Classification-band idea cut (nothing downstream uses it); submitted time-varying paragraph restored in short form

OLD:
```latex
The estimated Jacobian's eigenvalues determine the topological type of the nearby critical point (Table~\ref{tab:field_summary}), and Sections~IV-C and V-C evaluate this classification with a fixed band around the center and node boundaries. Replacing the band with a test against the covariance of $\hat{\mathbf{J}}$ under measurement noise would label fragile types with a stated confidence. Because the sign of $\det\hat{\mathbf{J}}$ on the first cycle decided the type of critical point reached (Section~IV-A), the same test could hold the cluster in place when the sensed type differs from the one the mission seeks.
```
NEW:
```latex
The estimation framework uses only instantaneous measurements and carries no state between control cycles. This memoryless property means the estimator does not assume field stationarity, suggesting applicability to environments where the field itself varies with time. Validating the estimator on a genuinely time-varying field remains future work.
```

## S-VID2
Survival. Submitted separatrix sentences restored; six-robot extension kept in one sentence

OLD:
```latex
Because the three-robot affine fit developed here returns a spatially constant Jacobian $\hat{\mathbf{J}}$, it cannot resolve how the velocity gradient varies across the formation. Extending the estimator to a second-order polynomial fit from six robots that do not lie on a common conic recovers a position-dependent Jacobian across the footprint. Work is being done to see how these new estimates can form controllers that can detect other topological features of practical interest beyond critical points.
```
NEW:
```latex
Beyond critical point detection and orbital control, vector fields contain other topological features of practical interest. Separatrices connected to saddle points define boundaries between distinct flow regimes and act as Lagrangian coherent structures in steady flows. The three-robot fit returns a spatially constant Jacobian, so detecting them calls for a second-order fit, which six robots that do not lie on a common conic can provide.
```

## S-fig1
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=\columnwidth]{six_vector_fields_3x2.png}
```
NEW:
```latex
\includegraphics[width=0.95\columnwidth]{six_vector_fields_3x2.png}
```

## S-fig2
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=\columnwidth]{control_architecture_2.png}
```
NEW:
```latex
\includegraphics[width=0.95\columnwidth]{control_architecture_2.png}
```

## S-fig3
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=\columnwidth]{fig_sim_convergence_band.pdf}
```
NEW:
```latex
\includegraphics[width=0.95\columnwidth]{fig_sim_convergence_band.pdf}
```

## S-fig4
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=\columnwidth]{fig_nonlinear_triangles.pdf}
```
NEW:
```latex
\includegraphics[width=0.95\columnwidth]{fig_nonlinear_triangles.pdf}
```

## S-fig5
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=0.62\columnwidth]{testbed_with_four.png}
```
NEW:
```latex
\includegraphics[width=0.59\columnwidth]{testbed_with_four.png}
```

## S-fig6
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=\columnwidth]{fig_reconstruction_3x2.png}
```
NEW:
```latex
\includegraphics[width=0.95\columnwidth]{fig_reconstruction_3x2.png}
```

## S-fig7
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\begin{minipage}[b]{0.47\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_a.png}
```
NEW:
```latex
\begin{minipage}[b]{0.45\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_a.png}
```

## S-fig8
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\begin{minipage}[b]{0.47\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_b.png}
```
NEW:
```latex
\begin{minipage}[b]{0.45\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_b.png}
```

## S-fig9
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=0.92\columnwidth]{placeholder_hw_distance.png}
```
NEW:
```latex
\includegraphics[width=0.87\columnwidth]{placeholder_hw_distance.png}
```

## S-fig10
Shrink. Figure width reduced 5% (uniform across all figures)

OLD:
```latex
\includegraphics[width=\columnwidth]{fig_v2s_effort.pdf}
```
NEW:
```latex
\includegraphics[width=0.95\columnwidth]{fig_v2s_effort.pdf}
```

## S-R2a
Shrink. Roadmap paragraph cut (style 6.4: no roadmap in papers); survival allows it

OLD:
```latex
The remainder of this paper is organized as follows. Section II develops the framework for Jacobian estimation from spatially distributed measurements and critical point prediction, and presents the control laws for approach and orbit. Section III describes the control architecture used to evaluate these strategies in simulation and hardware, and bounds the closed-loop error. Section IV presents simulation results across six vector field types and two nonlinear fields. Section V describes the hardware testbed and its results. Section VI compares the proposed estimate with the vector-to-scalar primitive and discusses limitations and future work. Section VII concludes.
```
NEW:
```latex
(deleted)
```

## S-R2b
Shrink. Counting paragraph cut; Proposition 1 and its proof cover n <= 2 and least squares

OLD:
```latex
With fewer than three robots, the system is underdetermined, because the matrix $\mathbf{A}$ in (\ref{eq:4}) has fewer rows than columns. This admits infinitely many solutions for the coefficients, preventing unique identification of the critical point location. With more than three robots ($n > 3$), the system is overdetermined and is solved by least squares.
```
NEW:
```latex
(deleted)
```

## S-R2c
Shrink, less math. Lyapunov display folded into one inline inequality; final-error sentence shortened

OLD:
```latex
and $V = \frac{1}{2}\|\mathbf{e}\|^2$ satisfies
\[
\dot{V} = -k\|\mathbf{e}\|^2 + k\,\mathbf{e}^T\boldsymbol{\delta} \le -k\|\mathbf{e}\|\bigl(\|\mathbf{e}\| - \|\boldsymbol{\delta}\|\bigr),
\]
so the error decreases
```
NEW:
```latex
and $V = \frac{1}{2}\|\mathbf{e}\|^2$ satisfies $\dot{V} \le -k\|\mathbf{e}\|(\|\mathbf{e}\| - \|\boldsymbol{\delta}\|)$, so the error decreases
```

## S-R2d
Shrink. Final-error sentence shortened

OLD:
```latex
It rests within $v_c/k$ of the estimate, and the estimate lies $\|\boldsymbol{\delta}_{\text{final}}\|$ from the critical point, so writing $\mathbf{e}_{\text{final}} = (\mathbf{p}_c - \hat{\mathbf{p}}^*) + \boldsymbol{\delta}_{\text{final}}$ gives $\|\mathbf{e}_{\text{final}}\| \le v_c/k + \|\boldsymbol{\delta}_{\text{final}}\|$.
```
NEW:
```latex
It rests within $v_c/k$ of the estimate, which lies $\|\boldsymbol{\delta}_{\text{final}}\|$ from the critical point, so $\|\mathbf{e}_{\text{final}}\| \le v_c/k + \|\boldsymbol{\delta}_{\text{final}}\|$.
```

## S-R2e
Shrink. Two classification sentences merged

OLD:
```latex
on every reading, the boundary types are labeled least reliably. The center is labeled correctly
```
NEW:
```latex
on every reading, the center is labeled correctly
```

## S-R2f
Shrink. Table II success rows merged into one, as in Table III

OLD:
```latex
Succ.\ 3~cm (\%) & 100 & 45 & 62 & \newnum{76} & \newnum{88} & \newnum{97} \\
Succ.\ 4~cm (\%) & 100 & 79 & 89 & \newnum{100} & \newnum{100} & \newnum{100} \\
Succ.\ 5~cm (\%) & 100 & 95 & 100 & \newnum{100} & \newnum{100} & \newnum{100} \\
```
NEW:
```latex
Succ.\ 3/4/5~cm (\%) & 100 & 45/79/95 & 62/89/100 & \newnum{76/100/100} & \newnum{88/100/100} & \newnum{97/100/100} \\
```

## S-R3a
Format. Table II padding 1.5pt -> 1.3pt so the merged success row fits (was 0.8pt overfull)

OLD:
```latex
\label{tab:simulation_results}
\setlength{\tabcolsep}{1.5pt}
```
NEW:
```latex
\label{tab:simulation_results}
\setlength{\tabcolsep}{1.3pt}
```

## S-R3b
Shrink. Forward pointer cut (style 3.3); V-A describes the reconstruction

OLD:
```latex
 Further details on the reconstructed fields are provided in Section~V.
```
NEW:
```latex

```

## S-R3c
Shrink. The two short future-work paragraphs merged into one

OLD:
```latex
time-varying field remains future work.

Beyond critical point detection
```
NEW:
```latex
time-varying field remains future work. Beyond critical point detection
```

