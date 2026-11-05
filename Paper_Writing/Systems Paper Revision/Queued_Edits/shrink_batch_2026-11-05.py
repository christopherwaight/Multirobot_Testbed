"""Shrink-and-fix batch for cwaight_systems_paper_with_revisions2.tex (2026-11-05).

Each edit is either
  ("sub",  id, why, old_substring, new)      old must occur exactly once
  ("line", id, why, line_prefix,   new)      the one line starting with prefix is replaced by new
                                             (new == None deletes the line)
Usage: python3 edits.py SRC OUT_TEX OUT_MD
"""
import sys

E = []
def sub(i, why, old, new): E.append(("sub", i, why, old, new))
def line(i, why, prefix, new): E.append(("line", i, why, prefix, new))

# ---------------------------------------------------------------- fixes
# F1 orbit closed form: derivation once, in III-D, with the formula the code uses
line("F1a", "Fix 1. Orbit closed form stated as the code computes it (theory.orbit_C); derivation moved here from VI-A",
"Lag and friction set the steady orbit radius even with an exact estimate.",
r"""Lag and friction set the steady orbit radius even with an exact estimate. Friction takes $v_c$ off the orbit speed, $V = k_t - v_c$, and passes only the fraction $V/k_t$ of the radial command. The lag makes the cluster follow the turning command a time $T$ late, so in a steady orbit of radius $r^*$ it drifts outward at $TV^2/r^*$. The radial term of (\ref{eq:11}) cancels this drift at
\begin{equation}\label{eq:orbit_cf}
r^* = \tfrac{1}{2}\bigl(r_d + \sqrt{r_d^2 + 4C}\bigr), \qquad C = TVk_t/k_r.
\end{equation}
In the sampled loop $T = \alpha'\Delta t/(1-\alpha') + \Delta t/2$, where $\alpha' = \alpha V/(\alpha V + (1-\alpha)k_t)$ is the lag pole after friction. Without sampling or friction, $T = \tau$ and $C = \tau V^2/k_r$. The error $r^* - r_d$ approaches the minimum radius $r_{\min} = \sqrt{C}$ as $r_d \to 0$ and falls as $C/r_d$ at large radii.""")

line("F1b", "Fix 1. VI-A keeps the submitted sentences and the numbers; the derivation now lives in III-D",
"Positive radial bias at small orbital radii is also a result of robot dynamics.",
r"""Positive radial bias at small orbital radii is also a result of robot dynamics. At small radii, the commanded angular rate $k_t / r_d$ is high relative to the radial correction bandwidth $k_r$, and the outward drift from the actuator lag sets the bias (\ref{eq:orbit_cf}). The closed form matches simulation to a median of \rev{0.03}~mm across sweeps of gains, lag, sample period, and friction. At the settings of Section~IV, $r_{\min} = \rev{0.133}$~m, and the error falls below 0.05~m only for $r_d \ge \rev{0.31}$~m. Lowering the orbital speed $k_t$ to 0.1~m/s cuts $r_{\min}$ to \rev{0.058}~m, and removing the lag cuts it to \rev{0.059}~m. Raising the radial gain $k_r$ to 2~s$^{-1}$ cuts it to \rev{0.067}~m, at the cost of increased sensitivity to noise in the critical point estimate. % [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3); C = T V k_t / k_r is theory.orbit_C""")
line("F1c", "Fix 1. Equation moved to III-D", r"\begin{equation}\label{eq:orbit_cf}", None)
line("F1d", "Fix 1. Equation moved to III-D", r"r^* = \tfrac{1}{2}\bigl(r_d + \sqrt{r_d^2 + 4C}\bigr), \qquad C = \tau V^2/k_r.", None)
line("F1e", "Fix 1. Equation moved to III-D (the end{equation} line right after it)", r"\end{equation}" + "\nThe error $r^* - r_d$", None)  # handled specially below
line("F1f", "Fix 1. Moved into F1b (numbers) and F1a (limits)", r"The error $r^* - r_d$ approaches the minimum radius $r_{\min} = \sqrt{C}$ as $r_d \to 0$", None)

# F2 r = 0 guard, "could" wording
sub("F2", "Fix 2. r = 0 guard phrased as what the controller could do",
r"When $r < 10^{-6}$~m the implementation sets $\mathbf{n}$ to an arbitrary unit vector, so the command stays defined and its tangential step $k_t\mathbf{t}$ carries the cluster off the singular point on the next cycle.",
r"When $r < 10^{-6}$~m, the controller could simply take a step in a random direction, which carries the cluster off the singular point on the next cycle.")

# F3 field count
sub("F3a", "Fix 3. Field count (Table II has three nonlinear columns)",
"across six canonical linear vector fields and two nonlinear fields using", "across six canonical linear vector fields and three nonlinear fields using")
sub("F3b", "Fix 3. Field count", "across six vector field types and two nonlinear fields. It also states", "across six vector field types and three nonlinear fields. It also states")
sub("F3d", "Fix 3. Field count", "Two nonlinear fields test how the estimate extrapolates.", "Three nonlinear fields test how the estimate extrapolates.")
sub("F3e", "Fix 3. Field count", "Simulations were therefore run on ten fields in total.", "Simulations were therefore run on eleven fields in total.")
sub("F3f", "Fix 3. Field count", "For each of the ten fields, the cluster", "For each of the eleven fields, the cluster")
sub("F3g", "Fix 3. Field count", r"\caption{Simulation results on ten fields.", r"\caption{Simulation results on eleven fields.")
sub("F3h", "Fix 3. Field count", "evaluated on the same ten fields at different", "evaluated on the same eleven fields at different")

# F4 conditioning paragraph
line("F4", "Fix 4. Amplification is 1/sigma_min(J) per (8), not kappa(J); edge-of-map bias comes from the extrapolation term of (8)",
r"Poor conditioning of the estimated Jacobian, measured by",
r"""Poor conditioning of the estimated Jacobian amplifies coefficient errors into errors in the critical point estimate (\ref{eq:9}) by the factor $1/\sigma_{\min}(\hat{\mathbf{J}})$ in (\ref{eq:bound}). Unlike the condition number of $\mathbf{A}$, it is a property of the field and is not under the designer's control. Where $\det\hat{\mathbf{J}}$ approaches zero, so does $\sigma_{\min}(\hat{\mathbf{J}})$, and Section~III-D describes how the controller handles it. A center is non-hyperbolic yet has $\det\mathbf{J} > 0$, so (\ref{eq:9}) locates it as well as any other critical point, and only its type is fragile (Section~IV-C).

The measurement term of (\ref{eq:bound}) grows with the extrapolation distance $\|\mathbf{e}\|/R$, and the reconstructed field degrades near the edge of the mapped region. Together they account for the orbital bias at the largest commanded radius falling below the controller-only bias on both reconstructed fields, to \rev{+0.026} against \rev{+0.035}~m on the vortex and \rev{+0.001} against \rev{+0.017}~m on the saddle. There the estimated distance to the critical point is inflated, and the radial controller pulls the cluster inward. % [REVIEW] revision_metrics/results/summary.json, reconstructed orbit r_d 0.50 (bias_w1, ctrl_bias_w1)""")

# F5 small items
sub("F5a", "Fix 5. Saturation (R1.2) addressed in one sentence",
r"leave the centroid error bounded.",
r"leave the centroid error bounded. The speed cap shortens the velocity without turning it, and it is inactive within $v_{\max}/k$ of the estimate, where this linear analysis applies.")
line("F5b", "Fix 5. Unverified tangential estimate-error paragraph cut (nothing used it)",
"To first order, an estimate error adds its radial component to the radial error.", None)
# F5c (cut "Actuator lag lowers this limit") and F5d (orbit RMSE) are inside S-VIC and S-IVm below.

# ---------------------------------------------------------------- shrink and survival
sub("S-ab1", "Survival. Submitted opening restored", "A variety of interesting phenomena exist at", "There are a variety of interesting phenomena that exist at")
sub("S-ab2", "Shrink and survival. Last sentence split and shortened, submitted opening words restored",
"While these findings establish the estimation method under lab conditions where printed fields do not advect the robots, the present controller does not yet compensate for the disturbance in oceanic and atmospheric flows where the field physically pushes the platform.",
"These findings establish this estimation method under lab conditions, where printed fields do not advect the robots. The present controller does not yet compensate for oceanic and atmospheric flows that push the platform.")

# Introduction, cosmetic reverts to submitted wording
for i, (o, n) in enumerate([
    ("systems enhance redundancy, throughput, coverage, and reconfigurability in", "systems enhance redundancy, throughput, coverage, and reconfigurability of robotic systems in"),
    ("environmental features \\cite{2,3,4}.", "environmental features across a region \\cite{2,3,4}."),
    ("collects a scalar measurement as input", "collects a single scalar measurement as the input"),
    ("and several have been field-deployed", "and several such systems have been field-deployed"),
    ("eddy acts as a center, trapping a body of water and the plankton", "eddy serves as a center, trapping a body of water, along with the plankton"),
    ("a fixed station would not", "a ground-fixed station would not"),
    ("These applications need the coordinate", "These applications require the coordinate"),
    ("drive the team onto", "drive the multirobot team onto"),
    ("reconstructed on hardware from", "reconstructed on physical hardware from"),
    ("The reconstruction's conditioning degrades", "The conditioning of this reconstruction degrades"),
    ("Both the estimator and the cluster's mechatronics limit", "Both the estimator and the mechatronics of the cluster limit"),
    ("are the minimum required for this estimation, and", "are the minimum number required for this estimation, and"),
]):
    sub(f"S-I{i+1}", "Survival. Cosmetic edit reverted to submitted wording", o, n)

line("S-I8", "Shrink and survival. Submitted paragraph restored without \"distributed\" and with \"indicate the type\" (round-1 R1.7 wording); centralized-computation sentence kept; nothing new",
"This paper introduces a cooperative Jacobian-inversion framework",
"This paper introduces a cooperative estimation strategy that recovers the local linear structure of the field. Using a minimum of three robots, the cluster estimates the local Jacobian in a single time step and treats critical point navigation as a local-to-global inference. Inverting the Jacobian predicts the location of the nearby critical point that organizes the surrounding flow, and its eigenvalues indicate the type, with no global map required. The same estimate supports two control laws, a proportional law that drives the cluster centroid onto the critical point, and an orbital law that holds the centroid at a commanded standoff radius. In the present implementation the robots sense and move, and a central computer forms the estimate and computes the commands.")

# II-A, II-B, II-C, II-D cosmetic reverts
for i, (o, n) in enumerate([
    ("to each point $\\mathbf{p} \\in \\mathbb{R}^n$ in a given domain", "to each point $\\mathbf{p} \\in \\mathbb{R}^n$ within a given domain"),
    ("For the planar fields of this work, this takes the form", "For the planar fields analyzed in this work, this takes the specific form"),
    ("determined by the Jacobian $\\mathbf{J} = ", "determined by the Jacobian matrix $\\mathbf{J} = "),
    ("The eigenvalues of $\\mathbf{J}$ at $\\mathbf{p}^*$", "The eigenvalues of $\\mathbf{J}$ evaluated at $\\mathbf{p}^*$"),
    ("degenerate scenarios such as a zero determinant.", "degenerate scenarios such as cases where the determinant is zero."),
    ("The coefficients are found by solving", "The coefficients are determined by solving"),
    ("are the collected vector field measurements.", "are the collected field measurements."),
    ("Two control primitives drive the cluster centroid", "We develop two control primitives that drive the cluster centroid"),
    ("The mapping from cluster velocity commands to robot velocities is handled by the cluster space controller of Section~III.",
     "The mapping from cluster-level velocity commands to individual robot velocities is handled by the cluster space controller described in Section~III."),
]):
    sub(f"S-II{i+1}", "Survival. Cosmetic edit reverted to submitted wording", o, n)

line("S-IIB1", "Shrink. Scope paragraph tightened; submitted sentences and math unchanged",
"Equation (\\ref{eq:9}) returns the zero of the fitted affine field",
r"""Equation (\ref{eq:9}) returns the zero of the fitted affine field $\hat{\mathbf{v}}(\mathbf{p}) = \hat{\mathbf{J}}\mathbf{p} + \hat{\mathbf{h}} = \hat{\mathbf{J}}(\mathbf{p} - \hat{\mathbf{p}}^*)$, which need not be a zero of the true field. Evaluating it at any critical point $\mathbf{p}^*$ of the true field gives""")
line("S-IIB2", "Shrink. Scope paragraph tightened",
"The true field vanishes at $\\mathbf{p}^*$, so the term",
r"""The true field vanishes at $\mathbf{p}^*$, so $\hat{\mathbf{v}}(\mathbf{p}^*)$ is the error of the affine model there. If the true field is affine over a region containing both the cluster and $\mathbf{p}^*$, the models (\ref{eq:2})--(\ref{eq:3}) reproduce it without error and the recovery is exact at any formation size. On a smooth nonlinear field, the fit carries a first-order linearization error that scales with the local curvature of the field; this error decreases as the formation contracts toward the critical point. The inference from $\hat{\mathbf{p}}^*$ to a true critical point is therefore valid only where the affine model remains accurate from the cluster out to $\mathbf{p}^*$.""")
line("S-IIB3", "Shrink. Correlation paragraph tightened, same content",
"Both $\\hat{\\mathbf{J}}$ and $\\hat{\\mathbf{h}}$ are computed from the same measurements",
r"""The errors in $\hat{\mathbf{J}}$ and $\hat{\mathbf{h}}$ come from the same measurements and are correlated, but (\ref{eq:scope}) carries them jointly through $\hat{\mathbf{v}}(\mathbf{p}^*)$. Taking norms gives $\|\hat{\mathbf{p}}^* - \mathbf{p}^*\| \le \|\hat{\mathbf{v}}(\mathbf{p}^*)\|/\sigma_{\min}(\hat{\mathbf{J}})$. The singular value $\sigma_{\min}(\hat{\mathbf{J}})$ is the slowest rate at which the fitted field changes in any direction, so a small value amplifies the model error.""")
line("S-IIB4", "Shrink. Bound derivation prose tightened; math unchanged",
"Let $\\mathbf{p}_c = \\frac{1}{3}\\sum_i\\mathbf{p}_i$ be the cluster centroid",
r"""Let $\mathbf{p}_c = \frac{1}{3}\sum_i\mathbf{p}_i$ be the cluster centroid, $\mathbf{e} = \mathbf{p}_c - \mathbf{p}^*$ its displacement from the critical point, and $R$ the distance from $\mathbf{p}_c$ to each robot of the equilateral formation. Let $M$ bound the Hessian norms of $u$ and $v$ over the region holding the robots and $\mathbf{p}^*$, and let $\bar{\varepsilon}$ bound each robot's measurement error. Each reading then lies within $MR^2/\sqrt{2} + \bar{\varepsilon}$ of the linearization about $\mathbf{p}_c$. The fit carries these deviations to $\mathbf{p}^*$ through the barycentric weights $w_i = \frac{1}{3} - \frac{2}{3}(\mathbf{p}_i - \mathbf{p}_c)^T\mathbf{e}/R^2$, whose magnitudes sum to at most $1 + \sqrt{2}\|\mathbf{e}\|/R$. Adding the linearization's own error $M\|\mathbf{e}\|^2/\sqrt{2}$ at $\mathbf{p}^*$ and merging the curvature terms gives""")
line("S-IIB5", "Shrink. Trade-off paragraph tightened; the collinearity sentence (which the equilateral formula cannot reach) dropped",
"Formation size moves the two terms in opposite directions.",
r"""Formation size moves the two terms in opposite directions. A large formation raises the curvature term, which grows with the square of the largest distance from a robot to $\mathbf{p}^*$. A small formation raises the measurement term, which grows with the extrapolation distance $\|\mathbf{e}\|/R$ measured in formation sizes. The bound assumes a true critical point exists nearby and does not establish that it does.""")
line("S-IIC", "Shrink. Proof prose tightened, same construction; \"Suppose\" replaced by choosing a field with p* off L (one counterexample proves necessity)",
"Any configuration of $n \\le 2$ robots, or any number of collinear robots",
r"""Any $n \le 2$ robots, or any collinear robots, lie on a line $L = \{\mathbf{p}_0 + s\mathbf{d}\}$ and measure $\mathbf{m}_0 + s_i\mathbf{m}_1$, with $\mathbf{m}_0 = \mathbf{J}(\mathbf{p}_0 - \mathbf{p}^*)$ and $\mathbf{m}_1 = \mathbf{J}\mathbf{d}$. Take a field with $\mathbf{p}^* \notin L$. For any $\mathbf{q} \notin L$, the field $\mathbf{J}_q(\mathbf{p} - \mathbf{q})$ with $\mathbf{J}_q = [\,\mathbf{m}_0\ \ \mathbf{m}_1\,][\,\mathbf{p}_0 - \mathbf{q}\ \ \mathbf{d}\,]^{-1}$ is nonsingular and reproduces every measurement on $L$, so the measurements cannot determine $\mathbf{p}^*$. Its determinant $\det(\mathbf{J}_q) = \det(\mathbf{J})\det[\,\mathbf{p}_0 - \mathbf{p}^*\ \ \mathbf{d}\,]/\det[\,\mathbf{p}_0 - \mathbf{q}\ \ \mathbf{d}\,]$ changes sign as $\mathbf{q}$ crosses $L$, so the type is also undetermined. Conversely, for three non-collinear robots $\mathbf{A}$ is invertible, so $\hat{\mathbf{J}} = \mathbf{J}$ and $\hat{\mathbf{p}}^* = \mathbf{p}^*$, and the same holds for least squares with $n > 3$ robots that are not all collinear.""")

# III cosmetic reverts and III-B restore
for i, (o, n) in enumerate([
    ("also samples the local field $\\mathbf{v}(\\mathbf{p}_i)", "also samples the local vector field $\\mathbf{v}(\\mathbf{p}_i)"),
    ("inputs to the estimation framework of Section~II.", "inputs to the estimation framework described in Section~II."),
    ("vector field measurements (Section~II-B), a navigation controller", "vector field measurements as described in Section~II-B, a navigation controller"),
    ("using the control laws of Section~II-D, and a formation", "using the control laws described in Section~II-D, and a formation"),
    ("three-layer control architecture of Section~III was developed", "three-layer control architecture described in Section~III was developed"),
]):
    sub(f"S-III{i+1}", "Survival. Cosmetic edit reverted to submitted wording", o, n)
line("S-IIIB", "Shrink and survival. Submitted III-B sentences restored, attribution to [50] kept",
"A cluster space controller layer is used to execute centroid velocity commands",
r"""The cluster space controller maintains the cluster's formation geometry while executing centroid velocity commands from the adaptive navigation layer. It translates cluster-level velocity commands into individual robot velocity commands using an inverse kinematic Jacobian, and transforms robot position data back into cluster variables for feedback control. In this study, we consider a cluster of three robots forming a triangular configuration. The shape can be parameterized using Side-Angle-Side (SAS) representation with parameters $(p, \beta, q)$, where $p$ is the distance between robots 1 and 2, $\beta$ is the interior angle at robot 2, and $q$ is the distance between robots 2 and 3. Differentiating the forward and inverse kinematic equations gives analytic forms for the cluster space kinematic Jacobian and its inverse \cite{50}.""")

# IV metric definitions condensed (F5d orbit RMSE defined here; triangle orbit rule and orphan Ctrl label dropped)
line("S-IVm1", "Shrink. itemize replaced by run-in paragraphs", r"\begin{itemize}\setlength{\itemsep}{0pt}", None)
line("S-IVm2", "Shrink. Attraction metrics condensed; definitions unchanged",
r"\item \emph{Attraction.}",
r"""
\emph{Attraction.} A trial ends at time $T$ and succeeds when the critical point lies inside the triangle of the three robots and the centroid never left the map. Its final error $e_f = \|\mathbf{p}_c(T) - \mathbf{p}^*\|$ is reported by its mean $\bar{e}_f$, standard deviation $s_e$, median, 95th percentile (p95), and the share of trials with $e_f$ within 0.03, 0.04, and 0.05~m. The bias is the distance of the mean final position from $\mathbf{p}^*$, and the precision $\sigma$ is the RMS distance of the final positions from that mean, so that bias$^2 + \sigma^2$ is the mean of $e_f^2$. The convergence time $T_{\mathrm{conv}}$ is the median time after which the centroid stays within 0.05~m. Localization error is the mean of $\|\hat{\mathbf{p}}^* - \mathbf{p}^*\|$ over the last 20\% of a trial, and control effort is $E_u = \sum_j \|\mathbf{p}_c[j+1] - \mathbf{p}_c[j]\|^2/\Delta t$. In a field with several critical points, success is also scored at the critical point nearest the final centroid, and the other metrics use only the trials that succeeded at the target $\mathbf{p}^*$.
""")
line("S-IVm3", "Shrink and Fix 5 (orbit RMSE defined). Triangle-based orbit success rule and the orphan (Ctrl.) label dropped; controller-only bias is now defined in VI-A where it is used",
r"\item \emph{Orbit.}",
r"""\emph{Orbit.} An orbit run is scored on its radial error $\|\mathbf{p}_c - \mathbf{p}^*\| - r_d$ after the first 10~s. Its mean is the radial bias, its RMS value the orbit RMSE, and its standard deviation the within-run spread. The standard deviation of run means across repeats is the between-trial spread. A radius is practically achievable when the magnitude of its radial bias is at most $0.25\,r_d$. A run is lost if the centroid leaves the map or turns less than a quarter revolution about $\mathbf{p}^*$ after the first 10~s, and the statistics use the runs that were not lost.
""")
line("S-IVm4", "Shrink. Run-in paragraph",
r"\item \emph{Classification.}",
r"""\emph{Classification.} Classification accuracy is the fraction of cycles on which the type of $\hat{\mathbf{J}}$ matches the type of the critical point the trial reached.""")
line("S-IVm5", "Shrink. itemize removed", r"\end{itemize}", None)

sub("S-IV1", "Survival. Cosmetic edit reverted to submitted wording",
"the final centroid position in each trial was used", "the final centroid position of the cluster in each trial was used")
sub("S-IVB", "Shrink. Triangle-rule orbit sentences cut with the rule itself",
r" By the triangle rule, no orbit succeeds at $r_d = 0.01$~m and every analytical orbit succeeds from 0.1~m up. The reconstructed saddle succeeds in \rev{43} to \rev{100}\% of runs from 0.1~m up, and the reconstructed vortex in \rev{0}\% at 0.1 and 0.2~m rising to \rev{100}\% at 0.5~m. % [REVIEW] revision_metrics/tables/table_III_revised.tex; triangle rule: revision_metrics/results/triangle_*.csv",
r" % [REVIEW] revision_metrics/tables/table_III_revised.tex")
sub("S-IVC", "Shrink. Coarse-class clause cut",
r"the star nodes on \rev{95}\%, or \rev{100}\% in coarse classes that merge nodes and spirals of the same stability, against", r"the star nodes on \rev{95}\%, against")

# V cosmetic reverts
for i, (o, n) in enumerate([
    ("OptiTrack positions and RGB readings, computes", "OptiTrack positions and RGB sensor readings, computes"),
    ("minimally affected by overhead lighting.", "minimally affected by overhead lighting variations."),
    ("interpolation models, used in Section~IV as a fixed", "interpolation models, referenced in Section~IV as a fixed"),
    ("previous studies in our group~", "previous studies in our research group~"),
    ("the only canonical field type with separatrices", "the only canonical field type that contains separatrices"),
    ("In the vortex, systematic bias was", "In the vortex field, systematic bias was"),
]):
    sub(f"S-V{i+1}", "Survival. Cosmetic edit reverted to submitted wording", o, n)

# VI-A
sub("S-VIA1", "Survival. Cosmetic edit reverted", "with its errors propagating directly", "with sensing errors propagating directly")
line("S-VIA2", "Controller-only bias defined where it is used (label dropped from IV)",
"The hardware orbit metrics separate estimation error from control error.",
r"""The hardware orbit metrics separate estimation error from control error. The controller-only bias measures the radius about $\hat{\mathbf{p}}^*$, which (\ref{eq:11}) regulates, instead of about $\mathbf{p}^*$. At $r_d = 0.01$~m it was \syn{0.177}~m in the vortex against a true bias of \syn{0.166}~m, so the small-radius offset arises in the closed loop. At larger radii the localization error grows with distance from the critical point, and the within-run spread grows with it. % [SYNTHETIC] V_VIA numbers.tex""")

# VI-B trimmed
line("S-VIB1", "Shrink. Variant description tightened",
"The vector-to-scalar primitive \\cite{48} reduces the flow to its magnitude",
r"""The vector-to-scalar primitive \cite{48} reduces the flow to its magnitude, the speed $S_i = \|\mathbf{v}(\mathbf{p}_i)\|$ at each robot, and a plane fit to the three speeds gives their gradient $\mathbf{g}$. For attraction the cluster descends $\mathbf{g}$. For orbits the primitive was modified to move tangentially to the direction of steepest descent. It was run in simulation with the robot model, gains, and formation of Section~IV in three forms (Table~\ref{tab:baselines}). The first is the primitive alone. The second, the oracle, takes the true critical point as the target of (\ref{eq:10}) and the center of the radial term of (\ref{eq:11}). The third places the critical point where the fitted speed reaches zero, $\hat{\mathbf{p}}_m = \mathbf{p}_c - S_c\mathbf{g}/\|\mathbf{g}\|^2$ with $S_c$ the fitted speed at the centroid, and feeds it to (\ref{eq:10}) and (\ref{eq:11}).""")
line("S-VIB2", "Shrink. Numbers that Table IV already gives dropped from the prose",
"On the six analytical fields the speed grows equally in every direction",
r"""On the six analytical fields the speed grows equally in every direction, so its gradient points directly away from $\mathbf{p}^*$, and the primitive reached $\mathbf{p}^*$ in every trial (Fig.~\ref{fig:effort}(a)). The fitted gradient keeps the same length at every distance, so the command holds the robots at their speed limit until the cluster passes $\mathbf{p}^*$, overshoots by a median of \newnum{0.031}~m, and rebounds. The Jacobian estimate scales the command with distance, and the cluster slows along a smooth exponential. The primitive settles faster and closer to $\mathbf{p}^*$, because friction stops a proportional command on the ring $v_c/k$, but at \newnum{1.9} times the control effort. % [REVIEW] baselines_v2s_summary.json linear: v2s overshoot_med 0.031, e_mean 0.003, t_conv 1.1, E_u 0.137 against 0.074""")
line("S-VIB3", "Shrink. Orbit paragraph tightened; lag-drift reference now points to III-D",
"In orbit, the primitive alone has no mechanism to detect or correct radial error",
r"""In orbit, the primitive alone has no mechanism to detect or correct radial error with respect to a specific point, and so centrifugal drift goes uncorrected. Every orbit spiraled outward off the map (Fig.~\ref{fig:effort}(b)). Given the oracle, the same tangent with the radial term of (\ref{eq:11}) matched the Jacobian method within \newnum{0.002}~m RMSE at every radius. The orbit law therefore holds the radius about whatever point it is given, and the Jacobian estimate supplies that point from the same three readings the primitive uses. With its own speed estimate the primitive held the radius at a lower RMSE, because the estimate's inward bias offsets the lag drift of Section~III-D. That estimate never lets the cluster rest during attraction, however, and the cluster still moved at \newnum{0.12}~m/s after 10~s. % [REVIEW] baselines_v2s_summary.json linear: v2s lost 100% at every r_d (all left the map); v2s_oracle RMSE 0.092/0.052/0.035 against 0.093/0.051/0.033; v2s_est RMSE 0.068/0.016/0.020, final_speed 0.116, E_u 0.306""")
line("S-VIB4", "Shrink. Anisotropic node and gyre paragraph tightened; \"the speed points\" corrected to the speed gradient",
"The speed points at the critical point only where it grows equally in every direction.",
r"""The speed gradient points away from the critical point only where the speed grows equally in every direction. The anisotropic node $\mathbf{J} = \operatorname{diag}(-1, -0.2)$ is affine, but its speed grows five times more slowly along one axis. There the primitive reached $\mathbf{p}^*$ in \newnum{7}\% of trials, and most of the rest ended on the slow axis, a median \newnum{0.67}~m from $\mathbf{p}^*$. Its speed estimate erred by \newnum{0.46}~m on average during orbits, and even with the oracle most orbits were lost, because the speed contours are ellipses. The Jacobian estimate is exact on any affine field, including the node. In the gyre, the primitive reached the same critical point as the Jacobian method in \newnum{96}\% of trials. The speed vanishes at the center and at the saddles alike, however, so it cannot tell which it has reached, while the Jacobian method labeled the type on every cycle (Section~IV-C). The primitive lost every gyre orbit, and with the oracle every orbit from $r_d = 0.3$~m, where the gyre's speed contours no longer circle the center. % [REVIEW] baselines_v2s_summary.json node_aniso: v2s success 6.6% (467 failures, median |y| 0.67 m, |x| 0.03 m), v2s_est est_orb 0.455, v2s_oracle lost 90/82/100%; gyre: same point as jacobian 96.4%, jacobian type_acc 100%, v2s lost 100%, v2s_oracle lost 0/100/100%, jacobian lost 0/0/100%""")

# VI-C submitted skeleton restored (F5c: "Actuator lag lowers this limit" cut)
line("S-VIC1", "Survival. Submitted First paragraph restored",
"The estimator fits a planar affine model across the formation, which limits the fields it can address.",
r"""Several limitations constrain the current implementation and experimental validation. First, the approach is restricted to planar (2D) vector fields, though many real-world applications involve three-dimensional flows. Extension to 3D fields would require at minimum four non-coplanar robots for Jacobian estimation.

Second, the testbed validates performance only on static fields where critical points remain fixed during operation. The static printed field approach enables precise ground truth validation but does not capture dynamic field behavior, and the printed fields do not push the robots. A physical flow advects the cluster and adds its own velocity $\mathbf{J}\mathbf{e}$ to the centroid near $\mathbf{p}^*$. For an ideal centroid with an exact estimate, the error then obeys $\dot{\mathbf{e}} = (\mathbf{J} - k\mathbf{I})\mathbf{e}$, which is unstable when an eigenvalue of $\mathbf{J}$ has real part above $k$. A physical critical point can also drift, and the attraction law trails one moving at constant speed by that speed divided by $k$.

Third, robot team sizes were limited to three. While the theoretical framework supports arbitrary team sizes through the overdetermined system approach, larger formations present practical implementation challenges including communication bandwidth, computational complexity, and formation control stability. With $n > 3$ robots the least-squares fit averages measurement errors across robots and survives the loss of one. However, a wider footprint asks the affine model to hold over more of the field's curvature.

Fourth, the estimator assumes the field is well approximated by an affine model from the cluster out to the critical point (Section~II-B). On fields that are highly nonlinear at the scale of the formation, this local linear fit degrades, biasing the recovered critical point location. Section~IV probes this regime on a gyre cell and a curved saddle, but fields with stronger curvature, or with critical points closer together than one formation side, were not tested.""")
line("S-VIC2", "Folded into Third", "All experiments used three robots, the minimum.", None)
line("S-VIC3", "Fix 5. Folded into Second; unverified \"Actuator lag lowers this limit in rotating flow\" cut",
"The printed fields give exact ground truth, but they are static and do not push the robots.", None)
line("S-VIC4", "Survival. Finally paragraph with the submitted closing sentence restored",
"The computation is centralized.",
r"""Finally, the computation is centralized. A central computer receives OptiTrack positions and field readings from every robot, forms the estimate, and returns velocity commands. The estimate needs only the positions and readings of one cycle in a common frame, so a robot receiving them from its neighbors could compute it. Onboard relative localization and inter-robot communication were not tested. A dropout can leave fewer than three readings, and the fit then has no unique solution, while latency mixes readings from different times into one fit. The robustness of the approach to lower update rates and communication dropouts was not tested.""")

# VI-D
line("S-VID1", "Shrink and survival. Classification-band idea cut (nothing downstream uses it); submitted time-varying paragraph restored in short form",
"The estimated Jacobian's eigenvalues determine the topological type",
r"""The estimation framework uses only instantaneous measurements and carries no state between control cycles. This memoryless property means the estimator does not assume field stationarity, suggesting applicability to environments where the field itself varies with time. Validating the estimator on a genuinely time-varying field remains future work.""")
line("S-VID2", "Survival. Submitted separatrix sentences restored; six-robot extension kept in one sentence",
"Because the three-robot affine fit developed here returns a spatially constant Jacobian",
r"""Beyond critical point detection and orbital control, vector fields contain other topological features of practical interest. Separatrices connected to saddle points define boundaries between distinct flow regimes and act as Lagrangian coherent structures in steady flows. The three-robot fit returns a spatially constant Jacobian, so detecting them calls for a second-order fit, which six robots that do not lie on a common conic can provide.""")


# ---------------------------------------------------------------- figures, uniform 5% width reduction (Chris, 2026-11-05)
for i, (o, n) in enumerate([
    (r"\includegraphics[width=\columnwidth]{six_vector_fields_3x2.png}", r"\includegraphics[width=0.95\columnwidth]{six_vector_fields_3x2.png}"),
    (r"\includegraphics[width=\columnwidth]{control_architecture_2.png}", r"\includegraphics[width=0.95\columnwidth]{control_architecture_2.png}"),
    (r"\includegraphics[width=\columnwidth]{fig_sim_convergence_band.pdf}", r"\includegraphics[width=0.95\columnwidth]{fig_sim_convergence_band.pdf}"),
    (r"\includegraphics[width=\columnwidth]{fig_nonlinear_triangles.pdf}", r"\includegraphics[width=0.95\columnwidth]{fig_nonlinear_triangles.pdf}"),
    (r"\includegraphics[width=0.62\columnwidth]{testbed_with_four.png}", r"\includegraphics[width=0.59\columnwidth]{testbed_with_four.png}"),
    (r"\includegraphics[width=\columnwidth]{fig_reconstruction_3x2.png}", r"\includegraphics[width=0.95\columnwidth]{fig_reconstruction_3x2.png}"),
    (r"\begin{minipage}[b]{0.47\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_a.png}", r"\begin{minipage}[b]{0.45\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_a.png}"),
    (r"\begin{minipage}[b]{0.47\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_b.png}", r"\begin{minipage}[b]{0.45\columnwidth}\centering\includegraphics[width=\textwidth]{placeholder_hw_trajectories_b.png}"),
    (r"\includegraphics[width=0.92\columnwidth]{placeholder_hw_distance.png}", r"\includegraphics[width=0.87\columnwidth]{placeholder_hw_distance.png}"),
    (r"\includegraphics[width=\columnwidth]{fig_v2s_effort.pdf}", r"\includegraphics[width=0.95\columnwidth]{fig_v2s_effort.pdf}"),
]):
    sub(f"S-fig{i+1}", "Shrink. Figure width reduced 5% (uniform across all figures)", o, n)


# ---------------------------------------------------------------- round 2 shrink (after the figure shrink)
line("S-R2a", "Shrink. Roadmap paragraph cut (style 6.4: no roadmap in papers); survival allows it",
"The remainder of this paper is organized as follows.", None)
line("S-R2b", "Shrink. Counting paragraph cut; Proposition 1 and its proof cover n <= 2 and least squares",
"With fewer than three robots, the system is underdetermined", None)
sub("S-R2c", "Shrink, less math. Lyapunov display folded into one inline inequality; final-error sentence shortened",
"""and $V = \\frac{1}{2}\\|\\mathbf{e}\\|^2$ satisfies
\\[
\\dot{V} = -k\\|\\mathbf{e}\\|^2 + k\\,\\mathbf{e}^T\\boldsymbol{\\delta} \\le -k\\|\\mathbf{e}\\|\\bigl(\\|\\mathbf{e}\\| - \\|\\boldsymbol{\\delta}\\|\\bigr),
\\]
so the error decreases""",
"""and $V = \\frac{1}{2}\\|\\mathbf{e}\\|^2$ satisfies $\\dot{V} \\le -k\\|\\mathbf{e}\\|(\\|\\mathbf{e}\\| - \\|\\boldsymbol{\\delta}\\|)$, so the error decreases""")
sub("S-R2d", "Shrink. Final-error sentence shortened",
r"It rests within $v_c/k$ of the estimate, and the estimate lies $\|\boldsymbol{\delta}_{\text{final}}\|$ from the critical point, so writing $\mathbf{e}_{\text{final}} = (\mathbf{p}_c - \hat{\mathbf{p}}^*) + \boldsymbol{\delta}_{\text{final}}$ gives $\|\mathbf{e}_{\text{final}}\| \le v_c/k + \|\boldsymbol{\delta}_{\text{final}}\|$.",
r"It rests within $v_c/k$ of the estimate, which lies $\|\boldsymbol{\delta}_{\text{final}}\|$ from the critical point, so $\|\mathbf{e}_{\text{final}}\| \le v_c/k + \|\boldsymbol{\delta}_{\text{final}}\|$.")
sub("S-R2e", "Shrink. Two classification sentences merged",
"on every reading, the boundary types are labeled least reliably. The center is labeled correctly", "on every reading, the center is labeled correctly")
sub("S-R2f", "Shrink. Table II success rows merged into one, as in Table III",
r"""Succ.\ 3~cm (\%) & 100 & 45 & 62 & \newnum{76} & \newnum{88} & \newnum{97} \\
Succ.\ 4~cm (\%) & 100 & 79 & 89 & \newnum{100} & \newnum{100} & \newnum{100} \\
Succ.\ 5~cm (\%) & 100 & 95 & 100 & \newnum{100} & \newnum{100} & \newnum{100} \\""",
r"""Succ.\ 3/4/5~cm (\%) & 100 & 45/79/95 & 62/89/100 & \newnum{76/100/100} & \newnum{88/100/100} & \newnum{97/100/100} \\""")


# ---------------------------------------------------------------- round 3: warnings and slack
sub("S-R3a", "Format. Table II padding 1.5pt -> 1.3pt so the merged success row fits (was 0.8pt overfull)",
r"""\label{tab:simulation_results}
\setlength{\tabcolsep}{1.5pt}""", r"""\label{tab:simulation_results}
\setlength{\tabcolsep}{1.3pt}""")
sub("S-R3b", "Shrink. Forward pointer cut (style 3.3); V-A describes the reconstruction",
" Further details on the reconstructed fields are provided in Section~V.", "")

sub("S-R3c", "Shrink. The two short future-work paragraphs merged into one",
"time-varying field remains future work.\n\nBeyond critical point detection", "time-varying field remains future work. Beyond critical point detection")

def apply(text):
    lines = text.split("\n")
    log = []
    for kind, i, why, old, new in E:
        if i == "F1e":
            # the \end{equation} that closes the old orbit_cf block: it directly precedes the "The error $r^* - r_d$" line in VI-A
            idx = [k for k, l in enumerate(lines) if l is not None and l.startswith("The error $r^* - r_d$ approaches")]
            assert len(idx) == 1, (i, len(idx))
            k = idx[0] - 1
            assert lines[k] == r"\end{equation}", (i, lines[k])
            log.append((i, why, lines[k], None))
            lines[k] = None
            continue
        if kind == "sub":
            joined = "\n".join(l for l in lines if l is not None)
            n = joined.count(old)
            assert n == 1, (i, n, old[:80])
            lines = joined.replace(old, new).split("\n")
            log.append((i, why, old, new))
        else:
            idx = [k for k, l in enumerate(lines) if l is not None and l.startswith(old)]
            assert len(idx) == 1, (i, len(idx), old[:80])
            k = idx[0]
            log.append((i, why, lines[k], new))
            lines[k] = None if new is None else new
            if i in ("F1c", "F1d"):
                pass
    return "\n".join(l for l in lines if l is not None), log


if __name__ == "__main__":
    src, out_tex, out_md = sys.argv[1:4]
    text = open(src).read()
    # F1c/F1d/F1e/F1f target the old VI-A block; resolve them before F1a inserts the new copy in III-D
    order = ["F1c", "F1d", "F1e", "F1f"]
    E.sort(key=lambda e: (order.index(e[1]) if e[1] in order else len(order)))
    new_text, log = apply(text)
    open(out_tex, "w").write(new_text)
    with open(out_md, "w") as f:
        f.write("# Shrink-and-fix batch, 2026-11-05\n\n")
        f.write("Target: `cwaight_systems_paper/cwaight_systems_paper_with_revisions2.tex`. Apply by exact-text match; abort the batch on any miss.\n")
        f.write("Fixes are F1 to F5 (approved plan). S- edits are shrink or survival. A deleted line shows NEW as (deleted).\n\n")
        f.write("Result on the candidate (Queued_Edits/candidate_revisions2.tex): 12 pages, both bios on page 12 with about 40 pt to spare, "
                "no errors, no undefined references, no warnings; survival.py 61.2% (was 56.5%); abstract 241 words; "
                "no new data numbers (only figure widths and table padding).\n\n")
        f.write("Re-apply against the live file with: python3 Queued_Edits/shrink_batch_2026-11-05.py <live.tex> <out.tex> <out.md> "
                "(every old string must match exactly once, or it stops).\n\n")
        for i, why, old, new in log:
            f.write(f"## {i}\n{why}\n\nOLD:\n```latex\n{old}\n```\nNEW:\n```latex\n{'(deleted)' if new is None else new}\n```\n\n")
    print(f"applied {len(log)} edits")
