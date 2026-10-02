import port
from port import P, T, write
port.PRE = 'ch8'

# Evidence: VF_Robot/experiments/thesis_ch8_{geometry,radius_sweep,magnitude_saddle}.py,
# outputs in experiments/outputs/thesis_ch8/. Numbers reviewed by the author
# 2026-09-30, including the RMS truncation-bias paragraph.

parts = [
T(r"""
% ===================================================================
% CHAPTER 8: CROSS-CUTTING FINDINGS
% ===================================================================
% New text throughout, written 2026-09-30 to Waight_style.md.
\chapter{Cross-Cutting Findings}
\label{ch:crosscutting}

Each rung of the ladder of Section~\ref{sec:ch4:ladder} reads more of the
field than the one below it, and it pays for that in robots, in formation
geometry, and in noise. This chapter compares the rungs on those costs.
Three short numerical experiments supply the evidence where no chapter
already does.

\section{Vector Fields as Scalar Landscapes}
\label{sec:ch8:scalar}

Earlier work in this laboratory navigated scalar fields with a library of
primitives for extrema finding, contour following, ridge and trench
following, and saddle point station keeping \cite{sys:5}. A vector field
is not a scalar field, but it supplies several. Each robot reads the
magnitude $\lVert\mathbf{v}\rVert$ directly. The divergence, the
vorticity, the determinant $D$, and the strain eigenvalue $s_1$ are all
built from the velocity gradient. Each of these is a landscape the
cluster can climb, descend, or follow.

Table~\ref{tab:ch8:scalar} shows what each fit order makes available. A
fit of order $k$ recovers derivatives of the field up to order $k$. A
quantity built from derivatives of order $m$ therefore has a value at
order $m$, a gradient at order $m + 1$, and a curvature at order $m + 2$.
A primitive needs the gradient to steer on a landscape and the curvature
to know what shape it is on. The magnitude sits one order below the
quantities built from $\mathbf{J}$. This is why a three-robot fit can
steer on the magnitude but only read the value of $D$ and $s_1$, and why
steering on $D$ or $s_1$ needs six robots. The true curvature of $D$,
which the second-order trackers approximate with the Hessian of the
fitted $\hat{D}$, needs ten.

\begin{table}[htbp]
\centering
\caption{Scalar fields supplied by a planar vector field, and the fit
order needed for each field's value, gradient, and curvature. A field is
objective when an observer rotating relative to the flow reads the same
value (Section~\ref{sec:ch3:relation}).}
\label{tab:ch8:scalar}
\begin{tabular}{llcccc}
\hline
Field & Built from & Value & Gradient & Curvature & Objective \\
\hline
Magnitude $\lVert\mathbf{v}\rVert$ & $\mathbf{v}$ & 0 & 1 & 2 & no \\
Divergence $\operatorname{tr}\mathbf{J}$ & $\mathbf{J}$ & 1 & 2 & 3 & yes \\
Vorticity $\omega$ & $\mathbf{J}$ & 1 & 2 & 3 & no \\
Determinant $D$ & $\mathbf{J}$ & 1 & 2 & 3 & no \\
Strain eigenvalue $s_1$ & $\mathbf{J}$ & 1 & 2 & 3 & yes \\
\hline
\end{tabular}
\end{table}

Critical points enter this table through the magnitude. A critical point
is where the velocity vanishes, so it is a zero of
$\lVert\mathbf{v}\rVert$. Near a nondegenerate critical point
$\mathbf{p}^*$,
\begin{equation}
    \lVert\mathbf{v}\rVert^2 \approx
    (\mathbf{p} - \mathbf{p}^*)^{\top}\mathbf{J}^{\top}\mathbf{J}\,
    (\mathbf{p} - \mathbf{p}^*),
    \label{eq:ch8:mag_quad}
\end{equation}
and $\mathbf{J}^{\top}\mathbf{J}$ is positive definite whenever
$\det\mathbf{J} \neq 0$, whatever the type. Every nondegenerate critical
point is therefore a minimum of the magnitude, saddles included.
Descending the magnitude reaches a critical point but cannot say which
type it reached, since a saddle and a center can present the same
magnitude landscape. The first-order estimate of
Chapter~\ref{ch:first_order} attacks the same zero directly. Its step
$\hat{\mathbf{p}}^* - \mathbf{p}_c = -\hat{\mathbf{J}}^{-1}\hat{\mathbf{v}}_c$
is one Newton step toward $\mathbf{v} = \mathbf{0}$, and the same
$\hat{\mathbf{J}}$ names the type.

A closed-loop test confirms this with the three-robot simulator of
Chapter~\ref{ch:first_order}. The vector-to-scalar primitive of
Section~\ref{sec:ch5:vector_to_scalar} descended the magnitude from 1000
random starts in a 1~m box around a saddle. On the canonical saddle of
Appendix~\ref{app:fields} it finished within 0.006~m of the saddle in
95\% of trials, the same as on the vortex, whose magnitude landscape is
identical. On a saddle with rates 1 and $-1/3$ the magnitude is an
elliptic cone with a sharp point at the saddle, and a plane fit across
that point is biased. The cluster then settled a median of 0.048~m from
the saddle, with 91\% of trials within 0.05~m. The attraction law of
Chapter~\ref{ch:first_order} finished within 0.015~m in 95\% of trials on
all three fields (Fig.~\ref{fig:ch8:magnitude}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.55\textwidth]{ch8_magnitude_saddle.png}
    \caption{Magnitude descent (blue) and the attraction law (red) from ten
    matched starts on a saddle with rates 1 and $-1/3$, simulated with the
    robot model of Chapter~\ref{ch:first_order}. The cross marks the
    saddle. Magnitude descent follows the steepest slope of the elliptic
    magnitude cone and settles a few centimeters from the saddle, while the
    attraction law runs nearly straight to it.}
    \label{fig:ch8:magnitude}
\end{figure}

\section{Sampling Geometry}
\label{sec:ch8:geometry}

A fit of order $k$ has $n_k = \tfrac{1}{2}(k+1)(k+2)$ coefficients per
component (Section~\ref{sec:ch4:minimality}), so it needs 3, 6, 10, and 15
robots for $k = 1$ to 4. With exactly $n_k$ robots the fit fails when the
robots lie on a common algebraic curve of degree $k$, a line for the
first order, a conic for the second, and a cubic for the third.
Interpolation theory calls a point set that avoids this unisolvent.

The failure has a physical reading. When the robots share such a curve,
the null space of $\boldsymbol{\Phi}_k$ holds the curve's own
coefficients. Any multiple of that polynomial can be added to the fitted
field without changing a single reading, so the robots cannot tell the
true field from the true field plus the curve. A regular hexagon lies on
its circumscribed circle, and its quadratic fit is singular at any size.
Adding robots to the same circle does not help, since each of them reads
the same ambiguity. Nine robots on a ring with a tenth at the center fail
a cubic fit in two ways at once. Their null space is two-dimensional,
the circle times any line through the center robot.

A formation that avoids these curves can still come close to them, and
its condition number indicates how much the fit amplifies noise.
Table~\ref{tab:ch8:geometry} lists the formations of this dissertation and
candidates for a cubic fit, with $\kappa$ computed on the
radius-normalized formation matrix of Section~\ref{sec:ch7:sensitivity}.
The equilateral triangle gives $\kappa(\boldsymbol{\Phi}_1) = 1.41$ and
the pentagon plus center gives $\kappa(\boldsymbol{\Phi}_2) = 8.26$. Ten
robots are harder to place. A triangular lattice of 1, 2, 3, and 4 robots
supports a cubic fit but gives $\kappa = 216$. A search over two-ring
layouts found $\kappa = 44.3$ for an inner pentagon at 0.65 of the outer
radius, rotated $36^\circ$ against the outer pentagon. The best cubic
layout found therefore has a condition number about five times that of
the pentagon plus center, before formation size enters at all.

\begin{table}[htbp]
\centering
\caption{Condition numbers of the radius-normalized formation matrix.
Singular formations lie on a common curve of the fit's degree.}
\label{tab:ch8:geometry}
\small
\begin{tabular}{lccl}
\hline
Formation & Order & Robots & $\kappa(\boldsymbol{\Phi}_k)$ \\
\hline
Equilateral triangle & 1 & 3 & 1.41 \\
Pentagon plus center & 2 & 6 & 8.26 \\
Regular hexagon & 2 & 6 & singular (circle) \\
Hexagon plus center & 2 & 7 & 8.84 \\
Nine-robot ring plus center & 3 & 10 & singular (circle $\times$ line) \\
Triangular lattice 1+2+3+4 & 3 & 10 & 216 \\
Inner triangle, outer hexagon, center & 3 & 10 & 211 \\
Two rotated pentagons, radii 0.65 and 1 & 3 & 10 & 44.3 \\
\hline
\end{tabular}
\end{table}

\section{Noise Growth with Fit Order}
\label{sec:ch8:noise_order}

A higher-order fit reads finer structure from the same readings, and it
pays for that structure in noise. A coefficient of order $q$ is a $q$-th
derivative of the field, estimated from differences between readings a
formation radius apart. Each additional derivative divides by another
factor of the radius.

Let a formation of fixed shape $\boldsymbol{\xi}_i$ be scaled to radius
$\rho$, so robot $i$ sits at $\rho\,\boldsymbol{\xi}_i$. Each entry of the
basis of Section~\ref{sec:ch4:basis} is a monomial of degree $|\alpha|$,
so the formation matrix factors as
\begin{equation}
    \boldsymbol{\Phi}_\rho = \boldsymbol{\Phi}_1\,\mathbf{S}_\rho, \qquad
    \mathbf{S}_\rho = \operatorname{diag}\bigl(\rho^{|\alpha|}\bigr).
    \label{eq:ch8:phi_scaling}
\end{equation}
With independent measurement noise of standard deviation $\sigma$ on each
reading, the fitted coefficient of degree $q$ has standard deviation
\begin{equation}
    \sigma_{\alpha} = \rho^{-q}\,\sigma\,
        \bigl\lVert \mathbf{e}_\alpha^{\top}\boldsymbol{\Phi}_1^{-1}\bigr\rVert,
    \qquad q = |\alpha|,
    \label{eq:ch8:coef_noise}
\end{equation}
where the row norm depends on the formation's shape alone.

What matters at the terminal step is the error in whatever quantity the
controller drives to zero. An error $\delta f$ in that quantity moves its
estimated zero by $\delta n \approx \delta f/|\partial f/\partial n|$,
where $n$ is the direction in which $f$ changes. If $f$ is built from
derivatives up to order $m$, then by (\ref{eq:ch8:coef_noise}) its error
$\delta f$ scales as $\rho^{-m}\sigma$, and
\begin{equation}
    \delta n \sim \frac{\rho^{-m}\,\sigma}{|\partial f/\partial n|}.
    \label{eq:ch8:terminal_noise}
\end{equation}
The first-order law drives the velocity itself to zero ($m = 0$). With
the centroid on the critical point, the fit's value there is the mean of
the three readings, so its error has standard deviation $\sigma/\sqrt{3}$
per component. This recovers the estimate $\sigma_p \approx
\lVert\mathbf{J}^{-1}\rVert\,\sigma_v/\sqrt{3}$ of
Section~\ref{sec:ch6:error}, which does not depend on $\rho$. Both
second-order trackers drive a transverse gradient to zero,
$\mathbf{w}_2^{\top}\nabla\hat{D}$ or $\mathbf{P}\nabla\hat{s}_1$, which
is built from second derivatives ($m = 2$). Their terminal error should
therefore grow as $\rho^{-2}$.

The denominator of (\ref{eq:ch8:terminal_noise}) is the field's share of
the error, the inverse slope of the quantity being zeroed. For the
critical point it is $\lVert\mathbf{J}^{-1}\rVert$, which diverges as
$\det\mathbf{J} \to 0$, the $\kappa(\mathbf{J})$ effect of
Section~\ref{sec:ch6:error}. For the $D$ tracker it is $1/|\lambda_2|$,
the inverse transverse curvature that already divides its across-trench
command, and it diverges as the trench flattens. The formation's share,
$\kappa(\mathbf{A})$ or $\kappa(\boldsymbol{\Phi})$, is set by the robot
positions alone.

A Monte Carlo test on the double gyre checks both predictions
(Fig.~\ref{fig:ch8:radius}). A triangle was centered on a gyre center and
on a saddle, and a pentagon plus center on the separatrix at
$(0, 0.25)$. The radius ran from a quarter to twice the nominal 0.075,
with a random heading and measurement noise $\sigma = 0.002$ on every
reading, over 1000 trials per radius. Each trial was fit twice, with and
without noise, so the difference isolates the noise. At first order the
scatter of the estimated critical point was flat in radius, with a
log-log slope of 0.01 at the gyre center and 0.02 at the saddle, and it
matched $\lVert\mathbf{J}^{-1}\rVert\sigma/\sqrt{3}$ to within 6\%. The
transverse errors of the $D$ and $s_1$ trackers grew with slopes of
$-2.04$ and $-2.01$.

Truncation bias runs the other way. The noise-free fits carried an RMS
error that grew as $\rho^{3.0}$ at first order and as $\rho^{0.95}$ and
$\rho^{0.98}$ for the two trackers. At first order the bias overtook the
noise at about 1.5 times the nominal radius, and for the $D$ tracker the
two were nearly equal at twice the nominal radius.

At every radius the transverse error of the $s_1$ tracker was about 3.6
times that of the $D$ tracker. Section~\ref{sec:ch7:disc_noise} traces
the $s_1$ tracker's failures under noise to the sign of its tangent,
which is selected from the same gradient.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{ch8_radius_sweep.png}
    \caption{Terminal error against formation radius on the double gyre,
    1000 trials per point. Left, the noise alone at $\sigma = 0.002$. The
    first-order critical point estimate is flat in radius, while the
    transverse errors of both second-order trackers fall as $\rho^{-2}$.
    Right, the RMS truncation error of the noise-free fits.}
    \label{fig:ch8:radius}
\end{figure}

\section{Scalability}
\label{sec:ch8:scalability}

More robots than $n_k$ turn the fit into least squares, and the extra
readings average down the noise. How much they help depends on where the
robots go (Fig.~\ref{fig:ch8:gain_vs_N}). For a ring with a center robot,
adding ring robots leaves the noise on the centroid value unchanged and
barely reduces the noise on the second-order coefficients, whose gain
falls only as $N^{-0.22}$. Every ring robot sits on one circle, so the
ring cannot separate the constant term from the curvature term
$x^2 + y^2$, and the center robot carries both alone. With the robots
split between two rings and a center, the gains fall as $N^{-0.46}$ to
$N^{-0.50}$ for $N \geq 10$, close to the $N^{-1/2}$ of independent
averaging, and the condition number holds near 7.4.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch8_gain_vs_N.png}
    \caption{Noise gain of the quadratic fit's coefficients against the
    number of robots, at radius 1. Dashed, a ring with a center robot.
    Solid, a center robot and two rings, shown from ten robots, where each
    ring holds at least four. The dotted lines fall as $N^{-1/2}$.}
    \label{fig:ch8:gain_vs_N}
\end{figure}

The same geometry decides whether a cluster survives losing a robot. A
formation with exactly $n_k$ robots cannot lose any of them. A ring with a
center robot keeps its quadratic fit when any ring robot fails, but
losing the center leaves the rest on a circle, which is fatal. Every
two-ring formation from seven to fifteen robots kept its quadratic fit
after the loss of any single robot.

The cost of more robots falls on the control architecture. Cluster space
control is centralized, requires global state, and becomes intractable
for suitably large clusters \cite{sep:30}. Each added robot adds two
cluster variables and two rows to the inverse kinematic Jacobian of
Section~\ref{sec:ch4:cluster_space}. The claims of
Chapter~\ref{ch:second_order} are for six robots with ample bandwidth
(Section~\ref{sec:ch9:second_order}), and a ten-robot cubic cluster would
carry the same limitation with more state to share.
"""),
]

write('ch08_crosscutting.tex', parts)
