import port
from port import P, T, write
port.PRE = 'ch8'

parts = [
T(r"""
% ===================================================================
% CHAPTER 8: CROSS-CUTTING FINDINGS
% ===================================================================
% New text throughout. The derivations were checked 2026-09-29 and are
% not in any of the three papers.
\chapter{Cross-Cutting Findings}
\label{ch:crosscutting}

Chapters~\ref{ch:zeroth} to~\ref{ch:second_order} built and tested one
rung of the ladder at a time. This chapter compares the rungs on three
questions that arise only side by side. These are how noise grows with
fit order, how formation size trades against it, and how each rung
behaves when the observer's frame changes.

\section{Noise Amplification with Fit Order}
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
where the row norm depends on the formation's shape alone. This is the
$\rho^{-q}$ law of Section~\ref{sec:ch7:sensitivity}, stated for any fit
order.

A feature is located by driving some fitted quantity $f$ to zero. An
error $\delta f$ moves the estimated zero by
$\delta n \approx \delta f/|\partial f/\partial n|$, where $n$ is the
direction in which $f$ changes. If $f$ is built from derivatives up to
order $m$, then (\ref{eq:ch8:coef_noise}) makes $\delta f$ scale as
$\rho^{-m}\sigma$ to leading order, and
\begin{equation}
    \delta n \sim \frac{\rho^{-m}\,\sigma}{|\partial f/\partial n|}.
    \label{eq:ch8:terminal_noise}
\end{equation}

For the critical point, $f$ is the velocity itself ($m = 0$) and its slope
is $\mathbf{J}$. From $\hat{\mathbf{p}}^* = \mathbf{p}_c -
\hat{\mathbf{J}}^{-1}\hat{\mathbf{v}}_c$,
\begin{equation}
    \delta\hat{\mathbf{p}}^* = -\mathbf{J}^{-1}\,\delta\hat{\mathbf{v}}_c
        + \mathbf{J}^{-1}\,\delta\hat{\mathbf{J}}\,\mathbf{J}^{-1}\mathbf{v}_c .
    \label{eq:ch8:pstar_pert}
\end{equation}
The second term vanishes as the centroid reaches the critical point,
where $\mathbf{v}_c \to \mathbf{0}$. For any nondegenerate triangle the
affine fit's value at the centroid is the mean of the three readings, so
its error has standard deviation $\sigma_v/\sqrt{3}$ per component. This
recovers the estimate $\sigma_p \approx
\lVert\mathbf{J}^{-1}\rVert\,\sigma_v/\sqrt{3}$ of
Section~\ref{sec:ch6:error}, which has no dependence on $\rho$.

For both second-order trackers the zero sought is a transverse gradient,
$\mathbf{w}_2^{\top}\nabla\hat{D}$ or $\mathbf{P}\nabla\hat{s}_1$. Each is
built from second derivatives ($m = 2$), and its slope is the transverse
curvature of $D$ or $s_1$. The terminal error of the first-order law
therefore does not depend on formation radius, while that of either
second-order tracker grows as $\rho^{-2}$. The scaling does not explain
the factor of four between the two second-order trackers, which share
$m = 2$. Section~\ref{sec:ch7:disc_noise} traces that factor to how long
the $s_1$ tracker carries a sign decision. No experiment here sweeps fit
order at a fixed radius, so (\ref{eq:ch8:terminal_noise}) is untested
directly.

The same argument names the field's share of the error. The amplifier is
the inverse slope of whatever quantity the controller drives to zero. For
the critical point that is $\lVert\mathbf{J}^{-1}\rVert$, which diverges
as $\det\mathbf{J} \to 0$, the $\kappa(\mathbf{J})$ effect of
Section~\ref{sec:ch6:error}. For the $D$ tracker it is $1/|\lambda_2|$,
the inverse transverse curvature that already divides its across-trench
command, and it diverges as the trench flattens. The formation's share,
$\kappa(\mathbf{A})$ or $\kappa(\boldsymbol{\Phi})$, is set by the robot
positions alone and can be computed before any reading is taken.

\section{Formation Size}
\label{sec:ch8:formation_size}

At first order the radius enters the location error only during the
approach. The second term of (\ref{eq:ch8:pstar_pert}) carries
$\delta\hat{\mathbf{J}}$, which scales as $\rho^{-1}$, and it vanishes at
the critical point. Truncation bias grows with the radius, and
Chapter~\ref{ch:first_order} notes that the linearization error decreases
as the formation contracts toward the critical point. A smaller triangle
therefore adds no noise at the terminal step.

At second order both effects remain at the terminal step, so the radius
is a real tradeoff. Chapter~\ref{ch:second_order} sets it per field,
$\rho = 0.075$ on the double gyre and $\rho = 0.098$ on the Santa Barbara
Channel record, where the larger footprint spans about five cells of the
2~km grid. Neither chapter sweeps the radius against its estimation
error, so neither identifies an optimal radius.

\section{Observer Frames and Sensor Registration}
\label{sec:ch8:objectivity}

Chapter~\ref{ch:second_order} showed that a rotating observer shifts the
determinant field but not the strain eigenvalues. The same shift acts on
the first-order classification, which reads the eigenvalues of the full
Jacobian. Write $\mathbf{J} = \mathbf{S} + \tfrac{\omega}{2}\mathbf{E}$
with $\mathbf{E} = \left[\begin{smallmatrix} 0 & -1\\ 1 & 0
\end{smallmatrix}\right]$. An observer rotating at rate $\Omega$, signed
as in Section~\ref{sec:ch3:relation} so that the measured vorticity rises
by $2\Omega$, sees $\mathbf{J}' = \mathbf{J} + \Omega\mathbf{E}$ in the
co-rotating basis. Since $\operatorname{tr}\mathbf{E} = 0$,
\begin{equation}
    \operatorname{tr}\mathbf{J}' = \operatorname{tr}\mathbf{J},
    \qquad
    \det\mathbf{J}' = \det\mathbf{J} + \Omega\,\omega + \Omega^2 .
    \label{eq:ch8:rotating_J}
\end{equation}
The second identity is the shift (\ref{eq:ch3:D_not_objective}). The real
part of the eigenvalues is half the trace and does not change, so a
rotating observer cannot turn a sink into a source. The determinant and
the discriminant do change, so a fast enough rotation can relabel a
saddle as a node or spiral, or a node as a spiral. The location of the
critical point moves as well, since the rotating observer adds a
solid-body rotation to the measured field, which is nonzero away from the
center of rotation. These are consequences of the decomposition, and
Chapter~\ref{ch:first_order} tests none of them.
The orbital law's tangent, a fixed $-90^\circ$ rotation of the radial
vector, is also a world-frame convention of the kind
Section~\ref{sec:ch7:disc_noise} rules out as a sign reference for a
cluster whose heading drifts.

A reflection of the sensor frame is harsher than a rotation. It reverses
the sign of $\det\hat{\mathbf{J}}$, so it exchanges saddles with nodes
while leaving the located critical point in place. The hardware logs show
this directly (Section~\ref{sec:ch6:classification}). With the saddle
map's frame unregistered, the estimate that located the saddle correctly
read it as a stable node or spiral in every terminal cycle. Location
survives frame conventions that type classification does not.

\section{What Each Rung Certifies at the Terminal Step}
\label{sec:ch8:terminal}

The zeroth-order primitives certify a direction and nothing about
location. The first-order attraction law converges exponentially to its
own estimate, and with no stiction, no speed limit, and instantaneous
response it reaches the controller's $10^{-6}$~m deadband
(Section~\ref{sec:ch6:error}). Its terminal error in every physical or
physically modeled trial comes from actuator stiction, outside the
estimator. The second-order trackers cannot certify a minimum from their
own curvature. The fitted $\hat{s}_1$ is concave by construction, so its
capture test uses a vanishing gradient, which suffices only because a
flow saddle happens to be a true minimum of $s_1$. The $D$ tracker's sign
test reads the fitted curvature, but near a flow saddle truncation error
sets that sign (Section~\ref{sec:ch7:sep_controller}).

\section{Primitive Library}
\label{sec:ch8:library}

Table~\ref{tab:ch8:library} lists every primitive in this dissertation
with its fit order, the smallest cluster it was run with, the quantity it
steers on, its stability result, and how it was verified.

\begin{table}[htbp]
\centering
\footnotesize
\caption{Primitive library. Stability entries are as stated in the
chapter that defines each primitive.}
\label{tab:ch8:library}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{2.0cm} c c p{2.7cm} p{3.4cm} p{3.0cm}}
\hline
Primitive & Order & Robots & Steers on & Stability & Verification \\
\hline
Vector-sum & 0 & 3 & mean sensed vector & none stated & hardware, fixed and sinking vortex, 10 runs each \\
Vector-to-scalar & 0 & 3 & slope of the sensed magnitude & none stated & hardware, fixed vortex, 10 runs plus randomized starts \\
Attraction & 1 & 3 & $\hat{\mathbf{p}}^* = -\hat{\mathbf{J}}^{-1}\hat{\mathbf{h}}$ & exponential convergence to a stationary estimate, $\tau = 1/k$ & simulation, 8 fields; hardware, 157 trials \\
Orbital & 1 & 3 & $\hat{\mathbf{p}}^*$ with radial and tangential terms & radial error $\dot{e}_r = -k_r e_r$ & simulation, 8 fields; hardware, 12 trials \\
$D$ tracker & 2 & 6 & $\hat{D}$ and $\hat{\mathbf{H}}_{D,0}$ & transverse practical stability, $\limsup|n| \leq \delta/a_\perp$ & simulation, double gyre and Santa Barbara Channel \\
$s_1$ tracker & 2 & 6 & $\hat{s}_1$ and $\nabla\hat{s}_1$ & same transverse bound, frame equivariant past its seed step & simulation, double gyre and Santa Barbara Channel \\
\hline
\end{tabular}
\end{table}
"""),
]

write('ch08_crosscutting.tex', parts)
