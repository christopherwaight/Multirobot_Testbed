import port
from port import P, T, write

# ------------------------------------------------------------------ A
port.PRE = 'appA'
write('appA_fields.tex', [
T(r"""
\chapter{Canonical Field Definitions}
\label{app:fields}

\section{Six Canonical Critical Point Fields}
\label{sec:appA:canonical}
"""),
# The canonical-field equation blocks are kept from the submitted paper: the
# revision moved the definitions inline and dropped the appendix listing.
P('sys0', 'Six vector field environments centered at the origin were used to evaluate',
  end='\\end{equation}', extra=0,
  subs=[('Six vector field environments centered at the origin were used to evaluate the multi-robot navigation algorithms.',
         'Six vector field environments centered at the origin were used to evaluate the first-order primitives of Chapter~\\ref{ch:first_order}.')]),
P('sys0', 'The sinking vortex (spiraling inward) and spewing vortex (spiraling outward)',
  end='\\end{equation}'),
T(r"""
\section{Testbed Vortex Fields}
\label{sec:appA:testbed}

% New text
The fixed and sinking vortex maps of Chapter~\ref{ch:zeroth} are given by
(\ref{eq:ch5:rtheta})--(\ref{eq:ch5:sinking}). The fixed vortex there is
the linear clockwise rotation $(y - c_y,\, -(x - c_x))$, the vortex of
(\ref{eq:appA:14}) with the opposite sense. Its sinking component falls
off as $1/r$, so it differs from the linear sinking vortex of
(\ref{eq:appA:15}) and is singular at the center apart from the
$10^{-6}$ regularization.

\section{Steady Double Gyre}
\label{sec:appA:dg}

% New text
The steady double gyre of Chapter~\ref{ch:second_order} is defined in
(\ref{eq:ch3:dg_u}), with its determinant, Hessian, and smaller strain
eigenvalue in closed form in Section~\ref{sec:ch3:dg_example}.

\begin{table}[htbp]
\centering
\caption{Fields used in this dissertation.}
\label{tab:appA:summary}
\begin{tabular}{>{\raggedright\arraybackslash}p{5.2cm} >{\raggedright\arraybackslash}p{5.2cm} c}
\hline
Field & Critical points & Chapter \\
\hline
Vortex, sink, source, saddle & center, stable node, unstable node, saddle & \ref{ch:first_order} \\
Sinking and spewing vortex (linear) & stable and unstable spiral & \ref{ch:first_order} \\
Fixed vortex (testbed) & center & \ref{ch:zeroth} \\
Sinking vortex (testbed, $1/r$ sink) & singular sink at the center & \ref{ch:zeroth} \\
Steady double gyre & two saddles, two centers & \ref{ch:second_order} \\
Santa Barbara Channel record & measured, time-varying & \ref{ch:second_order} \\
\hline
\end{tabular}
\end{table}
"""),
])

# ------------------------------------------------------------------ B
port.PRE = 'appB'
write('appB_kinematics3.tex', [
T(r"""
\chapter{Three-Robot Cluster Kinematics}
\label{app:kin3}
"""),
# Kept from the submitted paper: the revision dropped the three-robot
# kinematics appendix, which the dissertation keeps as Appendix B.
P('sys0', '\\subsection{Forward Kinematics}',
  end='The explicit forms of the equations are omitted from this paper for brevity.',
  subs=[('\\subsection{', '\\section{'),
        ('The explicit forms of the equations are omitted from this paper for brevity.',
         'The explicit entries are not listed here. The simulation software computes '
         'them by forward finite differences of the inverse kinematics '
         '(\\texttt{compute\\_inverse\\_jacobian} in \\texttt{src/control/kinematics.py}), '
         'and the \\texttt{cluster\\_builder} tool generates them in closed symbolic form '
         '(\\texttt{clusterbuilder.py} with the \\texttt{--symbolic} option).')]),
])

# ------------------------------------------------------------------ C
port.PRE = 'appC'
write('appC_kinematics6.tex', [
T(r"""
\chapter{Six-Robot Cluster Kinematics}
\label{app:kin6}

% New text. Checked against src/control/pentagon_kinematics.py
% (VF_Robot), which implements this parameterization.
The six-robot cluster of Chapter~\ref{ch:second_order} is a cluster of
clusters. The robots form three pairs, and the pair midpoints form a
triangle described with the same SAS variables as the three-robot cluster
of Appendix~\ref{app:kin3}. The estimator of
Chapter~\ref{ch:second_order} reads only the six measured positions, so
this parameterization matters to the formation controller alone.

\section{Forward Kinematics}
\label{sec:appC:forward}

Pair A holds robots 1 and 2, pair B robots 3 and 4, and pair C robots 5
and 6. The pair midpoints are
\begin{equation}
    \mathbf{m}_A = \tfrac{1}{2}(\mathbf{p}_1 + \mathbf{p}_2), \quad
    \mathbf{m}_B = \tfrac{1}{2}(\mathbf{p}_3 + \mathbf{p}_4), \quad
    \mathbf{m}_C = \tfrac{1}{2}(\mathbf{p}_5 + \mathbf{p}_6),
    \label{eq:appC:midpoints}
\end{equation}
and the cluster centroid is their mean,
$\mathbf{p}_c = \tfrac{1}{3}(\mathbf{m}_A + \mathbf{m}_B + \mathbf{m}_C)$.
The midpoint triangle has sides
$p_1 = \lVert\mathbf{m}_B - \mathbf{m}_A\rVert$ and
$q_1 = \lVert\mathbf{m}_C - \mathbf{m}_B\rVert$, with $\beta_1$ the
interior angle at $\mathbf{m}_B$, and the heading $\theta_c$ is the
direction from $\mathbf{m}_A$ to $\mathbf{m}_B$. Each pair has a length
and an orientation,
\begin{equation}
    L_2 = \lVert\mathbf{p}_2 - \mathbf{p}_1\rVert, \qquad
    \theta_2 = \operatorname{atan2}(y_2 - y_1,\, x_2 - x_1),
    \label{eq:appC:pair}
\end{equation}
and likewise $(L_3, \theta_3)$ for pair B and $(L_4, \theta_4)$ for pair C.
The twelve cluster variables are
\begin{equation}
    (x_c,\, y_c,\, \theta_c,\, p_1,\, \beta_1,\, q_1,\,
     L_2,\, \theta_2,\, L_3,\, \theta_3,\, L_4,\, \theta_4).
    \label{eq:appC:state}
\end{equation}

\section{Inverse Kinematics}
\label{sec:appC:inverse}

The inverse kinematics builds the midpoint triangle from
$(p_1, \beta_1, q_1)$ in a local frame, as in
Appendix~\ref{app:kin3}, then rotates it by the heading and translates it
to $(x_c, y_c)$. Each pair's two robots are then placed at its midpoint
plus and minus $\tfrac{1}{2}L_k(\cos\theta_k, \sin\theta_k)$.

\section{Inverse Jacobian}
\label{sec:appC:jacobian}

The $12 \times 12$ inverse Jacobian $\mathbf{J}_c^{-1}$ maps the rates of
the twelve cluster variables to the velocities of the six robots. Its
entries are the partial derivatives of the inverse kinematics and are not
listed here. The simulation software computes them analytically
(\texttt{compute\_inverse\_jacobian} in
\texttt{src/control/pentagon\_kinematics.py}), and the
\texttt{cluster\_builder} tool generates them in closed symbolic form
(\texttt{clusterbuilder.py} with the \texttt{--symbolic} option). The
cluster's heading is not regulated, since the formation controller
commands zero spin throughout.
"""),
])

# ------------------------------------------------------------------ D
port.PRE = 'appD'
write('appD_frame_equivariance.tex', [
T(r"""
\chapter{Objectivity and Frame Equivariance of the $s_1$ Tracker}
\label{app:frame_equivariance}

% New text. Draft 11 states the frame shift without the algebra; the
% derivation below was checked 2026-09-29.
Section~\ref{sec:ch3:relation} states that an observer rotating at rate
$\Omega$ sees the determinant field shifted by $\Omega\omega + \Omega^2$
and the strain eigenvalues unchanged. This appendix gives the algebra and
then separates three claims about the $s_1$ tracker that differ in
strength: its features are objective, its command is covariant, and its
trajectory is equivariant only under conditions.

Split the velocity gradient into strain and spin,
\begin{equation}
    \mathbf{S} = \mu\mathbf{I} +
    \begin{bmatrix} s_n & s_s \\ s_s & -s_n \end{bmatrix},
    \qquad
    \mathbf{W} = \frac{\omega}{2}\mathbf{E}, \qquad
    \mathbf{E} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix},
    \label{eq:appD:split}
\end{equation}
with $r = \sqrt{s_n^2 + s_s^2}$, so the strain eigenvalues are
$s_{1,2} = \mu \mp r$. Expanding the determinant,
\begin{equation}
    D = \det(\mathbf{S} + \mathbf{W})
      = \mu^2 - r^2 + \frac{\omega^2}{4}
      = s_1 s_2 + \frac{\omega^2}{4}.
    \label{eq:appD:D_split}
\end{equation}

The rotating observer sees the field
\begin{equation}
    \mathbf{v}'(\mathbf{p}') = \mathbf{Q}\,\mathbf{v}(\mathbf{Q}^{\top}\mathbf{p}')
        + \Omega\,(-y',\, x')^{\top}.
    \label{eq:appD:rotating_field}
\end{equation}
The gradient of the first term is $\mathbf{Q}\mathbf{J}\mathbf{Q}^{\top}$.
A planar rotation commutes with $\mathbf{E}$, so the strain co-rotates to
$\mathbf{Q}\mathbf{S}\mathbf{Q}^{\top}$, which keeps $\mu$ and $r$, and
the spin is unchanged. The gradient of the second term is
$\Omega\mathbf{E}$, which is pure spin. The observer's gradient is
therefore
$\mathbf{J}' = \mathbf{Q}\mathbf{S}\mathbf{Q}^{\top} +
(\tfrac{\omega}{2} + \Omega)\mathbf{E}$, and by (\ref{eq:appD:D_split})
\begin{equation}
    D' = \mu^2 - r^2 + \Bigl(\frac{\omega}{2} + \Omega\Bigr)^2
       = D + \Omega\,\omega + \Omega^2 ,
    \label{eq:appD:shift}
\end{equation}
while $s_1' = s_1$ and $s_2' = s_2$.

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
"""),
])

# ------------------------------------------------------------------ E
port.PRE = 'appE'
write('appE_stability.tex', [
T(r"""
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
"""),
# Kept from the Draft_11 snapshot (sep0): Draft_13 moved the transverse
# stability argument inline into the tracker sections of Chapter 7, but this
# proof states both trackers' bounds together, so the thesis keeps it. Draft_11
# ref 35 (Mas and Kitts) is sys:51 in bib_sources; 'the fold' is Draft_11's
# name for the kink of s_1 at the origin.
P('sep0', 'Both arguments assume the cluster realizes $\\dot{\\mathbf{p}}_c$',
  end='within $L_\\Gamma/(k\\,v_{\\min})$.',
  subs=[('\\cite{35,3}', '\\cite{sys:51,sep:3}'),
        ('as at the\nfold,', 'as at the\nkink of $s_1$,'),
        ('\\begin{IEEEproof}', '\\begin{proof}'), ('\\end{IEEEproof}', '\\end{proof}'),
        ('In (\\ref{eq:d_law}),', 'In (\\ref{eq:ch7:d_law}),'),
        ('In (\\ref{eq:s1_law}),', 'In (\\ref{eq:ch7:s1_law}),'),
        ('Both branches of (\\ref{eq:vpar}) are nonnegative', 'Both branches of (\\ref{eq:ch7:vpar}) are nonnegative'),
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

# ------------------------------------------------------------------ F
port.PRE = 'appF'
write('appF_statistics.tex', [
T(r"""
\chapter{Statistical Methods}
\label{app:statistics}

% New text
\section{Bias and Precision}
Chapter~\ref{ch:first_order} summarizes each set of convergence trials by
its final centroid positions. The systematic bias is the distance from
the mean final position to the true critical point, and it measures what
averaging cannot remove. The precision is the standard deviation of the
final positions, and it measures repeatability. Orbital trials are
summarized by the mean and standard deviation of the radial error.

\section{Circular Statistics for Hue}
Hue is an angle, so a reading near 0 and a reading near 1 are the same
direction. The hue network of Section~\ref{sec:ch2:calibration} predicts
the sine and cosine of hue for this reason, and its accuracy is reported
with circular statistics.

\section{Success Rates}
The noise sweeps of Chapter~\ref{ch:second_order} report success rates
over $N = 10^4$ trials per noise level. Each trial succeeds or fails
independently, so the standard error of a rate $\hat{p}$ is
$\sqrt{\hat{p}(1-\hat{p})/N} \leq 1/(2\sqrt{N}) = 0.005$.
"""),
])
