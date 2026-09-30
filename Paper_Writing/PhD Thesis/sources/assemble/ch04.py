import port
from port import P, T, write
port.PRE = 'ch4'

parts = [
T(r"""
% ===================================================================
% CHAPTER 4: COOPERATIVE ESTIMATION FRAMEWORK
% ===================================================================
\chapter{Cooperative Estimation Framework}
\label{ch:estimation}

% New text
Every primitive in this dissertation runs on the same control
architecture and follows the same estimation pattern. The robots sample
the field at one instant, a local polynomial is fit to those samples, and
a control law steers the cluster on what the fit reveals. This chapter
describes the architecture and the estimation pattern once.

\section{Multilayer Control Architecture}
\label{sec:ch4:architecture}
"""),
P('sep', 'The control architecture is the established multilayer approach of prior',
  end='positions alone.',
  subs=[('studies \\cite{2,10}, shown in\nFig.~\\ref{fig:control_architecture}.',
         'studies \\cite{2}, shown in\nFig.~\\ref{fig:ch4:control_architecture}.')]),
T(r"""
% Ported from: IDETC 2025 (DETC2025-167604), Sec. 2.1, transcribed from the published PDF
This architecture was selected because its modular layered approach
allows for independent development and testing of different components
of the overall architecture. This approach also maximizes system
reusability, enabling systematic comparison of navigation strategies
using standardized components in downstream layers. Additionally, the
architecture enables the same control laws to be applied across different
robot platforms and formations.
"""),
P('sys', 'At each control cycle (10~Hz), each robot reports its position and vector field measurement.'),
P('sys', '\\caption{Hierarchical control architecture: robot, cluster space, and adaptive',
  end='\\end{figure}', back=3,
  subs=[('control_architecture_2.png', 'ch4_control_architecture.png'),
        ('width=0.40\\textwidth', 'width=0.55\\textwidth')]),

T(r"""
\subsection{Robot Controller Layer}
\label{sec:ch4:robot_layer}
"""),
P('sys', "Each robot's controller receives velocity commands and translates them into motion.",
  subs=[('providing the inputs to the estimation framework described in Section~II.',
         'providing the inputs to the estimation framework of Section~\\ref{sec:ch4:estimation}.')]),
T(r"""
% New text
Section~\ref{sec:ch2:dynamics} gives the first-order lag, speed limit, and
stiction floor identified for the Decabots, which the simulations of
later chapters reproduce.

\subsection{Cluster Space Controller Layer}
\label{sec:ch4:cluster_space}

% Ported from: IDETC 2025, Sec. 2.3, transcribed from the published PDF
To maintain a fixed formation while commanding all robots with individual
velocity commands, a cluster space controller was used. Cluster space
control is a technique developed by Dr. Chris Kitts of Santa Clara
University \cite{idc:17}. In this technique, all robot positions are
described geometrically relative to a cluster frame via a set of
kinematic equations.

% New text
It is an operational space control approach in which the multirobot
formation is represented as a virtualized full degree-of-freedom
articulating mechanism.
"""),
P('sys', 'We refer to the group of robots as a cluster: a formation treated as a single virtual rigid body'),
T(r"""
% Ported from: IDETC 2025, Sec. 2.3, transcribed from the published PDF
Cluster space control also provides a closed form method for converting
between cluster-level commands and robot-level commands. This allows for
the user or adaptive navigation layer to specify the behaviour commands
for the entire cluster, while the individual robots are then commanded by
the robot control layer to react accordingly. These conversions happen on
a centralized computer, after the individual robots communicate their
instantaneous position and sensed information.

\subsubsection{Three-Robot SAS Formation}
\label{sec:ch4:sas}
"""),
P('sys', "The cluster space controller maintains the cluster's formation geometry while executing"),
P('sys', 'In this study, we consider a cluster of three robots forming a triangular configuration.',
  subs=[('In this study, we consider', 'Chapters~\\ref{ch:zeroth} and~\\ref{ch:first_order} consider')]),
P('sys', '\\caption{Three-robot triangular formation parameterized by SAS variables',
  end='\\end{figure}', back=3,
  subs=[('SAS_Robots.png', 'ch4_SAS_Robots.png'),
        ('width=0.30\\textwidth', 'width=0.45\\textwidth')]),
P('sys', 'The forward kinematic equations map the robot positions',
  subs=[('The full equations are given in Appendix~B.',
         'The full equations are given in Appendix~\\ref{app:kin3}.')]),

T(r"""
\subsubsection{Six-Robot Pentagon-Plus-Center Formation}
\label{sec:ch4:pentagon}
"""),
P('sep', 'Cluster space control treats a multirobot formation as a virtual rigid',
  end='the layer above through the forward kinematics.',
  subs=[('(Fig.~\\ref{fig:pentagon})', '(Fig.~\\ref{fig:ch4:pentagon})'),
        ('inverse Jacobian $\\mathbf{J}^{-1}$', 'inverse Jacobian $\\mathbf{J}_c^{-1}$'),
        ('through the forward kinematics.', 'through the forward kinematics (Appendix~\\ref{app:kin6}).')]),
P('sep', '\\caption{Pentagon-plus-center formation. Robot 1 sits at the centroid',
  end='\\end{figure}', back=3,
  subs=[('figures/pentagon_formation.png', 'ch4_pentagon_formation.png'),
        ('width=0.5\\columnwidth', 'width=0.45\\textwidth')]),

T(r"""
\subsection{Adaptive Navigation Layer}
\label{sec:ch4:an_layer}

% Ported from: IDETC 2025, Sec. 2.4, transcribed from the published PDF
The adaptive navigation layer integrates the cluster shape policy,
feature estimator, and control law, determining the desired actions of
the entire cluster in cluster space variables.

% New text
The feature estimator is the part that changes from rung to rung.
Section~\ref{sec:ch4:estimation} describes it in general, and each of
Chapters~\ref{ch:zeroth} to~\ref{ch:second_order} gives the control law
that consumes it.

\section{Local Polynomial Field Estimation}
\label{sec:ch4:estimation}

% New text. Generalizes the first-order fit of the critical-points paper
% and the second-order fit of Draft 11 to order k; checked 2026-09-29.
Each estimator fits a polynomial to the velocity samples the robots take
at one instant. The order of that polynomial decides what the cluster can
learn about the field.

\subsection{Monomial Basis and Formation Matrix}
\label{sec:ch4:basis}

Let $\boldsymbol{\phi}_k(\mathbf{p})$ collect the monomials $x^a y^b$
with $a + b \leq k$, each scaled by $1/(a!\,b!)$, with coordinates
measured from the cluster centroid. The scaling makes each fitted
coefficient a Taylor derivative of the field at the centroid. The basis
has
\begin{equation}
    n_k = \tfrac{1}{2}(k+1)(k+2)
    \label{eq:ch4:nk}
\end{equation}
entries. Robot $i$ at $\mathbf{p}_i$ contributes the row
$\boldsymbol{\phi}_k(\mathbf{p}_i)^{\top}$ to the formation matrix
$\boldsymbol{\Phi}_k$, and each velocity component is fit separately,
\begin{equation}
    \boldsymbol{\Phi}_k\,\hat{\boldsymbol{\theta}}_u = \mathbf{m}_u, \qquad
    \boldsymbol{\Phi}_k\,\hat{\boldsymbol{\theta}}_v = \mathbf{m}_v ,
    \label{eq:ch4:fit}
\end{equation}
where $\mathbf{m}_u$ and $\mathbf{m}_v$ stack the robots' readings. The
first-order fit of Chapter~\ref{ch:first_order} is the case $k = 1$, and
the second-order fit of Chapter~\ref{ch:second_order} is $k = 2$. The two
chapters order the columns differently, which changes nothing in the fit.

\subsection{Minimality and Degeneracy}
\label{sec:ch4:minimality}

Each robot adds one equation per component, so a unique fit needs at
least $n_k$ robots. The first order needs three and the second order
needs six. With exactly $n_k$ robots $\boldsymbol{\Phi}_k$ is square. It
is singular exactly when a nonzero coefficient vector $\mathbf{c}$
satisfies $\boldsymbol{\Phi}_k\mathbf{c} = \mathbf{0}$, which says that
the polynomial $\mathbf{c}^{\top}\boldsymbol{\phi}_k(\mathbf{p})$
vanishes at every robot. The fit therefore fails exactly when the robots
lie on a common algebraic curve of degree $k$. For $k = 1$ that curve is
a line, and for $k = 2$ it is a conic. With more than $n_k$ robots the
system is overdetermined and is solved by least squares.

\subsection{Formation Size}
\label{sec:ch4:size}

Formation size enters through the columns of $\boldsymbol{\Phi}_k$. If a
formation keeps its shape and is scaled to radius $\rho$, a monomial of
degree $q$ scales as $\rho^{\,q}$, so the noise reaching a coefficient of
order $q$ scales as $\rho^{-q}$ (Section~\ref{sec:ch8:noise_order}).
Truncation error runs the other way. The neglected terms are of degree
$k + 1$, so the bias they leave grows with $\rho$. A larger formation
therefore trades truncation bias for noise suppression.

\section{The Fit-Order Ladder}
\label{sec:ch4:ladder}

Table~\ref{tab:ch4:ladder} organizes this dissertation by fit order. Each
rung fixes a failure of the one below it. The zeroth-order primitives of
Chapter~\ref{ch:zeroth} use the mean sensed vector, or the gradient of
the sensed magnitude, and never assemble the velocity gradient. They
supply a heading, which drifts on an orbit. The first-order fit of
Chapter~\ref{ch:first_order} locates and classifies a critical point, but
it gives the velocity gradient at one point only, so it cannot follow a
curve. The second-order fit of Chapter~\ref{ch:second_order} gives the
velocity gradient as a field over the formation, which is enough to
acquire and ride a separatrix. A third-order fit from ten robots would
recover the third derivatives the second-order trackers lack, and
Chapter~\ref{ch:limitations} leaves it as future work.

\begin{table}[htbp]
\centering
\caption{The fit-order ladder. The minimum robot count is $n_k$ of
(\ref{eq:ch4:nk}), and the degeneracy is the curve on which that many
robots cannot support the fit.}
\label{tab:ch4:ladder}
\begin{tabular}{p{1.3cm} p{1.4cm} p{2.0cm} p{4.6cm} p{1.3cm}}
\hline
Fit order & Minimum robots & Degeneracy & Features reachable & Chapter \\
\hline
Zeroth & 1 & none & heading only, which drifts on orbits & \ref{ch:zeroth} \\
First & 3 & common line & critical points, with location, type, and orbit & \ref{ch:first_order} \\
Second & 6 & common conic & separatrices, OECS, flow saddles as minima & \ref{ch:second_order} \\
Third & 10 & common cubic & third derivatives, the true $\mathbf{H}_D$ and transverse curvature & \ref{ch:limitations} \\
\hline
\end{tabular}
\end{table}
"""),
]

write('ch04_estimation_framework.tex', parts)
