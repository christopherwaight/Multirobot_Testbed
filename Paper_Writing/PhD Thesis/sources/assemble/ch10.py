import port
from port import P, T, write
port.PRE = 'ch10'

parts = [
T(r"""
% ===================================================================
% CHAPTER 10: CONCLUSION
% ===================================================================
\chapter{Conclusion}
\label{ch:conclusion}

\section{Summary of Contributions}
\label{sec:ch10:summary}

% Ported from: IDETC 2025 (DETC2025-167604), Sec. 7, transcribed from the published PDF
The testbed of Chapter~\ref{ch:tools} overcame previous testbed
limitations through HSV color space representation and neural network
calibration, achieving $R^2$ accuracies of 0.96 for direction and 0.91 for
magnitude while maintaining precise formation control of the robot
cluster. On it, the vector-sum and vector-to-scalar primitives ran in
continuous motion, confirming behaviors previously demonstrated only in
simulation. After preventive maintenance, a pooled recalibration read
direction to within 6.1$^\circ$ and magnitude to within 3.8\% for the
trained robots (Section~\ref{sec:ch2:calibration}).
"""),
P('sys', 'This paper presents a framework for multirobot navigation in vector fields containing critical points.',
  end='reproduced the saddle\'s attraction error and about half of the orbit fluctuation.',
  subs=[('This paper presents', 'Chapter~\\ref{ch:first_order} presented')]),
P('sep', 'This paper has shown that six robots sampling at once can fit a local',
  end='can trust its own rotation-to-strain ratio.',
  subs=[('This paper has shown that six robots', 'Chapter~\\ref{ch:second_order} showed, in simulation, that six robots')]),

T(r"""
% New text. Primitive library, moved from Chapter 8 on 2026-09-30.
Table~\ref{tab:ch10:library} collects every primitive in this
dissertation with its fit order, the smallest cluster it was run with,
the quantity it steers on, its stability result, how it was verified, and
the lower rungs it calls on. The robot counts are the smallest clusters
with which each primitive was run and, for the fits, the minimum for an
unrestricted fit of that order from one set of instantaneous measurements;
they are not minimums for navigation in general.

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

% New text
The primitives in this dissertation are admittedly simple, reactive, and
minimal. Each uses the smallest cluster its fit order allows and only the
measurements of the current instant. Even in this form, however, they
show that the order of the instantaneous local fit that a cluster makes
determines the field information it has, and so the features of a flow that it can
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
"""),
]

write('ch10_conclusion.tex', parts)
