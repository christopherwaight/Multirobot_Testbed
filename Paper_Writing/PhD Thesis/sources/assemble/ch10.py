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
simulation.
"""),
P('sys', 'This paper presents a distributed framework for multirobot navigation',
  subs=[('This paper presents', 'Chapter~\\ref{ch:first_order} presented')]),
P('sep', 'This paper has shown that a second-order fit of the local flow, from six',
  end='map, no forecast, and no assumption about the structure\'s identity.',
  subs=[('This paper has shown', 'Chapter~\\ref{ch:second_order} showed')]),
P('sep', 'The experiments give an operational selection rule.', end='seed step.'),

T(r"""
% New text. Primitive library, moved from Chapter 8 on 2026-09-30.
Table~\ref{tab:ch10:library} collects every primitive in this
dissertation with its fit order, the smallest cluster it was run with,
the quantity it steers on, its stability result, and how it was verified.

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
Attraction & 1 & 3 & $\hat{\mathbf{p}}^* = -\hat{\mathbf{J}}^{-1}\hat{\mathbf{h}}$ & exponential convergence to a stationary estimate, $\tau = 1/k$ & simulation, 8 fields; hardware, 157 trials \\
Orbital & 1 & 3 & $\hat{\mathbf{p}}^*$ with radial and tangential terms & radial error $\dot{e}_r = -k_r e_r$ & simulation, 8 fields; hardware, 12 trials \\
$D$ tracker & 2 & 6 & $\hat{D}$ and $\hat{\mathbf{H}}_{D,0}$ & transverse practical stability, $\limsup|n| \leq \delta/a_\perp$ & simulation, double gyre and Santa Barbara Channel \\
$s_1$ tracker & 2 & 6 & $\hat{s}_1$ and $\nabla\hat{s}_1$ & same transverse bound, frame equivariant past its seed step & simulation, double gyre and Santa Barbara Channel \\
\hline
\end{tabular}
\end{table}

\section{Final Thoughts}
\label{sec:ch10:final}

% New text
The primitives in this dissertation are admittedly simple, reactive, and
minimal. Each uses the smallest cluster its fit order allows and only the
measurements of the current instant. Even in this form, however, they
show that the order of a cooperative fit sets which features of a flow a
cluster can reach, from a heading, to a critical point, to a separatrix.
Ongoing and future work targets a) hardware trials of the second-order
trackers on the Decabot testbed, b) control that accounts for a flow that
advects the robots, and c) a third-order fit from ten robots.
"""),
]

write('ch10_conclusion.tex', parts)
