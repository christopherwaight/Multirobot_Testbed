import os, port
from port import P, T, write
port.PRE = 'ch9'

IDC = '% Ported from: IDETC 2025 (DETC2025-167604), {}, transcribed from the published PDF'
NOTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corner_note.tex')
corner = '\n'.join('%% ' + l if l.strip() else '%%' for l in open(NOTE).read().rstrip('\n').split('\n'))

parts = [
T(r"""
% ===================================================================
% CHAPTER 9: LIMITATIONS AND FUTURE WORK
% ===================================================================
\chapter{Limitations and Future Work}
\label{ch:limitations}

% New text
Each chapter states the limits of its own claims. This chapter collects
them. The limits of the testbed itself are given in
Section~\ref{sec:ch2:limitations}.

\section{Limitations}
\label{sec:ch9:limitations}

\subsection{First-Order Estimation and Control}
\label{sec:ch9:first_order}
"""),
P('sys', 'Several limitations constrain the current implementation and experimental validation.',
  end='Onboard relative localization, inter-robot communication, and robustness to lower update rates and dropouts were not tested.',
  subs=[('Section~V-C)', 'Section~\\ref{sec:ch6:hw_attraction})'),
        ('the curvature term of (\\ref{eq:bound})', 'the curvature term of (\\ref{eq:ch6:bound})'),
        ('(Section~\\ref{sec:hw_testbed})', '(Section~\\ref{sec:ch2:testbed})')]),

T(r"""
\subsection{Second-Order Estimation and Tracking}
\label{sec:ch9:second_order}
"""),
P('sep', 'Cluster space control is centralized and needs global state and reliable',
  end='untested in a physical flow.'),

T(r"""
\section{Future Work}
\label{sec:ch9:future}

\subsection{Time-Varying Fields}
\label{sec:ch9:time_varying}
"""),
P('sys', 'The estimation framework uses only instantaneous measurements and carries no state between control cycles.'),
T(r"""
% New text
The Santa Barbara Channel trial of Section~\ref{sec:ch7:ocean} ran the
second-order trackers on a field that changes in time, but on one record
and with no analytic guarantee.

\subsection{Advection-Aware Control}
\label{sec:ch9:advection}
"""),
# Kept from the submitted paper: the revision folded the advection point into
# its limitations list, but the thesis keeps the dedicated future-work passage.
P('sys0', 'As noted in Section~III-A, the robot dynamics',
  subs=[('As noted in Section~III-A,', 'As noted in Section~\\ref{sec:ch2:dynamics},'),
        ('(\\ref{eq:continuous})--(\\ref{eq:momentum})', '(\\ref{eq:ch2:continuous})--(\\ref{eq:ch2:momentum})')]),

T(r"""
\subsection{Higher-Order Fits and Formation Reshaping}
\label{sec:ch9:higher_order}
"""),
P('sep', 'Future work targets three extensions. First, a fit from more than six',
  end='the transverse curvature of an $s_1$ trench depend.'),

T(r"""
\subsection{Three-Dimensional Fields}
\label{sec:ch9:3d}

% New text
The estimator of Section~\ref{sec:ch4:estimation} extends to three
dimensions by counting monomials in $x$, $y$, and $z$. A fit of order $k$
then has $\tfrac{1}{6}(k+1)(k+2)(k+3)$ coefficients per component. A
first-order fit needs four robots and fails when they are coplanar. A
second-order fit needs ten robots and fails when they lie on a common
quadric surface. The sensing and formation control of a
three-dimensional cluster are outside this dissertation.

\subsection{Additional Topological Features}
\label{sec:ch9:features}
"""),
P('sys', 'Beyond critical point detection and orbital control, vector fields contain other topological features',
  subs=[('Separatrices connected to saddle points define boundaries between distinct flow regimes and act as Lagrangian coherent structures in steady flows. ', '')]),

T(r"""
\subsection{Testbed Extensions}
\label{sec:ch9:testbed}

""" + IDC.format('Sec. 6') + r"""
The supervised learning calibration technique can be extended to
greyscale sensors for encoding and decoding scalar field monochrome color
maps, creating a unified platform for testing both vector and scalar
field navigation paradigms simultaneously. The modular architecture can
also be enhanced to facilitate controller substitution, enabling
comparative studies between cluster space controllers and alternative
approaches such as swarm-based methods \cite{idc:18}. This modularity
would allow researchers to isolate the effects of different control
strategies while maintaining consistent environmental conditions.
Outdoor testing with autonomous surface vessels \cite{idc:11} and UAVs for
environmental monitoring \cite{idc:20} would examine transferability to
larger scales and real-world deployment challenges.

% -------------------------------------------------------------------
% Parked research note, kept by the author's decision (commit ad76bbf,
% 2026-07-02). Not rendered: it describes the earlier D = 0 contour
% tracker, which this dissertation does not present.
% -------------------------------------------------------------------
""" + corner + "\n"),
]

write('ch09_limitations.tex', parts)
