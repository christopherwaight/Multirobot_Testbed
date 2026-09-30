import port
from port import P, T, write
port.PRE = 'ch6'

FIG = lambda cap, old, new, w: P('sys', cap, end='\\end{figure}', back=3,
                                 subs=[(old, new), (w[0], w[1])])

parts = [
T(r"""
% ===================================================================
% CHAPTER 6: FIRST-ORDER PRIMITIVES (critical-points paper)
% ===================================================================
\chapter{First-Order Primitives: Critical Points}
\label{ch:first_order}

\section{Introduction}
\label{sec:ch6:intro}

% New text
Chapter~\ref{ch:zeroth} showed that direction-only primitives reach a
feature only when the field itself carries the cluster there. Orbiting a
fixed vortex with the vector-sum primitive drifted steadily outward,
because nothing in a heading measures the distance to the center. This
chapter moves one rung up the ladder of Section~\ref{sec:ch4:ladder}.
"""),
P('sys', 'Recovering those coordinates is harder than following a heading'),
P('sys', 'The implementation problem is that the local field must be reconstructed'),
P('sys', 'This paper introduces a distributed cooperative estimation strategy',
  subs=[('This paper introduces', 'This chapter develops')]),

T(r"""
\section{Critical Point Estimation from Three Robots}
\label{sec:ch6:estimation}
"""),
P('sys', 'Given a cluster of three robots sampling a planar vector field',
  end='Thus, provided', extra=3),
P('sys', 'The eigenvalues of the Jacobian can be used to identify the type',
  subs=[('(Table~\\ref{tab:field_summary})', '(Table~\\ref{tab:ch3:field_summary})')]),
P('sys', "If the true field is affine over the cluster's spatial extent"),

T(r"""
\subsection{Minimum Robot Requirements}
\label{sec:ch6:minimum}
"""),
P('sys', 'each contain three coefficients. From', end='Three non-collinear robots are also sufficient'),

T(r"""
\section{Critical Point Classification}
\label{sec:ch6:classification}

% New text. Analysis: experiments/classify_from_hardware_logs.py, results
% in experiments/outputs/classify_hw/summary.json (VF_Robot), run
% 2026-09-29. The author asked for these numbers to be written in.
The same fit that locates the critical point also names its type. The
eigenvalues of $\hat{\mathbf{J}}$ place it in one of the classes of
Table~\ref{tab:ch3:field_summary}, at no extra sampling cost. The hardware
trials of Section~\ref{sec:ch6:attraction_hw} were designed to test
location, but every control cycle logged the positions and sensed
readings needed to recompute $\hat{\mathbf{J}}$. Recomputing it offline
reproduces the controller's logged critical point estimates to within
$5\times10^{-7}$~m, so the Jacobians classified here are the ones the
controller computed. The logs hold 78 saddle trials and 80 vortex trials
of 101 cycles each, one vortex trial more than the location statistics
of Section~\ref{sec:ch6:attraction_hw} use.
%% TODO(reconcile): the logs contain 80 vortex runs; the critical-points
%% paper reports 79 (157 total), and its bias and precision use that count.

Each estimate was classified as a saddle when $\det\hat{\mathbf{J}} < 0$,
a node when the eigenvalues are real and of one sign, and a rotational
type when they form a complex pair $\alpha \pm i\omega$. A measured center
never has $\alpha$ exactly zero, so a complex pair was labeled a center
when $|\alpha| < 0.2\,\omega$ and a spiral otherwise. That threshold is a
choice, and Table~\ref{tab:ch6:classification_hw} also gives the results
at 0.1 and 0.3.

\begin{table}[htbp]
\centering
\caption{Offline classification of the hardware Jacobian estimates. The
terminal window is the final 2~s of each trial, and the trial vote is the
majority label over that window.}
\label{tab:ch6:classification_hw}
\begin{tabular}{lcc}
\hline
 & Saddle map & Vortex map \\
\hline
Trials & 78 & 80 \\
Class correct (saddle, rotational, node), all cycles & 99.9\% & 99.9\% \\
Class correct, terminal window & 100\% & 100\% \\
Class correct, trial vote & 78/78 & 80/80 \\
Exact type, terminal window, threshold 0.1 & 100\% & 71\% \\
Exact type, terminal window, threshold 0.2 & 100\% & 89\% \\
Exact type, terminal window, threshold 0.3 & 100\% & 99\% \\
\hline
\end{tabular}
\end{table}

The estimate named the class of both fields correctly in effectively
every cycle, including cycles more than 0.8~m from the critical point.
Every vortex error is a stable spiral with a small negative real part,
and none is a saddle or a node. A small damping in the printed map or a
small bias in the fit is enough to turn a center into a weak spiral, and
the data cannot separate the two.

Classification carries one condition that location does not. A
reflection of the sensed vectors reverses the sign of
$\det\hat{\mathbf{J}}$, which exchanges saddles with nodes but leaves the
estimated critical point where it was. The two printed maps were placed
under different frame conventions, including reflections, rotations, and
camera mirroring. The results above use the registration from the field
reconstructions of Section~\ref{sec:ch2:reconstruction}, which negates
the sensed $v$ component for the saddle map only. The controller applied
no registration, since it used the estimate only for location, and its
unregistered Jacobian read the saddle map as a stable node or spiral
throughout the terminal window. A deployment that reports feature type
therefore needs its sensor frame registered to its navigation frame.

\section{Attraction Primitive}
\label{sec:ch6:attraction}
"""),
P('sys', 'We develop two control primitives that drive the cluster centroid',
  subs=[('described in Section~III', 'of Chapter~\\ref{ch:estimation}')]),
T(r"""
\subsection{Definition}
\label{sec:ch6:attraction_def}
"""),
P('sys', 'To navigate the cluster center $\\mathbf{p}_c$ to the critical point',
  end='provided the estimate is stationary.'),

T(r"""
\subsection{Simulation Verification}
\label{sec:ch6:attraction_sim}
"""),
P('sys', 'A Python-based simulation implementing the three-layer control architecture',
  subs=[('described in Section~III', 'of Chapter~\\ref{ch:estimation}'),
        ('primitives in Section~II.', 'primitives of this chapter.'),
        ('(Section~V)', '(Section~\\ref{sec:ch6:attraction_hw})')]),
P('sys', 'Six noise-free analytical fields, one for each of the canonical',
  subs=[('listed in Table~\\ref{tab:field_summary}', 'listed in Table~\\ref{tab:ch3:field_summary}'),
        ('provided in Appendix~A)', 'provided in Appendix~\\ref{app:fields})'),
        ('Further details on the reconstructed fields are provided in Section~V.',
         'Section~\\ref{sec:ch2:reconstruction} describes how the reconstructed fields were built.')]),
P('sys', 'To test convergence to critical points, the following simulation experiments'),
FIG('\\caption{Successful navigation to critical points from random', 'figure_5.png',
    'ch6_figure_5.png', ('width=0.44\\textwidth', 'width=0.75\\textwidth')),
P('sys', 'The cluster converged to the critical point in 100\\% of the 1000 trials'),
P('sys', '\\caption{Simulation results across eight field types', end='\\end{table}', back=3),

T(r"""
\subsection{Hardware Verification}
\label{sec:ch6:attraction_hw}
"""),
P('sys', 'Hardware validation focused on two representative field types',
  end='the saddle field validates that the framework operates reliably'),
P('sys', 'For each field, eight starting locations were selected around the perimeter'),
P('sys', 'Following the test plan above, 79 vortex trials and 78 saddle trials',
  subs=[('matching the simulation metrics in Section~IV.',
         'matching the simulation metrics of Section~\\ref{sec:ch6:attraction_sim}.')]),
P('sys', 'The cluster converged to the critical point in 100\\% of the 157 hardware trials'),
P('sys', 'shows convergence trajectories from all eight starting positions in the vortex field'),
FIG('\\caption{Convergence trajectories from eight starting positions', 'allrunsadjusted.png',
    'ch6_allrunsadjusted.png', ('width=0.30\\textwidth', 'width=0.55\\textwidth')),
FIG('\\caption{Distance to center over time, with mean trajectory', 'fig9.png',
    'ch6_fig9.png', ('width=0.42\\textwidth', 'width=0.7\\textwidth')),
P('sys', '\\caption{Hardware experimental results. Saddle: 78 runs.', end='\\end{table}', back=3),

T(r"""
\section{Orbital Primitive}
\label{sec:ch6:orbital}

\subsection{Definition}
\label{sec:ch6:orbital_def}
"""),
P('sys', 'To maintain a circular orbit of radius $r_d$ around',
  end='producing steady circular motion.'),

T(r"""
\subsection{Simulation Verification}
\label{sec:ch6:orbital_sim}
"""),
P('sys', 'Orbital control was evaluated on the same eight fields at different commanded radii'),
P('sys', '\\caption{Simulated orbital control across eight field types', end='\\end{table}', back=2),
P('sys', 'presents the orbital tracking results. On the six noise-free fields'),

T(r"""
\subsection{Hardware Verification}
\label{sec:ch6:orbital_hw}
"""),
P('sys', 'Additionally, each field was used to test the critical point orbiting controller'),
P('sys', 'Orbital control was evaluated on both fields at commanded radii from'),
P('sys', 'In the vortex field, mean radial errors decreased from 0.246'),
FIG('\\caption{Orbital trajectory comparison at a commanded radius of 0.40', 'orbit040_analytical_vs_real.png',
    'ch6_orbit040_analytical_vs_real.png', ('width=0.85\\columnwidth', 'width=0.75\\textwidth')),
P('sys', '\\caption{Hardware orbital control results.', end='\\end{table}', back=2),

T(r"""
\section{Comparison with Zeroth-Order Primitives}
\label{sec:ch6:comparison}
"""),
P('sys', 'The modular architecture allows navigation controllers to be interchanged'),
P('sys', 'We evaluate orbital control by comparing our orbital controller against two alternatives'),
P('sys', 'both alternative primitives exhibit outward spiral drift',
  subs=[('has been confirmed in hardware in~\\cite{22}.',
         'has been confirmed in hardware in~\\cite{22} and Chapter~\\ref{ch:zeroth}.')]),
FIG('\\caption{Orbital trajectories: vector-sum primitive versus', 'orbit_demonstration.png',
    'ch6_orbit_demonstration.png', ('width=0.85\\columnwidth', 'width=0.75\\textwidth')),

T(r"""
\section{Error Analysis}
\label{sec:ch6:error}
"""),
P('sys', 'Estimation and control errors arise from four sources',
  end='Across all four error sources',
  subs=[('the calibration errors of Section V-A give',
         'the calibration errors of Section~\\ref{sec:ch2:calibration} give')]),

T(r"""
\section{Summary}
\label{sec:ch6:summary}
"""),
P('sys', 'This paper presents a distributed framework for multirobot navigation',
  subs=[('This paper presents', 'This chapter presented'),
        ('uses instantaneous measurements', 'used instantaneous measurements'),
        ('supporting an attraction', 'and supported an attraction')]),
T(r"""
% New text
Offline, the same hardware estimates also named the type of both printed
fields in every trial by majority vote, once each map's sensor frame was
registered to the motion-capture frame.
"""),
]

write('ch06_first_order.tex', parts)
