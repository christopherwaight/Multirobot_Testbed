import port
from port import P, T, write
port.PRE = 'ch7'

DG = ('(Section~\\ref{sec:dg_example})', '(Section~\\ref{sec:ch3:dg_example})')

def FIG(cap, old, new, width, star=False, more=()):
    env = 'figure*' if star else 'figure'
    subs = [(f'figures/{old}', new)] + list(more)
    if star:
        subs += [('\\begin{figure*}', '\\begin{figure}'), ('\\end{figure*}', '\\end{figure}')]
    return P('sep', cap, end='\\end{' + env + '}', back=3, subs=subs, width=width[1].split('=',1)[1])

parts = [
T(r"""
% ===================================================================
% CHAPTER 7: SECOND-ORDER PRIMITIVES (Separatrix / OW paper, Draft 11)
% ===================================================================
\chapter{Second-Order Primitives: Coherent Structures}
\label{ch:second_order}

\section{Introduction}
\label{sec:ch7:intro}

% New text
Chapter~\ref{ch:first_order} located isolated critical points from a
first-order fit. This chapter moves to the second rung of the ladder of
Section~\ref{sec:ch4:ladder}, where the features are curves.
"""),
P('sep', 'Beyond critical points, vector fields have curves that are useful for navigation'),
P('sep', 'Coherent structures are typically identified through two primary methods'),
P('sep', 'New work is being done on using AN techniques with MRS in these scenarios',
  end='does not acquire a structure it does not already occupy.'),
P('sep', 'This paper shows that a second-order fit of the local flow is enough to',
  end='\\end{itemize}',
  subs=[('This paper shows', 'This chapter shows'),
        ('A\nlinear fit, as in \\cite{10}, gives', 'A\nlinear fit, as in Chapter~\\ref{ch:first_order}, gives'),
        ('The contributions are', 'The chapter contributes')]),
P('sep', 'Both primitives run on a simulated double gyre and on measured Santa',
  end='Barbara Channel radar currents.'),

T(r"""
\section{Six-Robot Quadratic Estimation}
\label{sec:ch7:estimation}
"""),
P('sep', 'The critical point estimator of \\cite{10} fits a plane to each',
  end='sampled by robot $i$.',
  subs=[('The critical point estimator of \\cite{10} fits',
         'The critical point estimator of Chapter~\\ref{ch:first_order} fits')]),
T(r"""
\subsection{Minimality}
\label{sec:ch7:minimality}
"""),
P('sep', 'Six robots is the minimum for recovery of the unconstrained local',
  end='the fit here independent of formation heading.'),
T(r"""
\subsection{Estimator Sensitivity}
\label{sec:ch7:sensitivity}
"""),
P('sep', 'The accuracy of the estimator is affected by the formation of the',
  end='\\label{eq:sigma_eff}', extra=1),

T(r"""
\section{Eulerian Surrogates from the Fit}
\label{sec:ch7:surrogates}
"""),
P('sep', 'Lagrangian methods locate coherent structures by integrating particle',
  end='gradient, and the second is the smaller eigenvalue of its symmetric', extra=1,
  subs=[('This paper instead', 'This chapter instead')]),
T(r"""
\subsection{Determinant Field}
\label{sec:ch7:det_field}
"""),
P('sep', 'The fitted coefficients define the velocity gradient at every point of',
  end='cannot recover at any radius.'),
T(r"""
% New text
Section~\ref{sec:ch3:ow} gives the physical reading of $D$, which
separates strain-dominated from rotation-dominated flow.

\subsection{Strain Eigenvalue Field}
\label{sec:ch7:s1_field}
"""),
P('sep', 'The second surrogate comes from the rate-of-strain tensor',
  end='so it is concave over the footprint for every field and formation.'),
T(r"""
% New text
Section~\ref{sec:ch3:oecs} defines the attracting and repelling OECS
that the trenches of $s_1$ mark, and Section~\ref{sec:ch3:relation}
relates the two fields, including how each changes for a rotating
observer.

\section{Six-Robot Control Architecture}
\label{sec:ch7:architecture}

% New text
The cluster runs on the three-layer architecture of
Section~\ref{sec:ch4:architecture}, with the pentagon-plus-center
formation of Section~\ref{sec:ch4:pentagon}.
"""),
P('sep', 'The adaptive navigation layer has three functional blocks. The feature',
  end='layers keep their interfaces.'),
P('sep', 'Each robot follows its velocity command through a first-order lag,',
  end='the two operating points, fixed across all experiments unless noted.',
  subs=[('identified in \\cite{10} on the Decabot', 'identified in Section~\\ref{sec:ch2:dynamics} on the Decabot')]),
P('sep', '\\caption{Operating points. The double-gyre column is non-dimensional,',
  end='\\end{table}', back=1),

T(r"""
\section{The $D$ Tracker}
\label{sec:ch7:sep_controller}
"""),
P('sep', 'On the benchmark the $D$ landscape around the separatrix is a mountain',
  end='never reverses (Appendix~\\ref{app:stability}).',
  subs=[('pass (Fig.~\\ref{fig:dg_fields}(b))', 'pass (Fig.~\\ref{fig:ch3:dg_fields}(b))'),
        DG]),

T(r"""
\section{The $s_1$ Tracker}
\label{sec:ch7:s1_controller}
"""),
P('sep', 'The $s_1$ landscape has the same two minima, and between them the',
  end='the measured flow and is guaranteed only under exact estimates.',
  subs=[('separatrix is again a trench (Fig.~\\ref{fig:dg_fields}(c))',
         'separatrix is again a trench (Fig.~\\ref{fig:ch3:dg_fields}(c))'),
        DG,
        ('(Section~\\ref{sec:surrogates}), squaring rounds',
         '(Section~\\ref{sec:ch3:relation}), squaring rounds'),
        ('figures/s1_channels.png', 'ch7_s1_channels.png')],
  width='0.95\\textwidth'),

T(r"""
\section{Double-Gyre Benchmark}
\label{sec:ch7:dg_bench}

\subsection{Setup}
"""),
P('sep', 'The double-gyre experiments use the steady field of',
  end='in the Monte Carlo sweeps and zero in the noise-free runs.',
  subs=[('Section~\\ref{sec:dg_example}', 'Section~\\ref{sec:ch3:dg_example}')]),
T(r"""
\subsection{Clean Runs}
\label{sec:ch7:disc_clean}
"""),
FIG('\\caption{$s_1$ tracker versus $D$ tracker from six matched starts',
    'traverse_vs_logic_c.png', 'ch7_traverse_vs_logic_c.png',
    ('width=0.85\\columnwidth', 'width=0.8\\textwidth')),
FIG('\\caption{The six robots from start S1 (star) under both trackers,',
    'separatrix_formation.png', 'ch7_separatrix_formation.png',
    ('width=0.85\\columnwidth', 'width=0.8\\textwidth')),
P('sep', 'These runs test acquisition, traversal, and the terminal step for both primitives.',
  end='(Section~\\ref{sec:sep_controller}).',
  subs=[DG]),
T(r"""
\subsection{Behavior Under Noise}
\label{sec:ch7:disc_noise}
"""),
FIG('\\caption{Far-saddle success of both trackers from the straddling start',
    'flip_resolution.png', 'ch7_flip_resolution.png',
    ('width=0.6\\columnwidth', 'width=0.65\\textwidth')),
P('sep', 'Both primitives sweep measurement and position noise separately from one',
  end='transverse curvature (Section~\\ref{sec:s1_controller}).'),
T(r"""
\subsection{Behavior Under a Rotating Observer}
\label{sec:ch7:rotating}
"""),
FIG('\\caption{Objectivity trial, both primitives started from',
    'objectivity_traverser.png', 'ch7_objectivity_traverser.png',
    ('width=0.95\\columnwidth', 'width=0.95\\textwidth'),
    more=[('(\\ref{eq:D_not_objective})', '(\\ref{eq:ch3:D_not_objective})')]),
P('sep', 'Both primitives were run from $(0.05, 0.40)$, just inside the upper',
  end='\\lvert\\hat{\\mathbf{v}}_0^\\top\\mathbf{t}_{\\text{raw}}\\rvert$.',
  subs=[('(\\ref{eq:D_not_objective})', '(\\ref{eq:ch3:D_not_objective})')]),

T(r"""
\section{Santa Barbara Channel Trial}
\label{sec:ch7:ocean}

\subsection{Setup}
\label{sec:ch7:ocean_sim}
"""),
P('sep', 'The ocean experiment uses hourly surface-current frames for the Santa',
  end='residual, so the noise model is validated on the double gyre only.'),
T(r"""
\subsection{Results}
\label{sec:ch7:disc_ocean}
"""),
FIG('\\caption{$D$ and $s_1$ tracker paths from a single shared start at',
    'ocean_progression_2km.png', 'ch7_ocean_progression_2km.png',
    ('width=\\textwidth', 'width=\\textwidth'), star=True),
P('sep', 'The field is divergent and not vorticity-free, so $s_2 = -s_1$ does not',
  end='never engages.'),

T(r"""
\section{Summary}
\label{sec:ch7:summary}
"""),
P('sep', 'This paper has shown that a second-order fit of the local flow, from six',
  end='seed step.',
  subs=[('This paper has shown', 'This chapter has shown')]),
]

write('ch07_second_order.tex', parts)
