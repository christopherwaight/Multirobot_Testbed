import port
from port import P, T, write
port.PRE = 'ch7'

# 2026-11-18: re-anchored to Draft_13. The relation between the fields, the
# OECS definition, and the double-gyre example live in Chapter 3; the
# architecture and pentagon formation live in Chapter 4; limitations in
# Chapter 9. Draft_13 labels (sec:minimality, eq:d_law, ...) gain the ch7
# prefix automatically, so the headings below reuse its label names.

DG = ('Section~\\ref{sec:dg_example}', 'Section~\\ref{sec:ch3:dg_example}')
DGF = ('(Fig.~\\ref{fig:dg_fields}(b))', '(Fig.~\\ref{fig:ch3:dg_fields}(b))')

def FIG(cap, old, new, width, star=False, subs=()):
    env = 'figure*' if star else 'figure'
    s = [(f'figures/{old}', new)] + list(subs)
    if star:
        s += [('\\begin{figure*}', '\\begin{figure}'), ('\\end{figure*}', '\\end{figure}')]
    return P('sep', cap, end='\\end{' + env + '}', back=3, subs=s, width=width)

parts = [
T(r"""
% ===================================================================
% CHAPTER 7: SECOND-ORDER PRIMITIVES (Separatrix / OW paper, Draft 13)
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
P('sep', 'A vector field also carries well-defined curves.'),
P('sep', 'Coherent structures are typically identified through one of two primary methods.',
  end='are a matter of ongoing research'),
P('sep', 'Mapping these curves lets a team target its search'),
P('sep', 'Recent work has begun applying AN techniques with MRS to navigate along coherent structures.',
  end="do not acquire a structure if they aren't."),
P('sep', 'This paper uses an Eulerian specification of the flow field and shows that a second-order',
  end='\\end{itemize}',
  subs=[('This paper uses', 'This chapter uses'),
        ('The contributions are', 'The chapter contributes')]),
P('sep', 'Both primitives run on a simulated double gyre and on measured Santa',
  end='Barbara Channel radar currents.'),

T(r"""
\section{Six-Robot Quadratic Estimation}
\label{sec:ch7:estimation}
"""),
P('sep', 'The critical point estimator of \\cite{10} fits a plane to each',
  end='the vector components sampled by the $i$-th robot.',
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
  end='since both readings move by the same $\\boldsymbol{\\xi}_i$.'),

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
  end='which is constant over the footprint.'),
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
P('sep', 'In the robot controller layer, each robot follows its velocity command',
  end='carries a speed limit and a stiction floor.'),
T(r"""
% New text. The six-robot simulations run the legacy Omnibot model
% (CLAUDE.md, 2026-11-16: Omnibot defaults stay legacy), not the refit.
The simulations of this chapter use this earlier momentum model, not the
refit of Section~\ref{sec:ch2:dynamics}.
"""),
P('sep', "A Python simulation integrates each robot's position by explicit Euler at",
  end='fixed across all experiments unless noted.'),
P('sep', '\\caption{Operating points. The double-gyre column is non-dimensional,',
  end='\\end{table}', back=1),

T(r"""
\section{The $D$ Tracker}
\label{sec:ch7:sep_controller}
"""),
P('sep', 'The $D$ tracker is built for a mountain pass in $D$.',
  end='which holds intrinsically unless inverted by', extra=1,
  subs=[DGF,
        ('critical point test of\n\\cite{10}.', 'critical point test of\nChapter~\\ref{ch:first_order}.'),
        ('the attraction law of \\cite{10},', 'the attraction law of Chapter~\\ref{ch:first_order},')]),

T(r"""
\section{The $s_1$ Tracker}
\label{sec:ch7:s1_controller}
"""),
P('sep', 'The fitted $\\hat{s}_1$ is concave for every field and formation',
  end='tangent ride to carry the cluster across alone.',
  subs=[('formation\n(Section~\\ref{sec:surrogates}). It never',
         'formation\n(Section~\\ref{sec:ch7:s1_field}). It never'),
        ('strain eigenvector (Section~\\ref{sec:surrogates}),',
         'strain eigenvector (Section~\\ref{sec:ch3:oecs}),'),
        ('figures/s1_channels.png', 'ch7_s1_channels.png')],
  width='0.95\\textwidth'),

T(r"""
\section{Double-Gyre Benchmark}
\label{sec:ch7:dg_bench}

\subsection{Setup}
"""),
P('sep', 'The double-gyre experiments use the steady field of',
  end='after their first straddle.',
  subs=[DG]),
T(r"""
\subsection{Clean Runs}
\label{sec:ch7:disc_clean}
"""),
FIG('\\caption{$s_1$ tracker versus $D$ tracker from seven matched starts',
    'traverse_vs_logic_c_seven.png', 'ch7_traverse_vs_logic_c_seven.png', '0.9\\textwidth'),
FIG('\\caption{The six robots from start S1 (star) under both trackers,',
    'separatrix_formation_seven.png', 'ch7_separatrix_formation_seven.png', '0.9\\textwidth'),
P('sep', 'Without noise, both trackers acquire the separatrix, traverse it in the',
  end='within 0.004 to 0.006.'),
T(r"""
\subsection{Behavior Under Noise}
\label{sec:ch7:disc_noise}
"""),
FIG('\\caption{Downstream-saddle success of both trackers from the seven starts of',
    'noise_seven_starts_col.png', 'ch7_noise_seven_starts.png', '0.65\\textwidth'),
P('sep', 'Both primitives sweep measurement and position noise separately from the',
  end='where it is definite.'),
T(r"""
\subsection{Behavior Under a Rotating Observer}
\label{sec:ch7:rotating}
"""),
FIG('\\caption{Rotating-observer trial from S1 (star)',
    'objectivity_rotating.png', 'ch7_objectivity_rotating.png', '0.9\\textwidth'),
P('sep', 'A cluster can sense the water without error and still see the flow from a',
  end='its capture test passes in only 1\\% of', extra=1,
  subs=[('(\\ref{eq:D_not_objective})', '(\\ref{eq:ch3:D_not_objective})')]),

T(r"""
\section{Santa Barbara Channel Trial}
\label{sec:ch7:ocean}

\subsection{Setup}
\label{sec:ch7:ocean_sim}
"""),
P('sep', 'The ocean experiment uses the public HFRNet U.S.~West Coast surface-current',
  end='which branch it meets.'),
T(r"""
\subsection{Results}
\label{sec:ch7:disc_ocean}
"""),
FIG('\\caption{$D$ and $s_1$ tracker paths from a single shared start at',
    'ocean_progression_2km.png', 'ch7_ocean_progression_2km.png', '\\textwidth', star=True),
P('sep', 'Both trackers follow the dominant ridge of the forward FTLE field from the',
  end='remained at sea at', extra=1),

T(r"""
\section{Summary}
\label{sec:ch7:summary}
"""),
P('sep', 'This paper has shown that six robots sampling at once can fit a local',
  end='should carry the $s_1$ tracker.',
  subs=[('This paper has shown', 'This chapter has shown')]),
]

write('ch07_second_order.tex', parts)
