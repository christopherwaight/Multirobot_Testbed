import port
from port import P, T, write
port.PRE = 'ch7'

# 2026-11-19: re-anchored to Draft_14 (11 pages, refit robot model, new
# Sections IV-E and V-B). The relation between the fields, the OECS
# definition, and the double-gyre closed forms live in Chapter 3; the
# architecture and pentagon formation live in Chapter 4; limitations in
# Chapter 9. Draft_14 labels (sec:minimality, eq:d_law, ...) gain the ch7
# prefix automatically; a reference to a label that now lives in another
# chapter is retargeted with an explicit sub (a two-colon label is left
# alone by the prefixing). \prelim{} numbers stay blue until the 10^4 runs.

DG = ('Section~\\ref{sec:dg_example}', 'Section~\\ref{sec:ch3:dg_example}')
FIT = ('Section~\\ref{sec:dg_example}', 'Section~\\ref{sec:ch7:fit_check}')

def FIG(cap, old, new, width, star=False, subs=()):
    env = 'figure*' if star else 'figure'
    s = [(f'figures/{old}', new)] + list(subs)
    if star:
        s += [('\\begin{figure*}', '\\begin{figure}'), ('\\end{figure*}', '\\end{figure}')]
    return P('sep', cap, end='\\end{' + env + '}', back=3, subs=s, width=width)

parts = [
T(r"""
% ===================================================================
% CHAPTER 7: SECOND-ORDER PRIMITIVES (Separatrix / OW paper, Draft 14)
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
P('sep', 'Coherent structures are typically identified in one of two ways.',
  end='study \\cite{19,20,21,22}.'),
P('sep', 'Mapping these curves tells a search where floating material has gathered',
  end='structures offline; adaptive navigation maps them locally with one small', extra=1),
P('sep', 'Michini et al. travel along a presumed stable or',
  end='\\cite{26,27} or that the center robot starts on it \\cite{29}.'),
P('sep', 'This paper shows that a second-order polynomial fit of the local flow can',
  end='indicates when either trench likely marks a separatrix.',
  subs=[('This paper shows', 'This chapter shows'),
        ('The\ncontributions are as follows.', 'The\nchapter contributes as follows.')]),
P('sep', 'Both primitives run on a simulated double gyre, three variants, and',
  end='radar currents.'),

T(r"""
\section{Six-Robot Quadratic Estimation}
\label{sec:ch7:estimation}
"""),
P('sep', 'The critical point estimator of \\cite{10} fits a plane to each',
  end='and $v_i$ that robot $i$ measures.',
  subs=[('The critical point estimator of \\cite{10} fits',
         'The critical point estimator of Chapter~\\ref{ch:first_order} fits'),
        ('In this work, we extend the linear approximation', 'This chapter extends the linear approximation'),
        ('and the one-half factors make each\none a derivative of the flow at $\\mathbf{p}_c$.',
         'and the one-half factors make each\nfitted coefficient an estimate of a derivative of the flow at $\\mathbf{p}_c$.')]),
T(r"""
\subsection{Minimality}
\label{sec:ch7:minimality}
"""),
P('sep', 'Six robots is the minimum for recovery of the unconstrained local',
  end='noise gains of the fit here independent of heading.',
  subs=[('for a linear fit\n\\cite[Prop.~1]{10}. Each robot',
         'for a linear fit\n(Proposition~\\ref{prop:min}). Each robot')]),
T(r"""
\subsection{Estimator Sensitivity}
\label{sec:ch7:sensitivity}
"""),
P('sep', 'Three things limit the fit: the formation, the noise on each reading,',
  end='local gradient, so unlike measurement noise it depends on the field.'),

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
P('sep', 'The fitted coefficients define the velocity gradient',
  end='which is constant over the footprint.'),
T(r"""
% New text
Section~\ref{sec:ch3:ow} gives the physical reading of $D$, which
separates strain-dominated from rotation-dominated flow.

\subsection{Strain Eigenvalue Field}
\label{sec:ch7:s1_field}
"""),
P('sep', 'The second surrogate comes from the rate-of-strain tensor',
  end='footprint for every field and formation.'),
T(r"""
% New text
Section~\ref{sec:ch3:oecs} defines the attracting and repelling OECS
that the trenches of $s_1$ mark, and Section~\ref{sec:ch3:relation}
relates the two fields, including how each changes for a rotating
observer.

\subsection{Fit Check on the Double Gyre}
\label{sec:ch7:fit_check}

% New text. The closed forms of the double gyre are in Section~\ref{sec:ch3:dg_example};
% these two paragraphs of Draft 14 compare the fit against them.
"""),
P('sep', 'These closed forms also check the fit. Along the separatrix at',
  end='the across-trench eigenvalue stays positive up to', extra=1),

T(r"""
\section{Six-Robot Control Architecture}
\label{sec:ch7:architecture}

% New text
The cluster runs on the three-layer architecture of
Section~\ref{sec:ch4:architecture}, with the pentagon-plus-center
formation of Section~\ref{sec:ch4:pentagon}.
"""),
P('sep', 'This paper fills two blocks of the first layer:',
  end='the heading holds its initial value.',
  subs=[('layer drives each robot. This paper fills two blocks of the first layer:',
         'This chapter fills two blocks of the first layer:'),
        ('Section~II, and the', 'Section~\\ref{sec:ch7:estimation}, and the')]),
P('sep', 'In the robot controller layer, each robot follows its velocity command',
  end='behind every figure are available from the authors on request.',
  subs=[('through the model that \\cite{10} identified on the Decabot indoor',
         'through the model of Section~\\ref{sec:ch2:dynamics}, identified on the Decabot indoor')]),

T(r"""
\section{The $D$ Tracker}
\label{sec:ch7:sep_controller}
"""),
P('sep', 'The $D$ tracker is built for a mountain pass in $D$.',
  end='flow, so neither is objective.',
  subs=[('(Fig.~\\ref{fig:dg_fields}(b))', '(Fig.~\\ref{fig:ch3:dg_fields}(b))'),
        FIT,
        ('critical point test of\n\\cite{10}, which', 'critical point test of\nChapter~\\ref{ch:first_order}, which'),
        ('the attraction law of \\cite{10},', 'the attraction law of Chapter~\\ref{ch:first_order},')]),

T(r"""
\section{The $s_1$ Tracker}
\label{sec:ch7:s1_controller}
"""),
P('sep', 'The fitted $\\hat{s}_1$ is concave for any field and formation, so it',
  end='\\end{figure}',
  subs=[('strain eigenvector (Section~\\ref{sec:surrogates}),',
         'strain eigenvector (Section~\\ref{sec:ch3:oecs}),'),
        ('figures/s1_channels_d14.png', 'ch7_s1_channels.png')],
  width='0.95\\textwidth'),

T(r"""
\section{Closed-Loop Behavior}
\label{sec:ch7:closed_loop}
"""),
P('sep', 'Both trackers pull the cluster onto the trench through one saturated loop',
  end='Table~\\ref{tab:params} collects the operating points.',
  subs=[('of the form \\cite{10} analyzed on the same robots.',
         'of the form analyzed in Section~\\ref{sec:ch6:stability} on the same robots.'),
        ('Section~\\ref{sec:arch}', 'Section~\\ref{sec:ch7:architecture}'),
        ('estimate error\n\\cite{10}. The result', 'estimate error\n(Section~\\ref{sec:ch6:stability}). The result'),
        ('saddles (Section~\\ref{sec:dg_example}). The $s_1$', 'saddles (Section~\\ref{sec:ch7:fit_check}). The $s_1$'),
        ('attraction law of \\cite{10}, so the cluster settles',
         'attraction law of Chapter~\\ref{ch:first_order}, so the cluster settles')]),
P('sep', '\\caption{Operating points. The double-gyre column is non-dimensional,',
  end='\\end{table}', back=1),

T(r"""
\section{Double-Gyre Benchmark}
\label{sec:ch7:dg_bench}

\subsection{Setup}
"""),
P('sep', 'The double-gyre experiments use the steady field of',
  end='of trials that succeed and never lose the straddle once they have it.',
  subs=[DG]),
T(r"""
\subsection{Clean Runs}
\label{sec:ch7:disc_clean}
"""),
FIG('\\caption{$s_1$ tracker versus $D$ tracker from seven matched starts',
    'traverse_vs_logic_c_seven_d14.png', 'ch7_traverse_vs_logic_c_seven.png', '0.9\\textwidth'),
FIG('\\caption{The six robots from start S1 (star) under both trackers,',
    'separatrix_formation_seven_d14.png', 'ch7_separatrix_formation_seven.png', '0.9\\textwidth'),
P('sep', 'Without noise, both trackers acquire the separatrix, traverse it in the',
  end='against under \\rev{0.001} for the $D$', extra=1),
T(r"""
\subsection{Behavior Under Noise}
\label{sec:ch7:disc_noise}
"""),
FIG('\\caption{Downstream-saddle success of both trackers from the seven starts of',
    'noise_seven_starts_col_d14.png', 'ch7_noise_seven_starts.png', '0.65\\textwidth'),
P('sep', 'Both primitives sweep measurement and position noise separately from the',
  end='and \\prelim{84}\\% of trials without changing success.',
  subs=[FIT]),
T(r"""
\subsection{Behavior Under a Rotating Observer}
\label{sec:ch7:rotating}
"""),
FIG('\\caption{Rotating-observer trial from S1 (star)',
    'objectivity_rotating_d14.png', 'ch7_objectivity_rotating.png', '0.9\\textwidth'),
P('sep', 'A cluster can sense the water without error and still see the flow from a',
  end='it ends \\rev{0.138} away.',
  subs=[('(\\ref{eq:D_not_objective})', '(\\ref{eq:ch3:D_not_objective})')]),
T(r"""
\subsection{Beyond the Steady Gyre}
\label{sec:ch7:beyond}
"""),
FIG('\\caption{Each tracker rides a trench of its own field.',
    'beyond_steady_d14.png', 'ch7_beyond_steady.png', '\\textwidth', star=True),
P('sep', '\\caption{Noise-free outcomes beyond the steady gyre', end='\\end{table}', back=1),
P('sep', 'The steady double gyre is kind to both trackers.',
  end='its heading is referenced to an inertial frame.',
  subs=[('of (\\ref{eq:dg_u}), which stays steady', 'of (\\ref{eq:ch3:dg_u}), which stays steady')]),

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
  end='rotation instead of noise, and the ratio $|\\omega|/2r$ flags where it', extra=1),

T(r"""
\section{Summary}
\label{sec:ch7:summary}
"""),
P('sep', 'This paper has shown that six robots sampling at once can fit a local',
  end='can trust its own rotation-to-strain ratio.',
  subs=[('This paper has shown', 'This chapter has shown')]),
]

write('ch07_second_order.tex', parts)
