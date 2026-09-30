import port
from port import P, T, write
port.PRE = 'ch3'

parts = [
T(r"""
% ===================================================================
% CHAPTER 3: VECTOR FIELD PRELIMINARIES
% ===================================================================
\chapter{Vector Field Preliminaries}
\label{ch:preliminaries}

% New text
The features a cluster can navigate to are properties of the velocity
field and its derivatives. This chapter defines them on the true field.
Chapters~\ref{ch:first_order} and~\ref{ch:second_order} read the same
quantities from local fits.

\section{Planar Vector Fields}
\label{sec:ch3:fields}
"""),
P('sys', 'In its most general form, a vector field assigns a vector'),

T(r"""
\section{Critical Points and Their Classification}
\label{sec:ch3:critical_points}
"""),
P('sys', 'Unlike scalar fields where $m=1$, vector fields can exhibit features'),
P('sys', 'Near a critical point, the field is well approximated by its linearization'),
P('sys', 'Eigenvalue analysis can also identify degenerate scenarios',
  subs=[('through the estimation framework we present in the following section',
         'through the estimation framework of Chapter~\\ref{ch:first_order}')]),
P('sys', '\\caption{Six linear vector field types with critical points.}', end='\\end{figure}', back=3,
  subs=[('six_vector_fields.png', 'ch3_six_vector_fields.png'),
        ('width=0.35\\textwidth', 'width=0.6\\textwidth')]),
P('sys', '\\caption{Classification of critical points by Jacobian eigenvalue structure',
  end='\\end{table}', back=2),

T(r"""
\section{Rotation, Strain, and the Okubo-Weiss Partition}
\label{sec:ch3:ow}

% New text
A critical point is one place where the field vanishes. Most of a flow
has no critical point, and there the velocity gradient still says how
nearby fluid moves relative to the point. The gradient
$\mathbf{J} = \nabla\mathbf{v}$ splits into a symmetric part
$\mathbf{S} = \tfrac{1}{2}(\mathbf{J} + \mathbf{J}^{\top})$, the
rate-of-strain tensor, and an antisymmetric part
$\mathbf{W} = \mathbf{J} - \mathbf{S}$, the spin, which carries the
vorticity $\omega = \partial v/\partial x - \partial u/\partial y$. The
eigenvalues $s_1 \leq s_2$ of $\mathbf{S}$ are the principal strain
rates, and their eigenvectors $\mathbf{e}_1$ and $\mathbf{e}_2$ are the
compression and stretching directions. The determinant
$D = \det\mathbf{J}$ combines both parts.
"""),
P('sep', 'For incompressible flow the determinant is a negative multiple of the',
  end='inside the strain-dominated regions.'),

T(r"""
\section{Objective Eulerian Coherent Structures}
\label{sec:ch3:oecs}

% New text
Coherent structures are the curves that organize transport in a flow.
Lagrangian coherent structures are found by integrating particle
trajectories over a time window. This dissertation uses the
instantaneous, Eulerian kind, read from the strain tensor at one time.
"""),
P('sep', 'Serra and Haller define objective Eulerian coherent structures (OECS)',
  end='structures are also trenches of $s_1$, and one tracker rides both.',
  subs=[('trenches of $s_1$, and one tracker rides both.', 'trenches of $s_1$.')]),

T(r"""
\section{Relation Between the Fields and Observer Frames}
\label{sec:ch3:relation}
"""),
P('sep', 'Both fields are read from the same fitted Jacobian, so moving between',
  end='trenches of $D$ lie.',
  subs=[('Both fields are read from the same fitted Jacobian, so moving between\nthem changes nothing in the estimator. Where',
         'Where')]),

T(r"""
\section{Benchmark Fields}
\label{sec:ch3:benchmarks}

\subsection{Canonical Linear Fields}
\label{sec:ch3:canonical}

% New text
Each critical point type of Table~\ref{tab:ch3:field_summary} has a
linear representative, shown in Fig.~\ref{fig:ch3:allfield}. Their
equations are given in Appendix~\ref{app:fields}. A linear field has a
constant Jacobian, so a first-order fit recovers it exactly on any
nondegenerate formation.

\subsection{The Steady Double Gyre}
\label{sec:ch3:dg_example}
"""),
P('sep', 'The double gyre is', end='\\label{fig:dg_fields}', extra=1,
  subs=[('(Section~\\ref{sec:surrogates}), so\nthe two fields share every trench.',
         '(Section~\\ref{sec:ch3:relation}), so\nthe two fields share every trench.'),
        ('OECS definition of Section~\\ref{sec:surrogates}',
         'OECS definition of Section~\\ref{sec:ch3:oecs}'),
        ('figures/double_gyre_fields.png', 'ch3_double_gyre_fields.png'),
        ('width=0.75\\columnwidth', 'width=0.7\\textwidth')]),
]

write('ch03_preliminaries.tex', parts)
