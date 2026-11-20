import port
from port import P, T, write
port.PRE = 'ch1'

IDC = '% Ported from: IDETC 2025 (DETC2025-167604), {}, transcribed from the published PDF'

parts = [
T(r"""
% ===================================================================
% CHAPTER 1: INTRODUCTION
% ===================================================================
\chapter{Introduction}
\label{ch:intro}

\section{Adaptive Navigation in Vector Field Environments}
\label{sec:ch1:an}
"""),
P('sep', '\\IEEEPARstart{M}{ultirobot} systems (MRS) offer redundancy, increased',
  end='field, while an MRS samples many points at once.',
  subs=[('\\IEEEPARstart{M}{ultirobot} systems (MRS)', 'Multirobot systems (MRS)')]),
# The trailing \cite{10} critical-point sentence is covered elsewhere in the
# introduction, so the passage ends before it (the end anchor stops at the
# subsurface-leaks clause rather than stripping the sentence by substitution).
P('sep', 'Some flows are better described as vector fields, which carry features',
  end='and subsurface leaks \\cite{9}.',
  subs=[('\\cite{9}. In', '\\cite{9}.')]),
P('sys', 'The oceanic and atmospheric environments that motivate this work are governed'),

T(r"""
\section{Literature Review}
\label{sec:ch1:lit}

\subsection{Adaptive Navigation in Scalar Fields}
"""),
P('sep', 'AN in scalar fields is well developed.', end='\\cite{6,7}.'),
P('sys', 'AN in scalar fields is well explored.',
  subs=[('AN in scalar fields is well explored. Each robot collects a single scalar measurement as the input to a navigation policy that reaches and follows features such as extrema, saddle points, ridges, trenches, contour lines, and fronts \\cite{5,6,7,8,9}. ', '')]),

T(r"""
\subsection{Navigation in Vector Fields}

""" + IDC.format('Sec. 1') + r"""
While multirobot adaptive navigation in scalar field environments is
well-studied both in simulation and practice, adaptive navigation through
vector fields remains largely theoretical, compared to extensive
experimental confirmation in scalar fields \cite{idc:2}. Existing research
in vector fields primarily focuses on constructing artificial vector
fields for single-robot navigation, formation control, and obstacle
avoidance \cite{idc:3}. In contrast, this research involves navigation
through naturally existing vector fields. The few studies examining
multirobot systems with distributed sensor measurements in physical
vector fields have identified promising control primitives for features
like flow maxima, sinks, sources, and vortices, but these approaches lack
comprehensive real-world validation: they were validated mainly in
simulation \cite{idc:4}, and the PIM-triple approach of \cite{sep:26} was
extended to robots in a flow tank \cite{sep:28}.
"""),
P('sys', 'By contrast, only a few works address navigation of physically sensed vector fields'),

T(r"""
\subsection{Robotic Sensing of Physical Flows}
"""),
P('sys', 'Work that senses the physical field falls into two camps.',
  subs=[('this paper gives', 'this dissertation gives')]),

T(r"""
\subsection{Experimental Testbeds}

""" + IDC.format('Sec. 1') + r"""
The gap between simulation and reality highlights the need to develop
experimental testbeds that allow validation of vector field navigation
strategies under real-world conditions. A testbed of this kind offers
researchers a cost-effective way to test vector field navigation designs
before deployment in challenging environments like ocean currents,
airflow systems, electromagnetic fields, and industrial magnetic settings
\cite{idc:6,idc:7}. Building upon the framework of Mokhtarian et al.
\cite{idc:8}, such a platform validates theoretical models prior to
real-world implementation \cite{idc:9,idc:10}.

\section{Problem Statement}
\label{sec:ch1:problem}

% New text
This dissertation asks what a small cluster of robots can learn about a
planar vector field from one synchronized set of measurements, and how it
can steer on what it learns. The cluster has no map, no forecast, and no
prior knowledge of the features in the field. Each robot reads the local
velocity and knows its own position. The features of interest are the
ones that organize transport, the critical points where the flow vanishes
and the separatrices that divide it into regions.

\section{Thesis Statement}
\label{sec:ch1:thesis}

% New text
The order of the instantaneous local fit that a cluster makes to its
measurements determines the field information available to it, and so the
feature-based navigation it can perform. The statement concerns what one
set of simultaneous measurements supports; methods that add temporal
measurements or prior knowledge lie outside it. Fit order here means the
order to which the vector field itself is reconstructed. A heading, or the
slope of the sensed magnitude, read directly from the samples supports
direction-following behaviors, which reach a target that the flow carries
the cluster to and drift when asked to orbit. A first-order fit from three robots recovers the velocity gradient, and
with it the location and type of a critical point. A second-order fit
from six robots recovers the velocity gradient as a field over the
formation, which is enough to acquire and follow a separatrix. Each rung
needs more robots than the one below it, and its highest-order
coefficients are more sensitive to measurement noise.

\section{Contributions}
\label{sec:ch1:contributions}

% New text, assembled from the contribution lists of the three papers
This dissertation makes seven contributions. First, it presents an indoor
testbed for vector field adaptive navigation, with direction encoded in
hue and magnitude in saturation on printed floor maps, and a cross-robot
neural network calibration that reads both from the robots' color sensors
(Chapter~\ref{ch:tools}). Second, it tests the vector-sum and
vector-to-scalar primitives on that testbed and shows where
direction-only navigation reaches its target and where it drifts
(Chapter~\ref{ch:zeroth}). Third, it presents a cooperative estimation algorithm that
estimates the location and type of a critical point from instantaneous
measurements alone, and proves that three robots are the minimum for an
unrestricted affine fit from one set of measurements
(Chapter~\ref{ch:first_order}). Fourth, it validates attraction
and orbital control laws built on that estimate in 459 hardware runs
with three and four robots (Chapter~\ref{ch:first_order}). Fifth, it presents a
second-order estimator that recovers the local quadratic model of a flow
from six robots, the minimum, and fails only when the robots lie on a
common conic (Chapter~\ref{ch:second_order}). Sixth, it builds two
primitives on that fit, the $D$ tracker and the $s_1$ tracker, which
acquire and ride the trench of their field from instantaneous measurements
without pre-straddling, and it tests them in simulation on a double gyre,
on three further fields, and on measured ocean currents
(Chapter~\ref{ch:second_order}). The $D$ tracker tolerates more
measurement noise, while the $s_1$ tracker, being built from objective
quantities, holds its path under a rotating observer. Finally, it compares
the three rungs on one flow, on formation geometry, on noise, and on steady
offsets, and it states for each rung when the feature that it steers to is
the feature of the flow (Chapter~\ref{ch:crosscutting}).

\section{Publications}
\label{sec:ch1:publications}

% New text
Portions of this dissertation appear in the following papers.
\begin{itemize}
\item C. Waight and C. A. Kitts, ``A functional indoor testbed for
multirobot adaptive navigation in vector field environments,'' in
\textit{Proc. ASME Int. Design Engineering Technical Conf. and Computers
and Information in Engineering Conf.}, 2025, DETC2025-167604. Published.
Chapters~\ref{ch:tools} and~\ref{ch:zeroth}.
\item C. Waight and C. A. Kitts, ``Adaptive navigation of multirobot
systems to critical points in 2D vector fields in simulation and
experiment,'' submitted to \textit{IEEE Systems Journal}.
Chapter~\ref{ch:first_order}.
%% TODO(status): update when the critical-points paper is accepted.
\item C. Waight and C. A. Kitts, ``Second-order cooperative field
estimation for multirobot tracking of coherent flow structures,'' in
preparation for \textit{IEEE Systems Journal}.
Chapter~\ref{ch:second_order}.
\end{itemize}

\section{Availability of Code and Data}
\label{sec:ch1:availability}

% New text. Plan0 optional reproducibility statement. Each campaign folder
% has a README that lists its files, robots, and maps.
The code for every simulation, the scripts behind each figure, the
calibration data, and the logs of the three hardware campaigns are
available from the author on request. Each hardware campaign has a README
that lists its files, its robots, and its maps.

\section{Dissertation Organization}
\label{sec:ch1:organization}

% New text
Chapter~\ref{ch:tools} describes the Decabot testbed, the HSV encoding of
vector fields, the sensor calibration, and the robot dynamics.
Chapter~\ref{ch:preliminaries} defines the vector field quantities the
later chapters estimate, from critical points to objective Eulerian
coherent structures, and the benchmark fields.
Chapter~\ref{ch:estimation} describes the control architecture shared by
every primitive and the local polynomial fit, and it lays out the
fit-order ladder that organizes the rest of the dissertation.
Chapters~\ref{ch:zeroth}, \ref{ch:first_order}, and~\ref{ch:second_order}
climb that ladder one rung at a time, from direction-only primitives on
hardware, to critical point estimation and control on hardware, to
separatrix tracking in simulation. Chapter~\ref{ch:crosscutting} compares
the rungs directly, on the scalar fields each makes available, on
formation geometry, on noise, and on the number of robots.
Chapter~\ref{ch:limitations} states the limitations and future work, and
Chapter~\ref{ch:conclusion} concludes. The appendices give the field
definitions, the cluster kinematics for three and six robots, the frame
equivariance of the $s_1$ tracker, the stability arguments, and the
statistical methods.
"""),
]

write('ch01_introduction.tex', parts)
