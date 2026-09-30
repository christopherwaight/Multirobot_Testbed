import port
from port import P, T, write
port.PRE = 'ch5'

IDC = '% Ported from: IDETC 2025 (DETC2025-167604), {}, transcribed from the published PDF'

parts = [
T(r"""
% ===================================================================
% CHAPTER 5: ZEROTH-ORDER BASELINES (IDETC 2025 testbed paper)
% ===================================================================
\chapter{Zeroth-Order Baselines}
\label{ch:zeroth}

\section{Direction-Only Navigation}
\label{sec:ch5:intro}

% New text
The primitives in this chapter steer on the robots' readings without
assembling the velocity gradient. The vector-sum primitive averages the
sensed vectors, and the vector-to-scalar primitive follows the slope of
the sensed magnitude. Both supply a heading. They were run on the testbed
of Chapter~\ref{ch:tools}, and they set the baseline for the higher rungs
of Section~\ref{sec:ch4:ladder}.

""" + IDC.format('Sec. 2.5') + r"""
To evaluate our platform, we implemented two adaptive navigation
primitives as control laws from \cite{idc:4}. These were both shown as
center finding primitives that can find environmental extrema or follow a
closed orbit around the extrema. Both primitives were previously
validated in simulation, with known limitations documented. These
primitives rely solely on instantaneous information, enabling a purely
reactive approach to adaptive navigation control.

\section{Vector-Sum Primitive}
\label{sec:ch5:vector_sum}

""" + IDC.format('Sec. 2.5') + r"""
The vector-sum technique combines the vectors sensed by each robot into a
resultant vector. The normalized direction of this resultant is then used
to specify $\dot{x}_c$ and $\dot{y}_c$ for the cluster.

\section{Vector-to-Scalar Primitive}
\label{sec:ch5:vector_to_scalar}

""" + IDC.format('Sec. 2.5') + r"""
The vector-to-scalar technique uses only the magnitude information from
vector readings, effectively transforming vector field navigation into a
scalar field problem. This allows the application of scalar field
gradient ascent primitives. In the max-seeking implementation,
differences in vector magnitude readings from individual robots determine
the direction of steepest change.

% New text. The hardware implementation used this affine fit. The
% published IDETC paper printed an older cross-product construction
% (its Eqs. 2-7), which the author describes as confusing; it is not
% reproduced here.
The implementation fits a plane to the magnitudes $z_i =
\lVert\mathbf{v}(\mathbf{p}_i)\rVert$ read by the three robots,
\begin{equation}
    \mathbf{A}\,\hat{\boldsymbol{\theta}}_z = \mathbf{z}, \qquad
    \hat{\boldsymbol{\theta}}_z = [\hat{a}, \hat{b}, \hat{c}]^{\top},
    \label{eq:ch5:mag_fit}
\end{equation}
with the same formation matrix $\mathbf{A}$ as the first-order fit of
Section~\ref{sec:ch6:estimation}. The fitted slope
$(\hat{a}, \hat{b})$ estimates the gradient of the magnitude, and the
max-seeking behaviour moves the cluster along it. A minimum-seeking
behaviour moves against it. This is a first-order fit of a scalar. It
never assembles the Jacobian of the vector field, so it recovers a
direction of steepest change but no critical point.

\section{Hardware Trials on Fixed and Sinking Vortices}
\label{sec:ch5:trials}

\subsection{Setup}

""" + IDC.format('Sec. 2.4') + r"""
For this research, we used a 3-robot cluster with a constant shape policy
as shown in Fig.~\ref{fig:ch4:threerobot}. The cluster parameters were set
to $p = 0.35$~m, $q = 0.35$~m, and $\beta = 1.05$~radians. These
parameters were chosen to form an equilateral triangle which maintains a
tight grouping and equidistant spacing. Cluster control was chosen to
maintain explicit formation control with full controllability and
observability which would not be easily achievable with alternative
approaches such as swarm architecture or leader-follower systems
\cite{idc:18}. Neither the cluster orientation nor individual robot
orientations were specified, ensuring all velocity commands related
directly to the control primitives and formation control rather than
orientation maintenance.

The feature estimator converts data collected by the Decabots into usable
information for the control law, converting RGB sensor data into
navigation parameters. Section~\ref{sec:ch2:calibration} details the
feature estimator's operation.

%% The formation here is 0.35 m on a side; the critical point experiments
%% of Chapter 6 used 0.33 m. These are two experiments, not a conflict.

""" + IDC.format('Sec. 3.2') + r"""
Two vector fields were constructed for these trials. The first studied
field was a fixed vortex, modeled as
\begin{equation}
    r = \sqrt{(x - c_x)^2 + (y - c_y)^2}, \qquad
    \theta = \operatorname{atan2}(y - c_y,\, x - c_x),
    \label{eq:ch5:rtheta}
\end{equation}
\begin{equation}
    u = r\sin\theta, \qquad v = -r\cos\theta ,
    \label{eq:ch5:fixed_vortex}
\end{equation}
where $(c_x, c_y)$ is the center of the map. Fig.~\ref{fig:ch5:fixed_map}
shows the vector field of (\ref{eq:ch5:rtheta})--(\ref{eq:ch5:fixed_vortex})
as a quiver plot overlaid on an HSV plot.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.65\textwidth]{ch5_fixed_vortex_map.png}
    \caption{Quiver plot and HSV plot of the fixed vortex centered at the
    origin.}
    \label{fig:ch5:fixed_map}
\end{figure}

The other vector field studied was that of a sinking vortex. Equations
(\ref{eq:ch5:rtheta})--(\ref{eq:ch5:fixed_vortex}) are used to create the
fixed vortex, and then a sinking component is added to the field,
\begin{equation}
    u \leftarrow u - \frac{c_s\,(x - c_x)}{r^2 + 10^{-6}}, \qquad
    v \leftarrow v - \frac{c_s\,(y - c_y)}{r^2 + 10^{-6}},
    \label{eq:ch5:sinking}
\end{equation}
with $c_s = 0.2$.
%% The published Eqs. 12-13 print this term with a plus sign and 0.1. The
%% plotting package (trunk/robots_3/Results_for TestBed_Paper/
%% plotting_package/plotter_avg_run_Sinking_Vortex.m) uses the inward sign
%% and 0.2, and the trials spiral inward. Author decision 2026-09-29: use
%% the code values.
A small value of $10^{-6}$ was added to the denominator to avoid errors
when attempting to divide by zero. Once again, all vectors in the plot are
rescaled to the range of 0.3 to 1. The sinking vortex vector field
resulting from these equations is shown in Fig.~\ref{fig:ch5:sinking_map}
as a quiver plot overlaid on an HSV plot.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.65\textwidth]{ch5_sinking_vortex_map.png}
    \caption{Quiver plot and HSV plot of the sinking vortex centered at the
    origin.}
    \label{fig:ch5:sinking_map}
\end{figure}

\subsection{Test Methodology}

""" + IDC.format('Sec. 3.5') + r"""
A 3-robot cluster with fixed formation parameters ($p = 0.35$~m,
$q = 0.35$~m, and $\beta = 1.05$~radians) was used to test both control
primitives (vector-sum and vector-to-scalar) across the two vector field
environments. Throughout all tests, the OptiTrack motion capture system
recorded robot positions at a frequency of 10~Hz.

For each environment-primitive combination, 10 trial runs were conducted
to gather statistically meaningful data. The fixed vortex environment was
tested with both the vector-sum and vector-to-scalar minimum seeking
primitives, while the sinking vortex was tested exclusively with the
vector-sum primitive.

Both consistent and randomized starting positions were used to evaluate
the robustness of the control primitives. Performance evaluation focused
on three metrics, the robot trajectory relative to the expected path, the
steady-state error between the final cluster position and the target
position (where applicable), and the cluster's ability to maintain its
shape.

\subsection{Results}

""" + IDC.format('Sec. 4.2') + r"""
The high accuracy achieved in hue and saturation interpretation ($R^2$
values of 0.96 and 0.91 respectively) enabled the testing of control
primitives in different environments that were previously only validated
in simulation.

The first set of test runs using a 3-robot cluster to navigate around the
center of a fixed vortex using the vector-sum primitive demonstrated an
unexpected but consistent physical phenomenon.
Fig.~\ref{fig:ch5:fixed_vs} shows the average path travelled over 10 runs
with the shaded region indicating 1 standard deviation, revealing very
high consistency between runs.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch5_fixed_vortex_vector_sum.png}
    \caption{Average path of the robots in the fixed vortex using the
    vector-sum primitive, with one standard deviation shaded.}
    \label{fig:ch5:fixed_vs}
\end{figure}

The outward drift rate appears stable at all angles, a behavior not
predicted in simulation studies. This drift occurs because the robots,
following only their perceived direction using the vector-sum technique,
are always moving tangent to the circle while experiencing momentum
effects not captured in previous simulation models. This real-world
validation highlights the importance of physical testing, as the physical
dynamics introduced nuances absent in simulated environments. The ability
to orbit a center with a fixed radius using adaptive navigation
techniques remained an open problem at this rung.

The second set of test runs were performed on the sinking vortex
environment using the vector-sum technique. Fig.~\ref{fig:ch5:sinking_vs}
shows the robots following a smooth path as they spiral inwards towards
the center as predicted in simulation. The solid lines show the average
run, and the blurred area shows 1 standard deviation. The smoothness can
be attributed to the very high $R^2$ value in interpreting hue as well as
to the stability of the center of this vortex field.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch5_sinking_vortex_vector_sum.png}
    \caption{Average path of the robots in the sinking vortex with the
    vector-sum primitive.}
    \label{fig:ch5:sinking_vs}
\end{figure}

Fig.~\ref{fig:ch5:sinking_dist} shows the distance of the cluster center
relative to the origin with time. In these tests, the cluster settles
around 0.1~m away from the center. Looking at
Fig.~\ref{fig:ch5:sinking_vs}, the desired location falls within the
boundaries of the triangle that connects the 3 robots in the cluster.
This suggests that smaller cluster space variable values would lead to a
more accurate prediction.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch5_sinking_vortex_distance.png}
    \caption{Average error of the cluster position relative to the center of
    the sinking vortex.}
    \label{fig:ch5:sinking_dist}
\end{figure}

For the third set of test runs, the vector-to-scalar minimum seeking
primitive was used to find the center of the fixed vortex floor map. The
robots are expected to follow a straight line to the center as seen in
Fig.~\ref{fig:ch5:fixed_v2s}. The trajectory of the robots is very
consistent across runs. The trajectory isn't a perfectly straight line,
as there may be noise in the printing, small fluctuations in robot
behavior, and some variance in the way that saturations affect estimates
of magnitudes.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch5_fixed_vortex_vector_to_scalar.png}
    \caption{Average path of the robots in the fixed vortex with the
    vector-to-scalar primitive.}
    \label{fig:ch5:fixed_v2s}
\end{figure}

The cluster settled within 0.2~m of center (Fig.~\ref{fig:ch5:v2s_dist}),
suggesting the center fell within the circumscribed circle of the robot
triangle. The graph shows steady approach and stable positioning once the
desired location was reached.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch5_vector_to_scalar_distance.png}
    \caption{Average error of the cluster position relative to the center of
    the fixed vortex with the vector-to-scalar primitive.}
    \label{fig:ch5:v2s_dist}
\end{figure}
%% The published caption of this figure (IDETC Fig. 18) says "sinking
%% vortex"; the text beside it and the data describe the fixed vortex.

For a fourth set of tests, the vector-to-scalar minimum seeking trials in
the fixed vortex were repeated with randomized starting positions on the
colormap. They showed a steady state error close to that of the trials
that all had the same starting location. This could be happening because
the velocity command is calculated from the gradient of saturation, and
not from the saturation itself. A small gradient could result in velocity
commands less than 0.05~m/s that are not sufficient for the robots to
overcome static friction.

\section{Formation Maintenance}
\label{sec:ch5:formation}

""" + IDC.format('Sec. 4.3') + r"""
Throughout all experiments, the cluster space controller maintained
precise formation control, as evidenced by
Table~\ref{tab:ch5:formation}. The values stayed within 1~cm of desired
parameters ($p = 0.35$~m, $q = 0.35$~m, and $\beta = 1.05$~radians) across
all test environments. This formation precision, within the resolution
limits of the OptiTrack motion capture system, demonstrates the
effectiveness of the cluster space controller in maintaining desired
geometric relationships between robots while executing adaptive
navigation primitives.

\begin{table}[htbp]
\centering
\caption{Formation control performance over the primitive testing trial
runs.}
\label{tab:ch5:formation}
\begin{tabular}{lcccccc}
\hline
 & \multicolumn{4}{c}{Fixed vortex} & \multicolumn{2}{c}{Sinking vortex} \\
 & \multicolumn{2}{c}{Vector-sum} & \multicolumn{2}{c}{Vector-to-scalar} & \multicolumn{2}{c}{Vector-sum} \\
Variable & Average & $\sigma$ & Average & $\sigma$ & Average & $\sigma$ \\
\hline
$p$ & 0.36~m & 0.028 & 0.35~m & 0.015 & 0.35~m & 0.019 \\
$q$ & 0.36~m & 0.032 & 0.36~m & 0.009 & 0.35~m & 0.015 \\
$\beta$ & 1.03~rad & 0.122 & 1.02~rad & 0.045 & 1.02~rad & 0.056 \\
\hline
\end{tabular}
\end{table}

\section{Direction Without Location}
\label{sec:ch5:summary}

% New text
Both primitives reached their target when the field carried the cluster
there. On the sinking vortex the flow itself points inward, and the
vector-sum primitive settled about 0.1~m from the center. On the fixed
vortex the vector-to-scalar primitive descended the magnitude to within
0.2~m. Orbiting was different. A heading keeps the cluster tangent to the
flow, and nothing in it measures the distance to the center, so momentum
carried the cluster steadily outward. Correcting that drift needs the
location of the center itself, which the three robots can estimate from
the same readings once they assemble the Jacobian.
"""),
]

write('ch05_zeroth_order.tex', parts)
