import port
from port import P, T, write
port.PRE = 'ch2'

IDC = '% Ported from: IDETC 2025 (DETC2025-167604), {}, transcribed from the published PDF'

parts = [
T(r"""
% ===================================================================
% CHAPTER 2: VERIFICATION TOOLS
% ===================================================================
\chapter{Verification Tools}
\label{ch:tools}

% New text
The hardware results of this dissertation come from the Decabot testbed
of the Santa Clara University Robotic Systems Laboratory. The robots, the
motion capture system, and the cluster space software predate this work
\cite{idc:16}. This dissertation adds the HSV encoding of vector fields
on printed floor maps, the neural network calibration of the robots'
color sensors, a remeasurement of the robots' dynamics, and the
simulation environments built around them.

\section{The Decabots}
\label{sec:ch2:decabots}
"""),
T(IDC.format('Sec. 2.2') + r"""
The robots used for this testbed are Decabots, a fleet of
omnidirectional rovers developed at Santa Clara University
\cite{idc:16}. They are equipped with TCS34725 RGB sensors on their
undercarriage, enabling them to sense the color patterns of the colored
floor mat underneath and estimate the vector field being represented at
that point. Each is powered by an Arduino and communicates over TCP/IP to
a host computer that interprets the robot's sensed value, uses the
information to calculate and return robot velocity commands to each
robot. Each robot has a maximum linear velocity of 0.3~m/s. No
modifications to these robots from their original design were made for
this study.

A perspective view of the Decabot can be seen in
Fig.~\ref{fig:ch2:decabot}. It has omnidirectional wheels, as well as
small reflective spheres on its roof, which enable tracking with the
motion capture cameras.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.5\textwidth]{ch2_decabot.jpg}
    \caption{A Decabot rover with OptiTrack motion tracking balls attached.}
    \label{fig:ch2:decabot}
\end{figure}

Fig.~\ref{fig:ch2:undercarriage} shows the undercarriage of the Decabot
rover. The TCS34725 has a white light LED onboard to illuminate the floor
map being sensed. The undercarriage also has a small shroud to minimize
the effects of external lighting. A simple test of turning the overhead
lab lights on and off confirmed that external lighting did not affect
readings from the test palette.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.5\textwidth]{ch2_undercarriage.png}
    \caption{Undercarriage of a Decabot rover with the LED turned on and the
    external light shroud.}
    \label{fig:ch2:undercarriage}
\end{figure}

\section{Testbed and Motion Capture}
\label{sec:ch2:testbed}
"""),
T(IDC.format('Sec. 3.1') + r"""
This research used an existing testbed as a starting point
\cite{idc:16}. Unlike general-purpose multi-agent platforms such as
Robotic Park \cite{idc:19} that emphasize heterogeneous robot
integration, our testbed specifically focuses on vector field
representation and interpretation through a homogeneous robot fleet with
consistent sensing capabilities. It includes three omnidirectional mobile
rovers (Decabots), 2.3~m$^2$ multicolored floor maps representing
different vector fields, and a motion capture system (OptiTrack) to
verify the position of the robots relative to the floor mat with
centimeter level accuracy. The Decabots and OptiTrack communicate to a
central computer via WiFi. The central computer acts as the main
controller, hosting the control architecture implemented in
MATLAB/Simulink. The central computer processes the information received
then communicates robot velocity commands back to each individual robot.
Fig.~\ref{fig:ch2:testbed_overview} shows the layout of all the
equipment.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.75\textwidth]{ch2_testbed_overview.png}
    \caption{Overview of the testbed developed for testing vector field
    adaptive navigation.}
    \label{fig:ch2:testbed_overview}
\end{figure}
"""),
P('sys', "The testbed used for validation is SCU's Robotic Systems Laboratory Decabot testbed",
  subs=[('The testbed used for validation is', 'The same testbed, used for the critical point experiments of Chapter~\\ref{ch:first_order}, is')]),
P('sys', '\\caption{Experimental testbed: 1.6 m', end='\\end{figure}', back=3,
  subs=[('testbed_with_four.png', 'ch2_testbed_with_four.png'),
        ('width=0.28\\textwidth', 'width=0.5\\textwidth')]),
P('sys', 'The multilayer control architecture from Section~III runs in MATLAB/Simulink',
  subs=[('from Section~III runs', 'of Chapter~\\ref{ch:estimation} runs')]),

T(r"""
\section{Vector Field Encoding in HSV Color Space}
\label{sec:ch2:hsv}

\subsection{Limitations of RGB Encoding}
"""),
T(IDC.format('Sec. 1') + r"""
Previous attempts at creating vector field testbeds for multirobot
adaptive navigation had limited success due to technical barriers
\cite{idc:5}. Approaches using red and green colors to represent
orthogonal vectors suffered from sensor channel interference on robotic
RGB sensors. This interference made encoding vectors to color maps
problematic, preventing accurate decoding of direction and magnitude
information. This issue was particularly severe when dealing with vectors
of small magnitude, as minor value changes could result in large
direction shifts. Additionally, the limited discrete sampling speed of
0.15~Hz made continuous motion of the robots impossible, limiting the
study of motion dynamics.

\subsection{Hue as Direction, Saturation as Magnitude}
"""),
T(IDC.format('Sec. 3.2') + r"""
Previous attempts to use RGB colors to represent orthogonal vector sets
showed limited success, as only a small portion of the spectrum remained
usable after eliminating high-interference regions. To allow for greater
color variance, an HSV color space was implemented to model vector space.
Hue, with its inherently circular nature, provided a natural
representation for vector direction. For magnitude representation,
saturation (ranging from 0 to 1) was chosen after early tests revealed
that the value channel was highly sensitive to printer ink variations,
ambient lighting conditions, and shadow effects.
Fig.~\ref{fig:ch2:hsv_space} shows a saturation-versus-hue plot
representing all possible vector representations.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.55\textwidth]{ch2_hsv_space.png}
    \caption{A plot of saturation versus hue representing vector space.}
    \label{fig:ch2:hsv_space}
\end{figure}
"""),
P('sys', 'Vector fields are physically realized as printed floor maps using HSV color encoding'),
T(r"""
\subsection{Printed Field Construction}

% New text
Each printed map is generated from an analytical field. The fixed and
sinking vortex maps of Chapter~\ref{ch:zeroth} and the vortex and saddle
maps of Chapter~\ref{ch:first_order} were built this way, and their
equations are given in Appendix~\ref{app:fields}.

""" + IDC.format('Sec. 3.2') + r"""
Once vectors were created at all plot points, they were scaled to the
range of 0.3 to 1. Values less than 0.2 were not used in training, and
values less than 0.3 produced velocities insufficient to overcome static
friction.

\section{Neural Network Sensor Calibration}
\label{sec:ch2:calibration}

""" + IDC.format('Sec. 3.3') + r"""
The RGB sensor data suffered from channel interference as noted in
previous studies \cite{idc:5}. The TCS34725 RGB sensor specification
sheet confirms that blue, green, and red channels cannot be fully
isolated. This effect persisted even after conversion to an HSV color
space. Traditional calibration methods neither significantly improved
performance nor supported effective scaling across multiple robots,
prompting exploration of a supervised learning approach.

\subsection{Calibration Palette and Data Collection}

A calibration palette with varying hues ($0$ to $2\pi$) and saturations
was created, with values spaced at 24 equal intervals in each dimension.
Three robots were positioned above each subplot, recording RGB and K
measurements alongside the intended hue and saturation levels (see
Fig.~\ref{fig:ch2:training_data}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch2_training_data.png}
    \caption{A Decabot collecting training data for the vector estimation
    models.}
    \label{fig:ch2:training_data}
\end{figure}

The dataset from three robots was combined and augmented by introducing
$\pm1\%$ noise to RGB readings to ensure model robustness. The collected
RGBK values were normalized between 0 and 1, converted to HSV using
standard MATLAB libraries, and appended to the original data, resulting
in a 7-dimensional input space (R,G,B,K,H,S,V). Data points with
saturation values below 0.2 were removed to boost model performance. All
inputs and targets were normalized to the $[-1,1]$ range before training.

\subsection{Network Architecture and Training}

Two separate neural networks were created using MATLAB's Deep Learning
Toolbox. The architecture for both networks consisted of 7 input neurons
(R,G,B,K,H,S,V), followed by two fully-connected hidden layers with tanh
activation functions. The saturation network had a single output neuron
for direct magnitude prediction, while the hue network had two output
neurons representing sine and cosine components to handle the circular
nature of directional data.

A random search approach determined the optimal hidden layer sizes,
exploring 2 to 12 neurons in the first layer and 4 to 10 neurons in the
second. Model patience and augmentation parameters were also randomized,
while the Levenberg-Marquardt optimizer and default learning rate
remained constant. Data was split into training (70\%), validation
(15\%), and test (15\%) sets.

The models were evaluated against 8 different validation sets including
calibration and verification plots from all four robots. The selection
criterion focused on worst-case performance. For each candidate model,
the lowest $R^2$ score across all validation sets was identified, and the
model with the highest minimum score was selected for each network. This
min-max selection strategy ensured consistent performance across all test
conditions.

\subsection{Cross-Robot Generalization}

""" + IDC.format('Sec. 4.1') + r"""
The supervised learning approach for sensor calibration demonstrated
excellent vector field interpretation capabilities across all validation
datasets. Using a cross-robot training strategy, we achieved high
accuracy in both direction and magnitude sensing, with robust
generalization to new robots.

As shown in Table~\ref{tab:ch2:calibration}, the cross-trained model
maintained circular $R^2$ values consistently above 0.96 for direction
(hue) and above 0.90 for magnitude (saturation) across all robots and
datasets. The model's generalization capability is most evident in its
performance on Robot 4 (R4), which wasn't included in the training
dataset. Despite having no individual calibration data for R4, the
cross-trained model achieved a circular $R^2$ of 0.993 (RMSE 0.024) for
direction and 0.908 (RMSE 0.073) for magnitude on the calibration plot.

\begin{table}[htbp]
\centering
\caption{Comparative performance of cross-trained and individually tuned
models. R4 was not in the training set and has no individually tuned
model.}
\label{tab:ch2:calibration}
\begin{tabular}{llcccc}
\hline
 & & \multicolumn{2}{c}{Hue} & \multicolumn{2}{c}{Saturation} \\
Robot-dataset & Model & $R^2$ & RMSE & $R^2$ & RMSE \\
\hline
R1-Cal & Cross & 0.966 & 0.017 & 0.950 & 0.054 \\
       & Indiv & 0.999 & 0.009 & 0.972 & 0.040 \\
R1-Ver & Cross & 0.969 & 0.050 & 0.950 & 0.064 \\
       & Indiv & 0.995 & 0.020 & 0.963 & 0.055 \\
R2-Cal & Cross & 0.998 & 0.013 & 0.935 & 0.061 \\
       & Indiv & 0.996 & 0.018 & 0.968 & 0.043 \\
R2-Ver & Cross & 0.967 & 0.052 & 0.943 & 0.069 \\
       & Indiv & 0.991 & 0.027 & 0.952 & 0.063 \\
R3-Cal & Cross & 0.998 & 0.014 & 0.948 & 0.055 \\
       & Indiv & 0.999 & 0.007 & 0.980 & 0.034 \\
R3-Ver & Cross & 0.991 & 0.027 & 0.953 & 0.062 \\
       & Indiv & 0.997 & 0.015 & 0.976 & 0.044 \\
R4-Cal & Cross & 0.993 & 0.024 & 0.908 & 0.073 \\
R4-Ver & Cross & 0.964 & 0.054 & 0.925 & 0.079 \\
\hline
\end{tabular}
\end{table}

While individually-tuned models achieved marginally better performance
(typically $R^2 > 0.99$ for hue, $R^2 > 0.95$ for saturation), the
cross-trained approach offered substantial time savings by eliminating
several hours of calibration per robot. This cross-trained approach
significantly improves fleet scalability for vector field navigation. New
robots can simply be loaded with the pre-trained model and verified using
a smaller validation plot, eliminating the need for extensive individual
calibration procedures while maintaining high accuracy.

\subsection{Recalibration After Maintenance}

% New text
The robots were recalibrated after preventive maintenance, before the
critical point experiments of Chapter~\ref{ch:first_order}. The
recalibrated models use the four raw channels as inputs and are the ones
the later hardware results rely on.

"""),
P('sys', 'The experimental platform used three omnidirectional Decabot rovers operating within',
  subs=[('The experimental platform used three omnidirectional Decabot rovers operating within a 1.6~m $\\times$ 1.6~m workspace. Each Decabot features three omnidirectional wheels providing holonomic motion capability and a downward-facing TCS34725 RGB color sensor enclosed in a light-shrouded housing to minimize external lighting effects. ', '')]),
P('sys', '\\caption{Neural network predictions for HSV field encoding on unit-normalized axes.',
  end='\\end{figure}', back=3,
  subs=[('color_sensor_hsv_predictions.png', 'ch2_color_sensor_hsv_predictions.png'),
        ('width=0.90\\columnwidth', 'width=0.8\\textwidth')]),

T(r"""
\section{Saturation-to-Velocity Characterization}
\label{sec:ch2:saturation}

""" + IDC.format('Sec. 3.4') + r"""
To validate the relationship between detected vector magnitude and robot
velocity, lanes of a single hue at different saturation levels were
printed (Fig.~\ref{fig:ch2:lanes}). Each robot was placed on each of the
different saturation lanes and commanded to follow the direction based on
hue, and to use the magnitude indicated by saturation to scale its
velocity.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{ch2_saturation_lanes.png}
    \caption{Lanes of constant hue and varying saturation for testing
    Decabot velocities.}
    \label{fig:ch2:lanes}
\end{figure}

Each robot's position was tracked over a 3 second period as it followed
each lane. This continuous motion capability stands in contrast to sparse
sampling approaches necessary in marine environments \cite{idc:14}, where
energy constraints and environmental factors limit sampling frequency.
The average velocity for each saturation run was computed. Across all
robots, the velocities scaled linearly with saturation at values above
0.3. As noted earlier, saturation values below 0.3 did not produce
velocity commands strong enough to overcome static friction of the
robots. The robots achieved a max velocity of 0.3~m/s. This consistency
between saturation and robot velocity across robots is key for
interpreting vector primitive performance.

\section{Robot Dynamics Identification}
\label{sec:ch2:dynamics}
"""),
P('sys', 'In simulation, each robot is modeled as a single omnidirectional point mass',
  end='the momentum coefficient $\\alpha = e^{-\\Delta t/\\tau}$. At a 10~Hz control rate'),
P('sys', 'The sensed field $\\mathbf{v}(\\mathbf{p}_i)$ enters only the estimation framework of Section~II.',
  subs=[('enters only the estimation framework of Section~II.',
         'enters only the estimation framework of Section~\\ref{sec:ch4:estimation}.'),
        ('Section~VI-D discusses the extension to flows that advect the robot directly.',
         'Section~\\ref{sec:ch9:advection} discusses the extension to flows that advect the robot directly.')]),

T(r"""
\section{Field Reconstruction as a Simulation Noise Model}
\label{sec:ch2:reconstruction}
"""),
P('sys', 'This method of representing vector fields in printed floor maps has imperfections',
  subs=[('referenced in Section~IV as the realistic noise model for simulation.',
         'used in Chapter~\\ref{ch:first_order} as the realistic noise model for simulation.')]),
P('sys', '\\caption{Analytical fields, sensor-based reconstructions, and reconstruction error',
  end='\\end{figure}', back=3,
  subs=[('measurement_error_comparison.png', 'ch2_measurement_error_comparison.png'),
        ('width=0.43\\textwidth', 'width=0.75\\textwidth')]),

T(r"""
\section{Simulation Environment}
\label{sec:ch2:simulation}

% New text
Both simulation studies use a Python implementation of the three-layer
architecture of Section~\ref{sec:ch4:architecture}, with the robot model
of Section~\ref{sec:ch2:dynamics}. The three-robot simulator of
Chapter~\ref{ch:first_order} reads either the analytical fields of
Appendix~\ref{app:fields} or the reconstructed fields above, which carry
the testbed's own sensing error. The six-robot simulator of
Chapter~\ref{ch:second_order} reads analytical or gridded fields and adds
optional measurement noise on each reading and position noise at
measurement time, leaving the true robot state untouched.

\section{Testbed Limitations}
\label{sec:ch2:limitations}

""" + IDC.format('Sec. 5') + r"""
Despite significant improvements over previous testbeds, important
limitations persist. Indoor small-scale testbeds often inadequately
represent large-scale environments \cite{idc:10}, and our system is
currently restricted to static vector fields, preventing reactive control
testing in dynamic environments \cite{idc:6} with no clear path to enable
this using colormaps. Constrained to two-dimensional vector fields, 3D
extensions would require additional sensing capabilities.

The robots sample only a single point in the field at their current
location without look-ahead capability, hindering testing of predictive
path planning methods. The testbed is limited to homogeneous robots and
lacks obstacle avoidance sensors, restricting studies of heterogeneous
multi-robot behaviors in complex environments. Physical constraints
include a maximum 3~m by 6~m work area, approximately one hour battery
life per robot, and a limited velocity range of 0.05 to 0.3~m/s.
Additionally, the Decabots' wheels degrade both paper and printed colors
on colormaps, necessitating replacement after a few dozen runs and adding
significant operational costs.
"""),
]

write('ch02_verification_tools.tex', parts)
