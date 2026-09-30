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
