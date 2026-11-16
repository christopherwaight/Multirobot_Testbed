function [fwdkin, x_c, y_c, th_c, d1, d2, r1, r2, phi] = Forward_Kinematics(x1, y1, th1, x2, y2, th2, x3, y3, th3, x4, y4, th4) %#ok<INUSD>
% 4-robot forward kinematics, matching forward_kinematics_quad in
% from_mac/VF_Robot/VF_Robot/cluster_builder/src/control/quad_kinematics.py (prepared 2026-11-12).
% Robots are paired across the diagonals: diagonal 1 = robot 1 -> 3, diagonal 2 = robot 2 -> 4.
%   d1, d2 : diagonal lengths
%   r1, r2 : where the diagonals cross, as a fraction along diagonal 1 (from robot 1) and 2 (from robot 2)
%   phi    : angle(diagonal 1) - angle(diagonal 2), WRAPPED to [-pi, pi] (Python's psi)
%   th_c   : angle of diagonal 1 (robot 1 -> 3)
%   x_c, y_c : centroid of the four robots
% fwdkin = [d1 d2 r1 r2 phi x_c y_c th_c] (layout expected by the inverse-Jacobian blocks)
% Changes vs the 2025 block code: phi wrapped (it could reach +-2*pi before), parallel-diagonal guard
% (r1 = r2 = 0.5 like Python), and the debug/persistent code removed.

x_c = (x1 + x2 + x3 + x4) / 4;
y_c = (y1 + y2 + y3 + y4) / 4;

d1 = sqrt((x3 - x1)^2 + (y3 - y1)^2);
d2 = sqrt((x4 - x2)^2 + (y4 - y2)^2);

th_c = atan2(y3 - y1, x3 - x1);

phi = atan2(y3 - y1, x3 - x1) - atan2(y4 - y2, x4 - x2);
phi = mod(phi + pi, 2*pi) - pi;

delta = (x3 - x1) * (y2 - y4) + (y3 - y1) * (x4 - x2);
if abs(delta) < 1e-10
    r1 = 0.5;
    r2 = 0.5;
else
    r1 = ((x2 - x1) * (y2 - y4) + (y2 - y1) * (x4 - x2)) / delta;
    r2 = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / delta;
end

fwdkin = [d1, d2, r1, r2, phi, x_c, y_c, th_c];
end
