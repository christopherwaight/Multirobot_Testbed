function y = field_jacobian_log(x1, y1, u1, v1, x2, y2, u2, v2, x3, y3, u3, v3)
% LOGGING ONLY: the code inside the 'LOG field_jacobian fn' MATLAB Function block in
% Jacobian_primitive/Critical_point_predicter. It takes the same inputs as Criticalpointfinder
% (robot positions in Motive x,z and the decoded field u,v) and does the same 3-point affine fit:
%   [u; v] = J * [x; z] + c
% Output (1x8): [J11 J12 J21 J22 trace det curl type]
%   J = [du/dx du/dz; dv/dx dv/dz]; curl = dv/dx - du/dz (> 0: rotation from +x toward +z)
%   type: 1 vortex/centre, 2 spiral, 3 node (source/sink), -1 saddle, 0 degenerate
A = [x1 y1 1; x2 y2 1; x3 y3 1];
cu = A \ [u1; u2; u3];
cv = A \ [v1; v2; v3];
J11 = cu(1); J12 = cu(2); J21 = cv(1); J22 = cv(2);
tr = J11 + J22;
dt = J11 * J22 - J12 * J21;
curl = J21 - J12;
if ~isfinite(dt) || ~isfinite(tr) || abs(dt) < 1e-12
    typ = 0;
elseif dt < 0
    typ = -1;
elseif tr^2 < 4 * dt
    if abs(tr) < 0.4 * sqrt(dt)
        typ = 1;
    else
        typ = 2;
    end
else
    typ = 3;
end
y = [J11, J12, J21, J22, tr, dt, curl, typ];
end
