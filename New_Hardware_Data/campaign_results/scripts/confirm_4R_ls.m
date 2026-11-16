function confirm_4R_ls(f)
S = load(f); rc = S.run_config;
fprintf('SWITCHES (as recorded):\n');
for s = rc.switches
    b = regexprep(strrep(s.block, newline, ' '), '^[^/]*/', '');
    if contains(b, 'Jacobians/'), continue; end
    p = s.passes; if ~ischar(p), p = strjoin(cellstr(string(p)), ','); end
    fprintf('   %-28s sw=%s passes: %s\n', b, num2str(s.sw), strrep(p, newline, ' '));
end
mf = rc.matlab_functions; blk = strrep({mf.block}, newline, ' ');
k = find(contains(blk, '/Jacobian/Critical_point_predicter/MATLAB Function'));
for kk = k, fprintf('\nCODE of %s:\n%s\n', blk{kk}, mf(kk).code); end
% recompute 4-point LS CP from logged poses + decoded readings
n = numel(S.tout); cpLS = nan(n, 2);
for t = 1:n
    A = zeros(4, 3); U = zeros(4, 2);
    for r = 1:4, p = S.(sprintf('pose_r%d', r))(t, 1:2); A(r, :) = [p 1]; U(r, :) = S.field_uv(t, 2 * r - 1:2 * r); end
    c = A \ U; J = c(1:2, :)'; cpLS(t, :) = (-J \ c(3, :)')';
end
fprintf('\nrecomputed 4-point least squares vs logged field_cp_jacobian: max diff %.2g m\n', max(abs(cpLS(:) - S.field_cp_jacobian(:))));
fprintf('logged field_cp_jacobian vs field_cp_jacobians (the averaging one): median diff %.3f m\n', median(vecnorm(S.field_cp_jacobian - S.field_cp_jacobians, 2, 2)));
% which CP does the radius/command actually use?
C = S.cluster_state(:, 1:2);
fprintf('orbit_radius_jacobian = |centroid - field_cp_jacobian| ? max diff %.2g\n', max(abs(S.orbit_radius_jacobian(:, 1) - vecnorm(C - S.field_cp_jacobian, 2, 2))));
for nm = {'field_cp_jacobian', 'field_cp_jacobians'}
    cp = S.(nm{1}); u = zeros(n, 2);
    for t = 1:n, [u(t, 1), u(t, 2)] = orb(C(t, :), cp(t, :), 0.01); end
    y = S.cmd_cluster_jacobian(:, 1:2);
    g = u(:) \ y(:); res = y(:) - g * u(:);
    fprintf('cmd_cluster_jacobian(:,1:2) explained by Orbiter on %-19s: gain %.3f, R^2 %.4f\n', nm{1}, g, 1 - var(res) / var(y(:)));
end
fprintf('cmd_robots_selected == cmd_robots_jacobian: max diff %.2g | == cmd_robots_jacobians: max diff %.2g\n', max(abs(S.cmd_robots_selected(:) - S.cmd_robots_jacobian(:))), max(abs(S.cmd_robots_selected(:) - S.cmd_robots_jacobians(:))));
end
function [vx, vy] = orb(c, cp, rd)
d = cp - c; r = norm(d); un = d / max(r, 1e-6);
rs = max(min(1 * (r - rd), 0.3), -0.3);
vx = 0.15 * un(2) + rs * un(1); vy = -0.15 * un(1) + rs * un(2);
end
