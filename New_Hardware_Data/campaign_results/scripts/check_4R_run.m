function check_4R_run(f)
S = load(f); rc = S.run_config; n = numel(S.tout); t = S.tout;
fprintf('run %s, %d steps (%.1f s), stop_time %g\n', rc.created, n, t(end), rc.stop_time);
fprintf('robots (run_config): %s\n', strjoin(strcat(rc.robots(:, 1), '=', rc.robots(:, 2))', ', '));
ot = rc.optitrack; fn = fieldnames(ot);
for k = 1:numel(fn), v = ot.(fn{k}); if ischar(v) || isstring(v), fprintf('optitrack.%s = %s\n', fn{k}, char(v)); elseif iscell(v), fprintf('optitrack.%s = {%s}\n', fn{k}, strjoin(cellfun(@char, v, 'uni', 0), ', ')); end, end
fprintf('decabots: %s\n', strjoin(cellfun(@(x) char(strrep(x, newline, ' ')), rc.decabots, 'uni', 0)', ' | '));
fprintf('switches: %s\n', strjoin(arrayfun(@(s) sprintf('%s=%s', regexprep(strrep(s.block, newline, ' '), '^[^/]*/', ''), num2str(s.sw)), rc.switches, 'uni', 0), ', '));
P = {'setpoint', 'jacobian', 'jacobians'}; e = zeros(1, 3);
for k = 1:3, e(k) = max(abs(S.(['cmd_robots_' P{k}])(:) - S.cmd_robots_selected(:))); end
[m, j] = min(e); fprintf('active primitive: %s (max diff %.2g; others %s)\n', P{j}, m, mat2str(e, 2));

% positions
X = zeros(n, 4, 2); for r = 1:4, X(:, r, :) = S.(sprintf('pose_r%d', r))(:, 1:2); end
c0 = squeeze(mean(X(1, :, :), 2))';
fprintf('\nstart positions (x, z) and angle around centroid (Motive x-z):\n');
a = zeros(1, 4);
for r = 1:4
    p = squeeze(X(1, r, :))'; a(r) = atan2(p(2) - c0(2), p(1) - c0(1));
    fprintf('   r%d %-13s (%.3f, %.3f)  angle %+6.1f deg  heading %+6.1f deg\n', r, rc.robots{r, 2}, p, rad2deg(a(r)), rad2deg(S.(sprintf('pose_r%d', r))(1, 3)));
end
[~, o] = sort(a); fprintf('   order going CCW (increasing angle in x-z): %s\n', mat2str(o));
fprintf('   end positions: '); for r = 1:4, fprintf('r%d (%.3f, %.3f)  ', r, squeeze(X(end, r, :))'); end; fprintf('\n');
% diagonal check: do segments 1-3 and 2-4 intersect?
p1 = squeeze(X(1, 1, :))'; p2 = squeeze(X(1, 2, :))'; p3 = squeeze(X(1, 3, :))'; p4 = squeeze(X(1, 4, :))';
fprintf('   segments 1-3 and 2-4 cross (true diagonals)? %d | 1-2 & 3-4 cross? %d | 1-4 & 2-3 cross? %d\n', segx(p1, p3, p2, p4), segx(p1, p2, p3, p4), segx(p1, p4, p2, p3));
L = {'xc', 'zc', 'thc', 'd1', 'd2', 'r1', 'r2', 'phi'};
fprintf('\n%-4s %9s %9s %9s %9s   (cluster_state start/end, setpoint, err_setpoint end)\n', '', 'start', 'end', 'setpoint', 'err');
for k = 1:8, fprintf('%-4s %9.3f %9.3f %9.3f %9.3f\n', L{k}, S.cluster_state(1, k), S.cluster_state(end, k), S.setpoint(1, k), S.err_setpoint(end, k)); end
% command vs motion matrix
fprintf('\nmotion of robot (row) vs command of robot (col): mean cosine of direction (1 = moves exactly as commanded)\n');
V = zeros(n - 1, 4, 2); Cg = zeros(n - 1, 4, 2);
for r = 1:4
    V(:, r, :) = diff(squeeze(X(:, r, :))) / 0.1;
    Cg(:, r, :) = S.cmd_robots_selected(1:end - 1, 3 * r - 2:3 * r - 1);
end
M = zeros(4);
for r = 1:4
    for c = 1:4
        v = squeeze(V(:, r, :)); u = squeeze(Cg(:, c, :));
        ok = vecnorm(v, 2, 2) > 0.02 & vecnorm(u, 2, 2) > 0.02;
        M(r, c) = mean(sum(v(ok, :) .* u(ok, :), 2) ./ (vecnorm(v(ok, :), 2, 2) .* vecnorm(u(ok, :), 2, 2)));
    end
end
disp(round(M, 2));
fprintf('per robot: commanded speed median, actual speed median, angle actual-vs-own-command median\n');
for r = 1:4
    v = squeeze(V(:, r, :)); u = squeeze(Cg(:, r, :)); ok = vecnorm(v, 2, 2) > 0.02 & vecnorm(u, 2, 2) > 0.02;
    d = angle(exp(1i * (atan2(v(ok, 2), v(ok, 1)) - atan2(u(ok, 2), u(ok, 1)))));
    fprintf('   r%d %-13s cmd %.3f  actual %.3f  angle %+6.0f deg (n %d)  | heading start %+5.0f end %+5.0f deg\n', r, rc.robots{r, 2}, median(vecnorm(u, 2, 2)), median(vecnorm(v, 2, 2)), rad2deg(median(d)), sum(ok), ...
        rad2deg(S.(sprintf('pose_r%d', r))(1, 3)), rad2deg(S.(sprintf('pose_r%d', r))(end, 3)));
end
fprintf('\nfirst 5 steps of cmd_robots_selected [vx vz w] per robot:\n'); disp(round(S.cmd_robots_selected(1:5, :), 3));
fprintf('cmd_cluster_setpoint first 3 rows: \n'); disp(round(S.cmd_cluster_setpoint(1:3, :), 3));
fprintf('NaN rows: cmd_selected %d, cluster_state %d\n', sum(any(~isfinite(S.cmd_robots_selected), 2)), sum(any(~isfinite(S.cluster_state), 2)));
end
function b = segx(a1, a2, b1, b2)
o = @(p, q, r) sign((q(1) - p(1)) * (r(2) - p(2)) - (q(2) - p(2)) * (r(1) - p(1)));
b = o(a1, a2, b1) ~= o(a1, a2, b2) && o(b1, b2, a1) ~= o(b1, b2, a2);
end
