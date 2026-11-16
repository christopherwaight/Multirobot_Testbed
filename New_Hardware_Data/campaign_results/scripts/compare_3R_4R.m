function compare_3R_4R(d, ctrue, rds)
fs = dir(fullfile(d, '*.mat'));
R = struct('N', {}, 'rd', {}, 'rad', {}, 'wob', {}, 'logr', {}, 'jit', {}, 'outl', {}, 'off', {}, 'spd', {}, 'laps', {}, 'vort', {}, 'shape', {}, 'batt', {}, 'dur', {});
for i = 1:numel(fs)
    S = load(fullfile(d, fs(i).name)); rc = S.run_config; t = S.tout; h = t > 10;
    N = 3 + isfield(S, 'pose_r4');
    k = find(endsWith(strrep({rc.constants.block}, newline, ' '), ternary(N == 3, 'Jacobian_primitive/Constant1', '/Jacobian/Constant1')), 1);
    rd = str2double(rc.constants(k).value); if ~any(abs(rds - rd) < 1e-9), continue; end
    if N == 3, cp = S.field_cp; lr = S.orbit_radius(:, 1); else, cp = S.field_cp_jacobian; lr = S.orbit_radius_jacobian(:, 1); end
    C = S.cluster_state(:, 1:2); r = vecnorm(C - ctrue, 2, 2);
    ok = all(isfinite(cp), 2) & h; cpm = median(cp(ok, :)); dv = vecnorm(cp(ok, :) - cpm, 2, 2);
    ang = unwrap(atan2(C(:, 2) - ctrue(2), C(:, 1) - ctrue(1))); i0 = find(h, 1);
    batt = inf;
    for rb = 1:N
        pose = S.(sprintf('pose_r%d', rb)); u = S.cmd_robots_selected(:, 3 * rb - 2:3 * rb - 1); v = [diff(pose(:, 1:2)) / 0.1; nan(1, 2)];
        m = (t > 5) & vecnorm(u, 2, 2) > 0.02 & vecnorm(v, 2, 2) > 0.02; batt = min(batt, median(vecnorm(v(m, :), 2, 2) ./ vecnorm(u(m, :), 2, 2)));
    end
    cs = S.cluster_state; sp = S.setpoint(1, :);
    if N == 3, se = sqrt(mean(mean((cs(h, 4:5) - sp(4:5)).^2))); else, se = sqrt(mean(mean((cs(h, 4:5) - sp(4:5)).^2))); end
    ty = S.field_jacobian(h, 8);
    R(end+1) = struct('N', N, 'rd', rd, 'rad', mean(r(h)), 'wob', std(r(h)), 'logr', mean(lr(h)), 'jit', 1.4826 * median(dv), 'outl', mean(dv > 0.1), ...
        'off', norm(cpm - ctrue), 'spd', median(vecnorm(diff(C(h, :)), 2, 2)) / 0.1, 'laps', abs(ang(end) - ang(i0)) / 2 / pi / (t(end) - t(i0)) * 10, ...
        'vort', mean(ty == 1 | ty == 2), 'shape', se, 'batt', batt, 'dur', t(end)); %#ok<AGROW>
end
fprintf('window t > 10 s; radius about true centre (%.3f, %.3f); runs with any robot speed ratio < 0.7 excluded\n\n', ctrue);
fprintf('%-5s %-3s %-9s | %-15s | %-8s | %-9s | %-21s | %-11s | %-8s | %-9s | %-9s | %s\n', 'r_d', 'N', 'runs(excl)', 'radius mean±sd', 'wobble', 'logged r', 'CP jitter / outliers', 'CP offset', 'speed', 'laps/10s', 'vortex%', 'shape err (m)');
for rd = rds
    for N = [3 4]
        a = R([R.N] == N & abs([R.rd] - rd) < 1e-9); if isempty(a), continue; end
        g = a([a.batt] >= 0.7); nx = numel(a) - numel(g); if isempty(g), g = a; end
        fprintf('%-5.2f %-3d %2d (%d)     | %.3f ± %.3f   | %.3f    | %.3f     | %.3f / %4.1f%%         | %.3f       | %.3f    | %.2f      | %3.0f%%      | %.3f\n', rd, N, numel(g), nx, ...
            mean([g.rad]), std([g.rad]), mean([g.wob]), mean([g.logr]), mean([g.jit]), 100 * mean([g.outl]), mean([g.off]), mean([g.spd]), mean([g.laps]), 100 * mean([g.vort]), mean([g.shape]));
    end
end
fprintf('\n(wobble = std of radius within a run; CP jitter = robust std of the live estimate; CP offset = run-median estimate vs true centre;\n vortex%% = steps the field fit reads as vortex or spiral; shape err = RMS of p,q (3R) or d1,d2 (4R) vs setpoint)\n');
end
function o = ternary(c, a, b), if c, o = a; else, o = b; end, end
