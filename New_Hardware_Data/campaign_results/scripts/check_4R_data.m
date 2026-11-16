function check_4R_data(files, ctrue)
for i = 1:numel(files)
    S = load(files{i}); rc = S.run_config; n = numel(S.tout); t = S.tout;
    [~, nm] = fileparts(files{i});
    fprintf('\n===== %s | created %s | %d steps = %.1f s | StopTime %g | notes "%s"\n', nm, rc.created, n, t(end), rc.stop_time, rc.notes);
    % --- logging integrity
    v4 = {'pose_r1','pose_r2','pose_r3','pose_r4','sensor_rgbk_raw_r1','sensor_rgbk_raw_r4','sensor_rgbk_norm_r1','sensor_rgbk_norm_r4','sensor_hue','sensor_sat', ...
        'field_uv','field_jacobian','field_cp_jacobian','field_cp_jacobians','orbit_radius_jacobian','orbit_radius_jacobians','cluster_state','setpoint', ...
        'err_setpoint','err_jacobian','err_jacobians','cmd_cluster_setpoint','cmd_cluster_jacobian','cmd_cluster_jacobians','cmd_robots_setpoint', ...
        'cmd_robots_jacobian','cmd_robots_jacobians','cmd_robots_selected','cmd_body_r1','cmd_body_r2','cmd_body_r3','cmd_body_r4','diag_angle_error_jacobians','timing'};
    bad = {}; nanv = {};
    for k = 1:numel(v4)
        if ~isfield(S, v4{k}), bad{end+1} = [v4{k} ' MISSING']; continue; end %#ok<AGROW>
        x = S.(v4{k}); if size(x, 1) ~= n, bad{end+1} = sprintf('%s %d rows', v4{k}, size(x, 1)); end %#ok<AGROW>
        if any(~isfinite(x(:))), nanv{end+1} = sprintf('%s(%d rows)', v4{k}, sum(any(~isfinite(x), 2))); end %#ok<AGROW>
    end
    fprintf('LOGGING: %d/%d variables OK size | problems: %s | NaN: %s\n', numel(v4) - numel(bad), numel(v4), strjoin([bad {'none'}], ', '), strjoin([nanv {'none'}], ', '));
    fn = fieldnames(S); st = fn(cellfun(@(f) ~isstruct(S.(f)) && ~strcmp(f, 'tout') && numrows(S.(f)) ~= n, fn));
    fprintf('   other vars with a different row count (stale from earlier runs / debug): %s\n', strjoin(st', ', '));
    % --- settings
    cv = @(s) cval(rc, s);
    orbJ = rc.matlab_functions(contains({rc.matlab_functions.block}, '/Jacobian/MATLAB Function')).code;
    orbJs = rc.matlab_functions(contains({rc.matlab_functions.block}, '/Jacobians/MATLAB Function')).code;
    fprintf('SETTINGS: model sha %s dirty %d | Orbiter Jacobian %s %s, Jacobians %s %s | r_d Jacobian %s, Jacobians %s | direction %s/%s\n', ...
        rc.model_file_sha256(1:12), rc.model_dirty, rx(orbJ, 'k_t = [\d.]+'), rx(orbJ, 'k_r = [\d.]+'), rx(orbJs, 'k_t = [\d.]+'), rx(orbJs, 'k_r = [\d.]+'), ...
        cv('/Jacobian/Constant1'), cv('/Jacobians/Constant1'), cv('/Jacobian/Constant'), cv('/Jacobians/Constant'));
    fprintf('   thC_gain: %s | switches: %s\n', strjoin(arrayfun(@(g) sprintf('%s=%s', regexprep(strrep(g.block, newline, ' '), '^[^/]*/', ''), g.value), rc.gains(contains({rc.gains.block}, 'thC')), 'uni', 0), ', '), ...
        strjoin(arrayfun(@(s) sprintf('%s=%s', regexprep(strrep(s.block, newline, ' '), '^[^/]*/', ''), num2str(s.sw)), rc.switches, 'uni', 0), ', '));
    P = {'setpoint', 'jacobian', 'jacobians'}; e = zeros(1, 3);
    for k = 1:3, e(k) = max(abs(S.(['cmd_robots_' P{k}])(:) - S.cmd_robots_selected(:))); end
    [~, j] = min(e); fprintf('   active primitive: %s\n', P{j});
    dtw = diff(S.timing(:, 2));
    fprintf('TIMING: wall step median %.3f p95 %.3f max %.3f s; %.1f s wall for %.1f s sim\n', median(dtw), prctile(dtw, 95), max(dtw), S.timing(end, 2) - S.timing(1, 2), t(end));
    % --- robot order / shape
    X = zeros(n, 4, 2); for r = 1:4, X(:, r, :) = S.(sprintf('pose_r%d', r))(:, 1:2); end
    for tt = [1 n]
        c0 = squeeze(mean(X(tt, :, :), 2))'; a = zeros(1, 4);
        for r = 1:4, p = squeeze(X(tt, r, :))'; a(r) = atan2(p(2) - c0(2), p(1) - c0(1)); end
        [~, o] = sort(a);
        p1 = squeeze(X(tt, 1, :))'; p2 = squeeze(X(tt, 2, :))'; p3 = squeeze(X(tt, 3, :))'; p4 = squeeze(X(tt, 4, :))';
        fprintf('ORDER at t=%4.1f: CCW-in-x-z order %s | 1-3 & 2-4 are the diagonals: %d\n', t(tt), mat2str(o), segx(p1, p3, p2, p4));
    end
    cs = S.cluster_state; sp = S.setpoint(1, :); h = t > min(5, t(end) / 2);
    L = {'d1', 'd2', 'r1', 'r2', 'phi'};
    fprintf('SHAPE (t > %.0f s, mean [min max]) vs setpoint:\n', min(5, t(end) / 2));
    for k = 4:8, fprintf('   %-4s %6.3f [%6.3f %6.3f]  set %6.3f\n', L{k - 3}, mean(cs(h, k)), min(cs(h, k)), max(cs(h, k)), sp(k)); end
    fprintf('   theta_c drift %.2f rad; theta_c commanded max %.2g\n', range(unwrap(cs(:, 3))), max(abs([S.cmd_cluster_jacobian(:, 3); S.cmd_cluster_jacobians(:, 3); S.cmd_cluster_setpoint(:, 3)])));
    % --- commands & tracking
    fprintf('COMMANDS per robot (t > %.0f s): |cmd| median / p95, actual speed median, motion-vs-command angle median, cmd_body check\n', min(5, t(end) / 2));
    for r = 1:4
        pose = S.(sprintf('pose_r%d', r)); u = S.cmd_robots_selected(:, 3 * r - 2:3 * r - 1); v = [diff(pose(:, 1:2)) / 0.1; nan(1, 2)];
        ok = h & vecnorm(u, 2, 2) > 0.02 & vecnorm(v, 2, 2) > 0.02;
        d = angle(exp(1i * (atan2(v(ok, 2), v(ok, 1)) - atan2(u(ok, 2), u(ok, 1)))));
        th = pose(:, 3); cb = S.(sprintf('cmd_body_r%d', r));
        res = max(abs([cos(-th) .* u(:, 1) - sin(-th) .* u(:, 2), sin(-th) .* u(:, 1) + cos(-th) .* u(:, 2)] - cb(:, 1:2)), [], 'all');
        fprintf('   r%d %-13s %.3f / %.3f  actual %.3f  angle %+5.0f deg (IQR %3.0f)  cmd_body resid %.1g\n', r, rc.robots{r, 2}, median(vecnorm(u(h, :), 2, 2)), prctile(vecnorm(u(h, :), 2, 2), 95), ...
            median(vecnorm(v(h, :), 2, 2), 'omitnan'), rad2deg(median(d)), rad2deg(iqr(d)), res);
    end
    % --- field / CP / orbit
    ty = S.field_jacobian(:, 8);
    fprintf('FIELD: 4-point fit type vortex %.0f%% spiral %.0f%% node %.0f%% saddle %.0f%% | curl median %+.2f\n', 100 * mean(ty == 1), 100 * mean(ty == 2), 100 * mean(ty == 3), 100 * mean(ty == -1), median(S.field_jacobian(:, 7)));
    for nmcp = {'field_cp_jacobian', 'field_cp_jacobians'}
        cp = S.(nmcp{1}); ok = all(isfinite(cp), 2) & h;
        fprintf('   %-19s median (%.3f, %.3f), jitter std (%.3f, %.3f), %.0f%% steps > 10 cm from median | offset from true centre %.3f m\n', nmcp{1}, median(cp(ok, :)), std(cp(ok, :)), ...
            100 * mean(vecnorm(cp(ok, :) - median(cp(ok, :)), 2, 2) > 0.1), norm(median(cp(ok, :)) - ctrue));
    end
    C = cs(:, 1:2); rr = vecnorm(C - ctrue, 2, 2); ang = unwrap(atan2(C(:, 2) - ctrue(2), C(:, 1) - ctrue(1)));
    orn = {'orbit_radius_jacobian', 'orbit_radius_jacobians'}; orn = orn{j - 1 + (j == 1)};
    fprintf('ORBIT (t > %.0f s): radius about true centre %.3f (min %.3f max %.3f) | logged %s %.3f | centroid speed %.3f m/s | %.1f laps\n', min(5, t(end) / 2), ...
        mean(rr(h)), min(rr(h)), max(rr(h)), orn, mean(S.(orn)(h, 1)), median(vecnorm(diff(C(h, :)), 2, 2)) / 0.1, abs(ang(end) - ang(find(h, 1))) / 2 / pi);
    fprintf('SENSORS: hue %s | sat %s | raw col means r1 [%s] r4 [%s]\n', mat2str(round([min(S.sensor_hue); max(S.sensor_hue)], 2)), mat2str(round([min(S.sensor_sat); max(S.sensor_sat)], 2)), ...
        num2str(round(mean(S.sensor_rgbk_raw_r1)), '%d '), num2str(round(mean(S.sensor_rgbk_raw_r4)), '%d '));
    if isfield(S, 'FK_results'), fprintf('FK_results (debug): %d rows\n', size(S.FK_results, 1)); end
end
end
function s = rx(c, p), s = regexp(c, p, 'match', 'once'); end
function v = cval(rc, suffix), k = find(endsWith(strrep({rc.constants.block}, newline, ' '), suffix), 1); if isempty(k), v = '?'; else, v = rc.constants(k).value; end, end
function n = numrows(v), if isa(v, 'timeseries'), n = numel(v.Time); else, n = size(v, 1); end, end
function b = segx(a1, a2, b1, b2)
o = @(p, q, r) sign((q(1) - p(1)) * (r(2) - p(2)) - (q(2) - p(2)) * (r(1) - p(1)));
b = o(a1, a2, b1) ~= o(a1, a2, b2) && o(b1, b2, a1) ~= o(b1, b2, a2);
end
