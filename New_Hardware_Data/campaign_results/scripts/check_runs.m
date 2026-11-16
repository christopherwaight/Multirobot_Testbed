function check_runs(dirName, modelFile)
% Check new-logging 3R runs against expectations (k_t 0.15, k_r 1, r_d 0.01 orbit of a vortex).
fs = dir(fullfile(dirName, '*.mat'));
% reference: in-file model values (to see what the unsaved edits were)
[~, mdl] = fileparts(modelFile); load_system(modelFile); c = onCleanup(@() close_system(mdl, 0));
rt = sfroot;
for i = 1:numel(fs)
    f = fullfile(dirName, fs(i).name); S = load(f); rc = S.run_config;
    n = numel(S.tout);
    fprintf('\n===== %s  (created %s, stop %g s, notes "%s")\n', fs(i).name, rc.created, rc.stop_time, rc.notes);
    % 1. rows / NaN
    newv = {'pose_r1','pose_r2','pose_r3','sensor_rgbk_raw_r1','sensor_rgbk_norm_r1','sensor_hue','sensor_sat','field_uv','field_jacobian','field_cp', ...
        'orbit_radius','cluster_state','setpoint','err_setpoint','err_jacobian','cmd_cluster_jacobian','cmd_robots_jacobian','cmd_robots_selected', ...
        'cmd_body_r1','cmd_body_r2','cmd_body_r3','simfield_hue','timing'};
    bad = {}; nn = {};
    for k = 1:numel(newv)
        if ~isfield(S, newv{k}), bad{end+1} = [newv{k} ' MISSING']; continue; end %#ok<AGROW>
        v = S.(newv{k}); if size(v, 1) ~= n, bad{end+1} = sprintf('%s rows %d', newv{k}, size(v, 1)); end %#ok<AGROW>
        if any(~isfinite(v(:))), nn{end+1} = sprintf('%s(%d)', newv{k}, sum(any(~isfinite(v), 2))); end %#ok<AGROW>
    end
    fprintf('rows: tout %d | problems: %s | rows with NaN: %s\n', n, strjoin([bad {'none'}], ', '), strjoin([nn {'none'}], ', '));
    stale = intersect(fieldnames(S), {'cluster_pose','robot1_pose','p','q','beta','hue','saturation','detected_critical_point','desired_velocities','hue_anticipated1','data','simout'});
    sr = cellfun(@(x) numrows(S.(x)), stale);
    fprintf('stale old-name vars in file: %d (rows %s)\n', numel(stale), mat2str(unique(sr)));
    % 2. config
    fprintf('model sha %s | dirty %d | switches: %s\n', rc.model_file_sha256(1:12), rc.model_dirty, strjoin(arrayfun(@(s) sprintf('%s=%s', short(s.block), num2str(s.sw)), rc.switches, 'uni', 0), ', '));
    orb = rc.matlab_functions(contains({rc.matlab_functions.block}, 'Jacobian_primitive/MATLAB Function'));
    kt = regexp(orb.code, 'k_t = [\d.]+', 'match', 'once'); kr = regexp(orb.code, 'k_r = [\d.]+', 'match', 'once');
    fprintf('Orbiter: %s, %s | direction %s | r_d %s | gains: %s\n', kt, kr, cval(rc, 'Jacobian_primitive/Constant'), cval(rc, 'Jacobian_primitive/Constant1'), ...
        strjoin(arrayfun(@(g) sprintf('%s=%s', short(g.block), g.value), rc.gains, 'uni', 0), ', '));
    % diff in-memory (run_config) vs model file on disk
    dd = {};
    for k = 1:numel(rc.constants)
        b = rc.constants(k).block;
        try, v0 = get_param(b, 'Value'); catch, v0 = '<no block>'; end
        if ~strcmp(strtrim(v0), strtrim(rc.constants(k).value)), dd{end+1} = sprintf('%s: file %s -> run %s', short(b), v0, rc.constants(k).value); end %#ok<AGROW>
    end
    for k = 1:numel(rc.matlab_functions)
        b = rc.matlab_functions(k).block; ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', b);
        if isempty(ch), dd{end+1} = sprintf('%s: not in file', short(b)); elseif ~strcmp(ch.Script, rc.matlab_functions(k).code), dd{end+1} = sprintf('%s: CODE differs', short(b)); end %#ok<AGROW>
    end
    for k = 1:numel(rc.switches)
        try, s0 = get_param(rc.switches(k).block, 'sw'); catch, s0 = '?'; end
        if ~strcmp(s0, num2str(rc.switches(k).sw)), dd{end+1} = sprintf('%s: switch file %s -> run %s', short(rc.switches(k).block), s0, num2str(rc.switches(k).sw)); end %#ok<AGROW>
    end
    fprintf('unsaved edits vs file: %s\n', strjoin([dd {'(none found)'}], ' ; '));
    % which primitive drove the robots
    P = {'setpoint','jacobian','vectorsum','ascent','descent'}; e = zeros(1, numel(P));
    for k = 1:numel(P), e(k) = max(abs(S.(['cmd_robots_' P{k}])(:) - S.cmd_robots_selected(:))); end
    [m, j] = min(e); fprintf('active primitive: %s (max diff %.2g)\n', P{j}, m);
    % 3. timing
    dtw = diff(S.timing(:, 2));
    fprintf('timing: sim dt %.3f | wall dt median %.3f, p95 %.3f, max %.3f s | run took %.1f s wall for %.1f s sim\n', median(diff(S.timing(:, 1))), median(dtw), prctile(dtw, 95), max(dtw), S.timing(end, 2) - S.timing(1, 2), S.timing(end, 1) - S.timing(1, 1));
    % 4. no spin
    fprintf('theta_c command max |.|: jacobian %.2g, setpoint %.2g\n', max(abs(S.cmd_cluster_jacobian(:, 3))), max(abs(S.cmd_cluster_setpoint(:, 3))));
    % 5. field
    ty = S.field_jacobian(:, 8); curl = S.field_jacobian(:, 7);
    fprintf('field type: vortex %.0f%% spiral %.0f%% node %.0f%% saddle %.0f%% degenerate %.0f%% | curl median %+.2f (sign + in %.0f%%)\n', ...
        100 * mean(ty == 1), 100 * mean(ty == 2), 100 * mean(ty == 3), 100 * mean(ty == -1), 100 * mean(ty == 0), median(curl, 'omitnan'), 100 * mean(curl > 0));
    cp = S.field_cp; ok = all(isfinite(cp), 2); cpm = median(cp(ok, :));
    fprintf('critical point median (%.3f, %.3f), spread std (%.3f, %.3f)\n', cpm, std(cp(ok, :)));
    % 6. orbit
    h = S.tout > 10;
    C = S.cluster_state(:, 1:2); rr = vecnorm(C - cpm, 2, 2);
    v = [diff(C) / 0.1; nan(1, 2)]; spd = vecnorm(v, 2, 2);
    fprintf('orbit (t>10 s): logged radius mean %.3f | radius about median CP %.3f | radius error mean %.3f | centroid speed median %.3f m/s (expect ~0.12)\n', ...
        mean(S.orbit_radius(h, 1), 'omitnan'), mean(rr(h)), mean(S.orbit_radius(h, 2), 'omitnan'), median(spd(h), 'omitnan'));
    % orbit direction: sign of angular velocity about CP
    ang = unwrap(atan2(C(:, 2) - cpm(2), C(:, 1) - cpm(1)));
    fprintf('orbit: %.1f revolutions, direction %s (Motive x-z angle)\n', abs(ang(end) - ang(find(h, 1))) / 2 / pi, ternary(ang(end) > ang(find(h, 1)), '+ (CCW in x-z)', '- (CW in x-z)'));
    % 7. shape
    cs = S.cluster_state; sp = S.setpoint;
    fprintf('shape (t>10): p %.3f (set %.3f), q %.3f (set %.3f), beta %.2f (set %.2f) | theta_c drift %.2f rad\n', mean(cs(h, 4)), sp(1, 4), mean(cs(h, 5)), sp(1, 5), mean(cs(h, 6)), sp(1, 6), range(unwrap(cs(:, 3))));
    % 8. command vs motion per robot: heading-zero check + cmd_body convention
    for r = 1:3
        pose = S.(sprintf('pose_r%d', r)); cg = S.cmd_robots_selected(:, 3 * r - 2:3 * r - 1); cb = S.(sprintf('cmd_body_r%d', r));
        vr = [diff(pose(:, 1:2)) / 0.1; nan(1, 2)];
        m = vecnorm(cg, 2, 2) > 0.03 & vecnorm(vr, 2, 2) > 0.02 & h;
        d = wrapToPi(atan2(vr(m, 2), vr(m, 1)) - atan2(cg(m, 2), cg(m, 1)));
        % cmd_body convention: find R(s*th) * cg ~ cb(:,1:2)
        th = pose(:, 3); best = inf; bs = '';
        for sgn = [1 -1], for flip = [1 -1]
            bx = cos(sgn * th) .* cg(:, 1) - sin(sgn * th) .* cg(:, 2); by = flip * (sin(sgn * th) .* cg(:, 1) + cos(sgn * th) .* cg(:, 2));
            res = max(abs([bx by] - cb(:, 1:2)), [], 'all'); if res < best, best = res; bs = sprintf('R(%+d*th), y*%+d', sgn, flip); end
        end, end
        fprintf('  r%d %-13s: cmd vs actual motion angle median %+5.0f deg (IQR %3.0f deg, n %d) | speed actual/cmd %.2f | cmd_body = %s (max resid %.1g)\n', ...
            r, rc.robots{r, 2}, rad2deg(median(d)), rad2deg(iqr(d)), sum(m), median(vecnorm(vr(m, :), 2, 2) ./ vecnorm(cg(m, :), 2, 2)), bs, best);
    end
    % 9. sensors
    raw = S.sensor_rgbk_raw_r1;
    fprintf('sensors: hue range %.2f-%.2f, sat range %.2f-%.2f | raw r1 col means [%s] (col 1 should be K/clear = largest)\n', min(S.sensor_hue(:)), max(S.sensor_hue(:)), min(S.sensor_sat(:)), max(S.sensor_sat(:)), num2str(round(mean(raw)), '%d '));
end
end

function n = numrows(v)
if isa(v, 'timeseries'), n = numel(v.Time); else, n = size(v, 1); end
end
function s = short(b)
s = regexprep(strrep(b, newline, ' '), '^[^/]*/', '');
end
function v = cval(rc, suffix)
k = find(endsWith({rc.constants.block}, suffix), 1); if isempty(k), v = '?'; else, v = rc.constants(k).value; end
end
function o = ternary(c, a, b)
if c, o = a; else, o = b; end
end
